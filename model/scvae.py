import os
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.autograd import Variable
from sklearn.decomposition import PCA
sys.path.append('..')
from utils.plot import plot_loss, plot_latent_space
from utils.normalize import unnormalize_generated_skeletons
from utils.visualize import plot_skel
from dataset.dataset import load_class
from sklearn.mixture import GaussianMixture
class MotionEncoder(nn.Module):
    def __init__(self, filters, mlp_dim, latent_dimension, hid_score, score_dim, num_classes):
        super(MotionEncoder,self).__init__()
        self.filters = filters
        self.latent_dimension=latent_dimension
        self.score_mlp = nn.Sequential(
            nn.Linear(score_dim, hid_score),
            nn.ReLU(),
            nn.Linear(hid_score, mlp_dim),
            nn.ReLU(),
        )
 
        self.conv1 = nn.Conv1d(54, self.filters, kernel_size=70)
        self.conv2 = nn.Conv1d(self.filters,self.filters,kernel_size=40)
        self.conv3 = nn.Conv1d(self.filters,self.filters,kernel_size=20)
        self.conv4 = nn.Conv1d(self.filters,self.filters,kernel_size=10)
        self.conv5 = nn.Conv1d(self.filters,self.filters,kernel_size=5)
        self.conv6 = nn.Conv1d(self.filters,self.filters,kernel_size=3)
        self.mu = nn.Linear(self.filters*606+mlp_dim , self.latent_dimension)
        self.log_var = nn.Linear(self.filters*606+mlp_dim, self.latent_dimension)
    def forward(self, x,score):
        score = self.score_mlp(score.float())
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x,score), dim=1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var
class MotionDecoder(nn.Module):
    def __init__(self, filters, mlp_dim, latent_dimension, hid_score, score_dim, num_classes):
        super(MotionDecoder,self).__init__()
        self.filters = filters
        self.score_mlp = nn.Sequential(
            nn.Linear(score_dim, hid_score),
            nn.ReLU(),
            nn.Linear(hid_score, mlp_dim),
            nn.ReLU(),
        )

        self.fc = nn.Linear(latent_dimension  + mlp_dim, filters * 606)
        self.deconv6 = nn.ConvTranspose1d(filters, filters, kernel_size=3)
        self.deconv5 = nn.ConvTranspose1d(filters, filters, kernel_size=5)
        self.deconv4 = nn.ConvTranspose1d(filters, filters, kernel_size=10)
        self.deconv3 = nn.ConvTranspose1d(filters, filters, kernel_size=20)
        self.deconv2 = nn.ConvTranspose1d(filters, filters, kernel_size=40)
        self.deconv1 = nn.ConvTranspose1d(filters, filters, kernel_size=70)
        self.deconv0 = nn.ConvTranspose1d(filters, 54, kernel_size=1)

    def forward(self, z, score):
        score = self.score_mlp(score.float())
       

        z = torch.cat((z, score), dim=1)
        z = self.fc(z)
        z = z.view(z.size(0), self.filters, 606)
        z = F.relu(self.deconv6(z))
        z = F.relu(self.deconv5(z))
        z = F.relu(self.deconv4(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv1(z))
        z = torch.sigmoid(self.deconv0(z))
        z = z.permute(0, 2, 1)
        z = z.view(z.size(0), z.size(1), 18, 3)
        return z

class SCVAE(nn.Module):
    """This SCVAE is conditioned only with the score,it is trained on each class separately"""
    def __init__(self, output_directory, device,epochs=2000, latent_dimension=256, num_classes=5, hid_dim=16, mlp_dim=16, hid_score=16, score_dim=1, filters=128, lr=1e-4, w_kl=1e-3, w_rec=0.999):
        super(SCVAE, self).__init__()
        self.output_directory = output_directory
        self.epochs = epochs
        self.latent_dimension = latent_dimension
        self.num_classes = num_classes
        self.filters = filters
        self.lr = lr
        self.w_kl = w_kl
        self.w_rec = w_rec
        self.encoder = MotionEncoder(filters, mlp_dim, latent_dimension, hid_score, score_dim, num_classes)
        self.decoder = MotionDecoder(filters, mlp_dim, latent_dimension, hid_score, score_dim, num_classes)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, score):
        mu, log_var = self.encoder(x,score)
        z = self.reparameterize(mu, log_var)
       
        x_reconst = self.decoder(z, score)
        return x_reconst, mu, log_var

    @staticmethod
    def kl_loss(mu,log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1)
    @staticmethod
    def mse_loss(reconstructed_x,x):
        return torch.mean(torch.sum((reconstructed_x - x) ** 2, axis=1),axis=[1,2])

    def train_step(self, x, score, optimizer):
        self.train()
        x = x.to(self.device)
     
        score = score.to(self.device)
        
        optimizer.zero_grad()
        mu, log_var = self.encoder(x,score)
        latent_space = self.reparameterize(mu, log_var)
      
        reconstructed_samples = self.decoder(latent_space, score)
        loss_rec = self.mse_loss(reconstructed_samples, x)
        loss_kl = self.kl_loss(mu, log_var)
        total_loss = torch.mean(self.w_rec * loss_rec + self.w_kl * loss_kl)
        total_loss.backward()
        optimizer.step()
        return loss_rec.mean().item(), loss_kl.mean().item(), total_loss.item()
    def train_function(self, dataloader, device):
        self.device = device
        self.to(device)
        loss = []
        loss_kl = []
        loss_rec = []
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        min_loss = float('inf')
        for epoch in range(self.epochs):
            loss_value = 0.0
            loss_kl_value = 0.0
            loss_rec_value = 0.0
            for batch_idx, (data, _, score) in enumerate(dataloader):
                loss_rec_tf, loss_kl_tf, loss_tf= self.train_step(data, score, optimizer)
                loss_value += loss_tf
                loss_kl_value += loss_kl_tf
                loss_rec_value += loss_rec_tf
                
            loss.append(loss_value / len(dataloader))
            loss_kl.append(loss_kl_value / len(dataloader))
            loss_rec.append(loss_rec_value / len(dataloader))
            print(f"epoch: {epoch}, total loss: {loss[-1]}, rec loss: {loss_rec[-1]}, kl loss: {loss_kl[-1]}")
            if loss[-1] < min_loss:
                min_loss = loss[-1]
                torch.save(self.encoder.state_dict(), os.path.join(self.output_directory, 'best_encoder.pth'))
                torch.save(self.decoder.state_dict(), os.path.join(self.output_directory, 'best_decoder.pth'))
                

        torch.save(self.encoder.state_dict(), os.path.join(self.output_directory, 'last_encoder.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(self.output_directory, 'last_decoder.pth'))
        plot_loss(self.epochs, loss, loss_rec, loss_kl,self.output_directory)



    def generate_skeleton(self, device,score):

        self.device =device
        self.to(device)
        
        self.decoder.load_state_dict(torch.load(self.output_directory + 'best_decoder.pth', map_location=device))
        
        self.encoder.eval()
        self.decoder.eval()
        sample = torch.randn(1, latent_dimension).to(device)
        score = torch.tensor([score_value ]).unsqueeze(1).to(device)
     
        with torch.no_grad():
                generated_sample = self.decoder(sample, score).cpu().double().numpy()
        np.save('generated_sample.npy', generated_sample)
        return generated_sample
    
      
    # def generate_samples_from_posterior(self, device, class_index, gif_directory, dataloader):
    #         self.device = device
    #         self.to(device)
    #         self.decoder.load_state_dict(torch.load(self.output_directory + 'last_decoder.pth', map_location=device))
    #         self.encoder.load_state_dict(torch.load(self.output_directory + 'last_encoder.pth', map_location=device))
    #         print(self.output_directory + 'last_encoder.pth')
    #         print(self.output_directory + 'last_decoder.pth')

    #         generated_samples = []
    #         generated_samples_unnormalized = []
    #         scores = []
    #         true_samples = []

    #         for data, label, score_value in dataloader:
    #             data = data.to(device)
    #             label = label.to(device)
    #             label_enc = torch.tensor(label, dtype=torch.long).to(device)
    #             label_enc = F.one_hot(label_enc, num_classes=self.num_classes)
    #             score = score_value.to(device)
    #             z_mu, z_logvar = self.encoder(data, label_enc)
    #             z = self.reparameterize(z_mu, z_logvar)
    #             generated_sample = self.decoder(z, label_enc, score).cpu().detach().numpy()
    #             generated_samples.append(generated_sample)
    #             true_samples.append(data.cpu().detach().numpy())  
    #             scores.append(score_value * 100.0)
    #             unnormalized_sample = unnormalize_generated_skeletons(generated_sample)
    #             plot_skel(unnormalized_sample,gif_directory,title='post_'+'score='+str(score))
           
    #         generated_samples_array = np.concatenate(generated_samples, axis=0)
          
    #         # generated_samples_unnormalized = np.concatenate(generated_samples_unnormalized, axis=0)
    #         true_samples = np.concatenate(true_samples, axis=0)
    #         scores = np.array(scores)

    #         # Save generated samples, unnormalized samples, true samples, and scores
    #         # np.save(os.path.join(gif_directory, 'generated_samples_unnormalized.npy'), generated_samples_unnormalized)
    #         np.save(os.path.join(gif_directory, 'generated_samples.npy'), generated_samples_array)
    #         np.save(os.path.join(gif_directory, 'scores.npy'), scores)
    #         np.save(os.path.join(gif_directory, f'true_samples_{class_index}.npy'), true_samples)

    def generate_samples_from_prior(self, device, gif_directory,dataloader):
        self.device = device
        self.to(device)
        self.decoder.load_state_dict(torch.load(self.output_directory + 'best_decoder.pth', map_location=device))
       
        generated_samples = []
        scores = []
        scores = np.array(scores)

        for data, _, score in dataloader:
            scores=np.append(scores,score * 100.0)
        for score in np.unique(scores):
            with torch.no_grad():
                sample = torch.randn(1,self.latent_dimension).to(device)
                score = torch.tensor([score/100]).unsqueeze(1).to(device)
           
                generated_sample = self.decoder(sample, score).cpu().double().detach().numpy()
                generated_samples.append(generated_sample)
                # unnormalized_sample = unnormalize_generated_skeletons(generated_sample)
             
                
        generated_samples_array = np.concatenate(generated_samples, axis=0)
        np.save(os.path.join(gif_directory,f'generated_samples_prior.npy'), generated_samples_array)
        np.save(os.path.join(gif_directory, 'scores.npy'), scores)
    
        return generated_samples_array,scores
        

    
    def evaluate_function(self, dataloader, device):
        self.device = device
        self.to(device)
        
        # Load the encoder and decoder models for the specific fold
        self.encoder.load_state_dict(torch.load(os.path.join(self.output_directory, f'last_encoder.pth'), map_location=device))
        self.encoder.eval()
        self.decoder.load_state_dict(torch.load(os.path.join(self.output_directory, f'last_decoder.pth'), map_location=device))
        self.decoder.eval()

        total_loss = 0.0
        total_loss_rec = 0.0
        total_loss_kl = 0.0
        with torch.no_grad():
            for batch_idx, (data, _, score) in enumerate(dataloader):
                data = data.to(device)
               
                score = score.to(device)

                reconstructed_samples, mu, log_var = self.forward(data, score)
                loss_rec = self.mse_loss(reconstructed_samples, data)
                loss_kl = self.kl_loss(mu, log_var)

                total_loss_rec += loss_rec.mean().item()
                total_loss_kl += loss_kl.mean().item()

        total_loss_rec /= len(dataloader)
        total_loss_kl /= len(dataloader)
        total_loss = self.w_rec * total_loss_rec + self.w_kl * total_loss_kl

        return total_loss

        



