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

sys.path.append('../utils')
from utils.plot import plot_loss, plot_latent_space
from utils.normalize import unnormalize_generated_skeletons
from utils.visualize import plot_skel
from dataset.dataset import load_class
from sklearn.mixture import GaussianMixture

class MotionEncoder(nn.Module):
    def __init__(self, filters, latent_dimension):
        super(MotionEncoder,self).__init__()
        self.filters = filters
        self.latent_dimension=latent_dimension

        self.conv1 = nn.Conv1d(54, self.filters, kernel_size=70)
        self.conv2 = nn.Conv1d(self.filters,self.filters,kernel_size=40)
        self.conv3 = nn.Conv1d(self.filters,self.filters,kernel_size=20)
        self.conv4 = nn.Conv1d(self.filters,self.filters,kernel_size=10)
        self.conv5 = nn.Conv1d(self.filters,self.filters,kernel_size=5)
        self.conv6 = nn.Conv1d(self.filters,self.filters,kernel_size=3)
        self.mu = nn.Linear(self.filters*606 , self.latent_dimension)
        self.log_var = nn.Linear(self.filters*606, self.latent_dimension)
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
    
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var
class MotionDecoder(nn.Module):
    def __init__(self, filters, latent_dimension):
        super(MotionDecoder,self).__init__()
        self.filters = filters
        self.fc = nn.Linear(latent_dimension  , filters * 606)
        self.deconv6 = nn.ConvTranspose1d(filters, filters, kernel_size=3)
        self.deconv5 = nn.ConvTranspose1d(filters, filters, kernel_size=5)
        self.deconv4 = nn.ConvTranspose1d(filters, filters, kernel_size=10)
        self.deconv3 = nn.ConvTranspose1d(filters, filters, kernel_size=20)
        self.deconv2 = nn.ConvTranspose1d(filters, filters, kernel_size=40)
        self.deconv1 = nn.ConvTranspose1d(filters, filters, kernel_size=70)
        self.deconv0 = nn.ConvTranspose1d(filters, 54, kernel_size=1)

    def forward(self, z):
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

class VAE(nn.Module):
    def __init__(self, output_directory, epochs, device, latent_dimension=16,filters=128, lr=1e-4, w_kl=1e-3, w_rec=0.999):
        super(VAE, self).__init__()
        self.output_directory = output_directory
        self.epochs = epochs
        self.latent_dimension = latent_dimension
     
        self.filters = filters
        self.lr = lr
        self.w_kl = w_kl
        self.w_rec = w_rec
        self.encoder = MotionEncoder(filters, latent_dimension)
        self.decoder = MotionDecoder(filters, latent_dimension)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
       
        x_reconst = self.decoder(z)
        return x_reconst, mu, log_var

    @staticmethod
    def kl_loss(mu,log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1)
    @staticmethod
    def mse_loss(reconstructed_x,x):
        return torch.mean(torch.sum((reconstructed_x - x) ** 2, axis=1),axis=[1,2])

    def train_step(self, x, optimizer):
        self.train()
        x = x.to(self.device)
       
        optimizer.zero_grad()
        mu, log_var = self.encoder(x)
        latent_space = self.reparameterize(mu, log_var)
      
        reconstructed_samples = self.decoder(latent_space)
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
            for batch_idx, (data, _, _) in enumerate(dataloader):
                loss_rec_tf, loss_kl_tf, loss_tf= self.train_step(data, optimizer)
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



    def generate_skeleton(self, device):

        self.device =device
        self.to(device)
        
        self.decoder.load_state_dict(torch.load(self.output_directory + 'best_decoder.pth', map_location=device))
        
        self.encoder.eval()
        self.decoder.eval()
        sample = torch.randn(1, latent_dimension).to(device)
       
       
        with torch.no_grad():
                generated_sample = self.decoder(sample).cpu().double().numpy()
        np.save('generated_sample.npy', generated_sample)
        return generated_sample



    def generate_samples_from_posterior(self, device, gif_directory, dataloader):
        self.device = device
        self.to(device)
        self.decoder.load_state_dict(torch.load(self.output_directory + 'last_decoder.pth', map_location=device))
        encoder = self.encoder.load_state_dict(torch.load(self.output_directory + 'last_encoder.pth', map_location=device))
        generated_samples = []
        generated_samples_unnormalized = []
        
        true_samples =[]
       
        for data, _, _ in dataloader: 
            data = data.to(device)
            with torch.no_grad():
                z_mu, z_logvar = self.encoder(data)
                z = self.reparameterize(z_mu,z_logvar)
                generated_sample = self.decoder(z ).cpu().double().detach().numpy()
        generated_samples_array = np.concatenate(generated_samples, axis=0)
        # generated_samples_unnormalized = np.concatenate(generated_samples_unnormalized, axis=0)
        true_samples = np.concatenate(true_samples,axis=0)
      
        # np.save(os.path.join(gif_directory,'generated_samples_unnormalized.npy'), generated_samples_unnormalized)
        np.save(os.path.join(gif_directory,'generated_samples.npy'), generated_samples_array)
        np.save(os.path.join(gif_directory,f'true_samples_{class_index}.npy'), true_samples)
      


    def generate_samples_from_prior(self, device, gif_directory,num_samples):
        self.device = device
        self.to(device)
        print(self.output_directory + 'last_decoder.pth')
        self.decoder.load_state_dict(torch.load(self.output_directory + 'last_decoder.pth', map_location=device))
        generated_samples = []
        generated_samples_unnormalized = []
        for i in range(num_samples):
            with torch.no_grad():
                sample = torch.randn(1,self.latent_dimension).to(device)
                generated_sample = self.decoder(sample ).cpu().double().detach().numpy()
             
        generated_samples_array = np.concatenate(generated_samples, axis=0)
        np.save(os.path.join(gif_directory,'generated_samples_prior.npy'), generated_samples_array)
        return generated_samples_array
      

