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
sys.path.append('/home/hferrar/REHABProject/model/utils')
from utils.plot import plot_loss, plot_latent_space
from utils.normalize import unnormalize_generated_skeletons
from utils.visualize import plot_skel
from dataset.dataset import load_class
from sklearn.mixture import GaussianMixture
class MotionEncoder(nn.Module):
    def __init__(self, filters, mlp_dim, latent_dimension, hid_dim, num_classes):
        super(MotionEncoder,self).__init__()
        self.filters = filters
        self.latent_dimension=latent_dimension
        self.label_mlp = nn.Sequential(
            nn.Linear(num_classes, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, mlp_dim),
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
    def forward(self, x, label):
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, self.label_mlp(label.float())), dim=1)
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
        self.label_mlp = nn.Sequential(
            nn.Linear(num_classes, hid_score),
            nn.ReLU(),
            nn.Linear(hid_score, mlp_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(latent_dimension + mlp_dim + mlp_dim, filters * 606)
        self.deconv6 = nn.ConvTranspose1d(filters, filters, kernel_size=3)
        self.deconv5 = nn.ConvTranspose1d(filters, filters, kernel_size=5)
        self.deconv4 = nn.ConvTranspose1d(filters, filters, kernel_size=10)
        self.deconv3 = nn.ConvTranspose1d(filters, filters, kernel_size=20)
        self.deconv2 = nn.ConvTranspose1d(filters, filters, kernel_size=40)
        self.deconv1 = nn.ConvTranspose1d(filters, filters, kernel_size=70)
        self.deconv0 = nn.ConvTranspose1d(filters, 54, kernel_size=1)

    def forward(self, z, label, score):
        score = self.score_mlp(score.float())
        label = self.label_mlp(label.float())
        z = torch.cat((z, label, score), dim=1)
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

class CVAEE(nn.Module):
    def __init__(self, output_directory, epochs, device, latent_dimension=16, num_classes=5, hid_dim=16, mlp_dim=16, hid_score=16, score_dim=1, filters=128, lr=1e-4, w_kl=1e-3, w_rec=0.999):
        super(CVAEE, self).__init__()
        self.output_directory = output_directory
        self.epochs = epochs
        self.latent_dimension = latent_dimension
        self.num_classes = num_classes
        self.filters = filters
        self.lr = lr
        self.w_kl = w_kl
        self.w_rec = w_rec
        self.encoder = MotionEncoder(filters, mlp_dim, latent_dimension, hid_dim, num_classes)
        self.decoder = MotionDecoder(filters, mlp_dim, latent_dimension, hid_score, score_dim, num_classes)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, label, score):
        mu, log_var = self.encoder(x, label)
        z = self.reparameterize(mu, log_var)
       
        x_reconst = self.decoder(z, label, score)
        return x_reconst, mu, log_var

    @staticmethod
    def kl_loss(mu,log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1)
    @staticmethod
    def mse_loss(reconstructed_x,x):
        return torch.mean(torch.sum((reconstructed_x - x) ** 2, axis=1),axis=[1,2])

    def train_step(self, x, label, score, optimizer):
        self.train()
        x = x.to(self.device)
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(self.device)
        score = score.to(self.device)
        label = F.one_hot(label, num_classes=self.num_classes)
        optimizer.zero_grad()
        mu, log_var = self.encoder(x, label)
        latent_space = self.reparameterize(mu, log_var)
      
        reconstructed_samples = self.decoder(latent_space, label, score)
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
            for batch_idx, (data, label, score) in enumerate(dataloader):
                loss_rec_tf, loss_kl_tf, loss_tf= self.train_step(data, label, score, optimizer)
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


#ADD MEAN AND VARIANCE 2D VISUALIZATION.
    def visualize_latent_space(self, dataloader, device):
        self.device = device
        self.to(device)
        self.encoder.load_state_dict(torch.load(self.output_directory + 'last_encoder.pth', map_location=device))
        self.encoder.eval()
        with torch.no_grad():
            latent_space = []
            labels = []
            for data, batch_labels, score in dataloader:
                labels.extend(batch_labels)
                data = data.to(device)
                score = score.to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long)
                batch_labels = batch_labels.to(device)
                batch_labels = F.one_hot(batch_labels, num_classes=self.num_classes)
                mu, _ = self.encoder(data, batch_labels)
                latent_space.append(mu.cpu().numpy())
            latent_space = np.vstack(latent_space)
            labels = [int(label.item()) for label in labels]
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_space)
        plot_latent_space(latent_2d, labels, "2D Visualization of Latent Space using TSNE ",self.output_directory)
        pca = PCA(n_components=2, random_state=42)
        latent_2d = pca.fit_transform(latent_space)
        plot_latent_space(latent_2d, labels, "2D Visualization of Latent Space using PCA ",self.output_directory)
    

    def generate_skeleton(self, device,score, label):

        self.device =device
        self.to(device)
        
        self.decoder.load_state_dict(torch.load(self.output_directory + 'best_decoder.pth', map_location=device))
        
        self.encoder.eval()
        self.decoder.eval()
        sample = torch.randn(1, latent_dimension).to(device)
        score = torch.tensor([score_value ]).unsqueeze(1).to(device)
        c = torch.eye(num_classes)[label].unsqueeze(0).to(device)
        with torch.no_grad():
                generated_sample = self.decoder(sample, c, score).cpu().double().numpy()
        np.save('generated_sample.npy', generated_sample)
        return generated_sample
    

    def generate_samples(self,device,class_index,gif_directory,dataloader):
        n_components = 1

        gmm = GaussianMixture(n_components=n_components)
        self.device =device
        self.to(device)
        self.decoder.load_state_dict(torch.load(self.output_directory + 'last_decoder.pth', map_location=device))
        encoder = self.encoder.load_state_dict(torch.load(self.output_directory + 'last_encoder.pth', map_location=device))
        z_list = []
        for data, label, score_value in dataloader: 
            data = data.to(device)
            label = torch.tensor(label, dtype=torch.long).to(device)
            label = F.one_hot(label, num_classes=self.num_classes)
            score = score_value.to(device) 
            with torch.no_grad():
                z_mu, z_logvar = self.encoder(data,label)
                z = self.reparameterize(z_mu,z_logvar)
                z_list.append(z)
        z_all = torch.cat(z_list, dim=0).cpu().numpy()
        # mean_z = torch.mean(z_all, dim=0, keepdim=True).to(device)
        gm = gmm.fit(z_all)
        means = np.asarray(gm.means_).reshape((self.latent_dimension,))
        covariances = np.asarray(gm.covariances_).reshape((self.latent_dimension, self.latent_dimension))
        
        sample = np.random.multivariate_normal(mean=means,cov=covariances, size=1)
        sample = torch.tensor(sample,dtype=torch.float32).to(device)
        # sample = torch.randn(1, self.latent_dimension).to(device)
        ex1_scores =[60.0, 90.83333, 100.0, 90.0, 84.16667, 76.66667, 97.5, 80.0, 100.0, 82.5, 92.5, 
                    77.5, 100.0, 75.0, 92.5, 75.0, 80.0, 97.5, 92.5, 74.16667, 85.83333, 77.5, 
                    70.0, 87.5, 90.0, 81.66667, 85.0, 100.0, 100.0, 100.0, 90.0, 86.66667,92.5,
                    95.83333, 90.83333, 95.83333, 98.33333, 100.0, 100.0, 91.66667, 40.0, 40.83333, 41.66667, 85.0, 
                    42.5, 59.16667, 65.0, 71.02688, 60.0, 75.0,57.5, 42.5, 44.16667, 95.0, 65.0, 
                    10.0, 100.0, 11.66667, 55.0, 66.66667, 60.77045, 100.0, 67.5, 64.16667, 77.5, 58.34933, 
                    70.0, 75.83333, 85.83333, 88.33333, 45.5252]
        generated_samples = []
        generated_samples_unnormalized =[]
        generated_scores = []
        for score_value in ex1_scores:
            score = torch.tensor([score_value/100.0]).unsqueeze(1).to(device)
            c = torch.eye(self.num_classes)[class_index].unsqueeze(0).to(device)
            with torch.no_grad():
                generated_sample = self.decoder(sample, c, score).cpu().double().numpy()
                unnormalized_sample = unnormalize_generated_skeletons(generated_sample)
                # plot_skel(unnormalized_sample,gif_directory,title='exercice'+str(class_index)+'_score='+str(score_value))
            generated_samples.append(generated_sample)
            generated_samples_unnormalized.append(unnormalized_sample)
        generated_samples_array = np.array(generated_samples)
        generated_samples_unnormalized = np.array(generated_samples_unnormalized)
        scores_array = np.array(ex1_scores)
 
        np.save(os.path.join(gif_directory,'generated_samples_unnormalized.npy'), generated_samples_unnormalized)
        np.save(os.path.join(gif_directory,'generated_samples.npy'), generated_samples_array)
        np.save(os.path.join(gif_directory,'true_scores.npy'), scores_array)
  




















# # Find the component with the highest probability for each sample
# labels = gmm.predict(z_all)

# # Compute the mean or mode of the component with the highest probability
# summary_index = labels.argmax()  # Find the index of the component with the highest probability
# summary_z = gmm.means_[summary_index]









    # def generate_samples(self, device, class_index, gif_directory, dataloader):
    #     self.device = device
    #     self.to(device)
    #     self.decoder.load_state_dict(torch.load(self.output_directory + 'last_decoder.pth', map_location=device))
    #     encoder = self.encoder.load_state_dict(torch.load(self.output_directory + 'last_encoder.pth', map_location=device))
    #     generated_samples = []
    #     generated_samples_unnormalized = []
    #     generated_scores = []
    #     ex1_scores = [60.0, 90.83333, 100.0, 90.0, 84.16667, 76.66667, 97.5, 80.0, 100.0, 82.5, 92.5, 77.5, 100.0, 75.0, 92.5, 
    #      75.0, 80.0, 97.5, 92.5, 74.16667, 85.83333, 77.5, 70.0, 87.5, 90.0, 81.66667, 85.0, 100.0, 100.0, 100.0, 90.0, 86.66667,
    #       92.5, 95.83333, 90.83333, 95.83333, 98.33333, 100.0, 100.0, 91.66667, 40.0, 40.83333, 41.66667, 85.0, 42.5, 59.16667, 65.0, 71.02688, 60.0, 75.0,
    #       57.5, 42.5, 44.16667, 95.0, 65.0, 10.0, 100.0, 11.66667, 55.0, 66.66667, 60.77045, 100.0, 67.5, 64.16667, 77.5, 58.34933, 70.0, 75.83333, 85.83333, 88.33333, 45.5252]
    #     for data, label, score_value in dataloader: 
    #         data = data.to(device)
    #         label = torch.tensor(label, dtype=torch.long).to(device)
    #         label = F.one_hot(label, num_classes=self.num_classes)
    #         score = score_value.to(device) 
           
    #         with torch.no_grad():
    #             z_mu, z_logvar = self.encoder(data,label)
    #             z = self.reparameterize(z_mu,z_logvar)
    #     c = torch.eye(self.num_classes)[class_index].unsqueeze(0).to(device)
    #     for score_value in ex1_scores:
    #         score = torch.tensor([score_value/100.00]).unsqueeze(1).to(device)
    #         print(score.dtype)
    #         with torch.no_grad():
    #             generated_sample = self.decoder(z, c, score_value).cpu().double().detach().numpy()
    #             unnormalized_sample = unnormalize_generated_skeletons(generated_sample)
    #                     # plot_skel(unnormalized_sample,gif_directory,title='exercice'+str(class_index)+'_score='+str(score_value))

    #             generated_samples.append(generated_sample)
    #             generated_samples_unnormalized.append(unnormalized_sample)
    #             generated_scores.append(score_value)

    #     generated_samples_array = np.concatenate(generated_samples, axis=0)
    #     generated_samples_unnormalized = np.concatenate(generated_samples_unnormalized, axis=0)
    #     scores_array = np.array(generated_scores)
        
    #     np.save(os.path.join(gif_directory,'generated_samples_unnormalized.npy'), generated_samples_unnormalized)
    #     np.save(os.path.join(gif_directory,'generated_samples.npy'), generated_samples_array)
    #     np.save(os.path.join(gif_directory,'true_scores.npy'), scores_array)


