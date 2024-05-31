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

class CVAE(nn.Module):
    def __init__(self,output_directory,epochs,device,latent_dimension=16,num_classes=5,hid_dim=16,mlp_dim=16,hid_score=16,score_dim=1,filters=128,lr=1e-4,w_kl=1e-3,w_rec=0.999):
        super(CVAE, self).__init__()
        self.output_directory = output_directory
        self.epochs = epochs
        self.latent_dimension = latent_dimension
        self.num_classes = num_classes
        self.mlp_dim = mlp_dim
        self.hid_dim = hid_dim
        self.score_dim = score_dim
        self.hid_score= hid_score
        self.filters = filters
        
        self.lr = lr
        self.w_kl =w_kl
        self.w_rec=w_rec
        self.label_mlp = nn.Sequential(
            nn.Linear(num_classes, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,mlp_dim),
            nn.ReLU(),
           
        )
        self.score_mlp = nn.Sequential(
              nn.Linear(score_dim,hid_score),
              nn.ReLU(),
              nn.Linear(hid_score,mlp_dim),
              nn.ReLU(),
        )

        self.conv1 = nn.Conv1d(54, self.filters, kernel_size=200)
        self.conv2 = nn.Conv1d(self.filters,self.filters,kernel_size=100)
        self.conv3 = nn.Conv1d(self.filters,self.filters,kernel_size=50)
        self.conv4 = nn.Conv1d(self.filters,self.filters,kernel_size=25)
        self.conv5 = nn.Conv1d(self.filters,self.filters,kernel_size=10)
        self.conv6 = nn.Conv1d(self.filters,self.filters,kernel_size=5)
        self.mu = nn.Linear(self.filters*364+mlp_dim , self.latent_dimension)
        self.log_var = nn.Linear(self.filters*364+mlp_dim, self.latent_dimension)

        self.fc = nn.Linear(self.latent_dimension+self.mlp_dim+self.mlp_dim , self.filters*364)
        self.deconv6 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=5)
        self.deconv5 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=10)
        self.deconv4 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=25)   
        self.deconv3 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=50)
        self.deconv2 = nn.ConvTranspose1d(self.filters,self.filters,kernel_size=100)
        self.deconv1 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=200)
        self.deconv0 = nn.ConvTranspose1d(self.filters,54,kernel_size=1)



    def condition_on_label(self, y):
        """ Projects one hot encoded labels to latent space.
        Args:
            y: Input labels.   """
        projected_label = self.label_mlp(y.float())
        return projected_label
    
    def condition_on_score(self,y):
          projected_score = self.score_mlp(y.float())
          return projected_score
        

    def encode(self,x,label):
            """Encodes input data into latent space.
                Args:
                x (tensor): Input data.
                label (tensor): Input labels."""
            x =x.view(x.size(0),x.size(1),-1)
            x = x.permute(0,2,1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))  
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))
            x = x.view(x.size(0), -1)
            x = torch.cat((x, self.condition_on_label(label)), dim=1) 
            mu = self.mu(x)
            log_var = self.log_var(x)
            return mu, log_var

    def reparameterize(self, mu, log_var):
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return mu + eps*std

    def decode(self,z,label,score):
            """ Decodes the latent representation to reconstruct the input data."""
            score = self.condition_on_score(score)
            label =self.condition_on_label(label)
            z = torch.cat((z, label,score), dim=1)
            z = self.fc(z)
            z = z.view(z.size(0), self.filters, 364)   
            z = F.relu(self.deconv6(z))
            z = F.relu(self.deconv5(z))
            z = F.relu(self.deconv4(z))
            z = F.relu(self.deconv3(z))
            z = F.relu(self.deconv2(z))
            z = F.relu(self.deconv1(z))
            z = torch.sigmoid(self.deconv0(z))
            z = z.permute(0,2,1)
            z = z.view(z.size(0),z.size(1),18,3)
            return z

    def forward(self,x,label,score):
            mu,logvar = self.encoder(x,label)
            z = self.reparameterize(mu,logvar)
            x_reconst = self.decoder(z,label,score)
            return x_reconst, mu, logvar
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
        mu, log_var = self.encode(x, label)
        latent_space = self.reparameterize(mu, log_var)
        reconstructed_samples = self.decode(latent_space, label, score)
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
                loss_rec_tf, loss_kl_tf, loss_tf = self.train_step(data, label, score, optimizer)
                loss_value += loss_tf
                loss_kl_value += loss_kl_tf
                loss_rec_value += loss_rec_tf
            loss.append(loss_value / len(dataloader))
            loss_kl.append(loss_kl_value / len(dataloader))
            loss_rec.append(loss_rec_value / len(dataloader))
            print(f"epoch: {epoch}, total loss: {loss[-1]}, rec loss: {loss_rec[-1]}, kl loss: {loss_kl[-1]}")
            if loss[-1] < min_loss:

                min_loss = loss[-1]
                torch.save(self.state_dict(), os.path.join(self.output_directory, 'best_model.pth'))

        # Save the final model
        torch.save(self.state_dict(), os.path.join(self.output_directory, 'last_model.pth'))

        plot_loss(self.epochs, loss, loss_rec, loss_kl,self.output_directory)
    # @staticmethod
    # def visualize_latent_space(self,dataloader,device):
    #     self.device = device
    #     self.eval()
        
    #     self.encode.load_state_dict(torch.load(self.output_directory + 'best_encoder.pth'))
    #     self.encode.to(self.device)
    #     self.encode.eval()
        
    #     with torch.no_grad(): 
    #         latent_space = []
    #         labels = []
    #         for data, batch_labels, score in dataloader:
    #             labels.extend(batch_labels)
    #             data = data.to(self.device)
    #             batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    #             batch_labels = batch_labels.to(self.device)
    #             batch_labels = F.one_hot(batch_labels, num_classes=self.num_classes)
    #             mu, var = self.encode(data,batch_labels)  
    #             latent_space.append(mu.cpu().numpy())
    #         latent_space = np.vstack(latent_space)
    #         labels = [int(label.item()) for label in labels]
    #     tsne = TSNE(n_components=2, random_state=42)
    #     latent_2d = tsne.fit_transform(latent_space)
    #     plot_latent_space(latent_2d, labels, "2D Visualization of Latent Space using TSNE ",self.output_directory)
    #     pca = PCA(n_components=2, random_state=42)
    #     latent_2d = pca.fit_transform(latent_space)
    #     plot_latent_space(latent_2d, labels, "2D Visualization of Latent Space using PCA ",self.output_directory)
    # # def generate_samples():


#ADD MEAN AND VARIANCE 2D VISUALIZATION.

    def visualize_latent_space(self, dataloader, device):
        self.device = device
        self.to(device)
        self.eval()
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
                mu, _ = self.encode(data, batch_labels)
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
        sample = torch.randn(1, latent_dimension).to(device)
        score = torch.tensor([score_value * 100]).unsqueeze(1).to(device)
        c = torch.eye(num_classes)[label].unsqueeze(0).to(device)
        with torch.no_grad():
                generated_sample = model.decode(sample, c, score).cpu().double().numpy()
        np.save('generated_sample.npy', generated_sample)
        return generated_sample
    
    def generate_samples(self,device,class_index):
        sample = torch.randn(1, self.latent_dimension).to(device)
        score_values = [0, 0.1 ,0.2 ,0.25 ,0.3 ,0.388 ,0.4 ,0.5 ,0.55 ,0.6 ,0.677 ,0.7 ,0.8 ,0.833 ,0.85 ,0.875 ,0.9 ,0.955 ,0.97 ,1.0]  
        generated_samples = []
        generated_scores = []

        for score_value in score_values:
            score = torch.tensor([score_value*100]).unsqueeze(1).to(device)
            c = torch.eye(self.num_classes)[class_index].unsqueeze(0).to(device)

            with torch.no_grad():
                generated_sample = self.decode(sample, c, score).cpu().double().numpy()
                unnormalized_sample = unnormalize_generated_skeletons(generated_sample)
                plot_skel(unnormalized_sample,self.output_directory,title='exercice'+str(class_index)+'_score='+str(score_value))

            generated_samples.append(generated_sample)

        generated_samples_array = np.array(generated_samples)
        scores_array = np.array(score_values)
   
            

 
        np.save(os.path.join(self.output_directory,'generated_samples.npy'), generated_samples_array)
        np.save(os.path.join(self.output_directory,'true_scores.npy'), scores_array)





