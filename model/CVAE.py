import os
import sys
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

class CVAE(nn.Module):
    def __init__(self,output_directory,latent_dimension=16,num_classes=5,hid_dim=16,mlp_dim=16,hid_score=16,score_dim=1,filters=128,epochs=500,lr=1e-3,w_kl=1e-3,w_rec=0.999):
        super(CVAE, self).__init__()
        self.output_directory = output_directory
        
        self.latent_dimension = latent_dimension
        self.num_classes = num_classes
        self.mlp_dim = mlp_dim
        self.hid_dim = hid_dim
        self.score_dim = score_dim
        self.hid_score= hid_score
        self.filters = filters
        self.epochs = epochs
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

        self.conv1 = nn.Conv1d(54, self.filters, kernel_size=60)
        self.conv2 = nn.Conv1d(self.filters,self.filters,kernel_size=30)
        self.conv3 = nn.Conv1d(self.filters,self.filters,kernel_size=20)
        self.conv4 = nn.Conv1d(self.filters,self.filters,kernel_size=11)
        self.conv5 = nn.Conv1d(self.filters,self.filters,kernel_size=7)
        self.conv6 = nn.Conv1d(self.filters,self.filters,kernel_size=3)
        self.mu = nn.Linear(self.filters*623+mlp_dim , self.latent_dimension)
        self.log_var = nn.Linear(self.filters*623+mlp_dim, self.latent_dimension)
        self.fc = nn.Linear(self.latent_dimension+self.mlp_dim+self.mlp_dim , self.filters*623)
        self.deconv6 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=3)
        self.deconv5 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=7)
        self.deconv4 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=11)   
        self.deconv3 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=20)
        self.deconv2 = nn.ConvTranspose1d(self.filters,self.filters,kernel_size=30)
        self.deconv1 = nn.ConvTranspose1d(self.filters,self.filters, kernel_size=60)
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
            z = self.fc3(z)
            z = z.view(z.size(0), self.filters, 623)   
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
            mu,logvar = self.encode(x,label)
            z = self.reparameterize(mu,logvar)
            x_reconst = self.decode(z,label,score)
            return x_reconst, mu, logvar
    
    def kl_loss(mu,log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1)

    def mse_loss(reconstructed_x,x):
        return torch.mean(torch.sum((reconstructed_x - x) ** 2, axis=1),axis=[1,2])

    def train_step(self, x, label, score, optimizer):
        self.train()
        x = x.to(device)
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(device)
        score = score.to(device)
        label= F.one_hot(label, num_classes=self.num_classes)
        optimizer.zero_grad()
        mu, log_var = self.encode(x, label)
        latent_space = self.reparameterize(mu, log_var)
        reconstructed_samples = self.decode(latent_space, label, score)
        loss_rec = self.mse_loss(reconstructed_samples, x)
        loss_kl = self.kl_loss(mu, log_var)
        total_loss = torch.mean(self.w_rec*loss_rec + self.w_kl*loss_kl)
        total_loss.backward()
        optimizer.step()
        return loss_rec.mean().item(), loss_kl.mean().item(), total_loss.mean().item()
    
    def train(self,dataloader):
        loss = []
        loss_kl = []
        loss_rec = []
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        min_loss = float('inf')
        for epoch in range(self.epochs):
            loss_value = 0.0
            loss_kl_value = 0.0
            loss_rec_value = 0.0
            for data, labels,score in enumerate(dataloader): 
                loss_rec_tf, loss_kl_tf, loss_tf = self.train_step(data, label, score, optimizer)
                loss_value += loss_tf
                loss_kl_value += loss_kl_tf
                loss_rec_value += loss_rec_tf
            loss.append(loss_value / number_batches)
            loss_kl.append(loss_kl_value / number_batches)
            loss_rec.append(loss_rec_value / number_batches)
            print(f"epoch: {_epoch}, total loss: {loss[-1]}, rec loss: {loss_rec[-1]}, kl loss: {loss_kl[-1]}")
        plot_loss(self.epochs,loss,loss_rec,loss_kl)


    def extract_latent_space(self, dataloader):
        self.eval()
        with torch.no_grad():
            latent_space = []
            labels = []
            for data, batch_labels ,score in dataloader:
                labels.extend(batch_labels)
                data = data.to(device)
                score = score.to(device)
                batch_labels = batch_labels.to(device)
                batch_labels = torch.tensor(batch_labels,dtype=torch.long).squeeze()
                batch_labels = F.one_hot(batch_labels, num_classes=num_classes)
                mu, _ = model.encode(data, batch_labels)
                latent_space.append(mu.cpu().numpy())
            latent_space = np.vstack(latent_space)
            labels = [int(label.item()) for label in labels]
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_space)
        plot_latent_space(latent_2d, labels, "2D Visualization of Latent Space isung TSNE ")
        pca = PCA(n_components=2, random_state=42)
        latent_2d = pca.fit_transform(latent_space)
        plot_latent_space(latent_2d, labels, "2D Visualization of Latent Space using PCA ")
           





