
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
sys.path.append('../')
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from utils.plot import plot_regressor_loss,plot_true_pred_scores
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error




class REG(nn.Module):
    def __init__(self, device,output_directory,filters=128, latent_dimension=16,lr=1e-4,epochs=2000):
        super(REG, self).__init__()
        self.filters = filters
        self.lr = lr
        self.epochs = epochs
        self.device=device
        self.output_directory = output_directory
        self.conv1 = nn.Conv1d(54, self.filters, kernel_size=70)
        self.conv2 = nn.Conv1d(self.filters,self.filters,kernel_size=40)
        self.conv3 = nn.Conv1d(self.filters,self.filters,kernel_size=20)
        self.conv4 = nn.Conv1d(self.filters,self.filters,kernel_size=10)
        self.conv5 = nn.Conv1d(self.filters,self.filters,kernel_size=5)
        self.conv6 = nn.Conv1d(self.filters,self.filters,kernel_size=3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear0 = nn.Linear(128*606, 256)
        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 1)

    def forward(self, x,extract_feature=False):
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = x.view(x.size(0), -1)
        
        x = self.linear0(x)
        if extract_feature == True:
            return x
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    def loss(self,score,output):

        criterion = nn.MSELoss()
        mse_loss=criterion(output,score)
        return mse_loss

    def train_fun(self, device, train_loader, test_loader):
        train_losses = []
        test_losses = []

        self.device=device
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        min_loss = float('inf')
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            for batch_idx, (data,_, score) in enumerate(train_loader):
              
                data, score = data.to(self.device), score.to(self.device)
                optimizer.zero_grad()
                output = self(data)
                score = score.view_as(output)
                loss = self.loss(output, score)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            self.eval()
            test_loss = 0.0
            with torch.no_grad():
                for data,_, score in test_loader:
                    data, score = data.to(self.device), score.to(self.device)
                    output = self(data)

                    score = score.view_as(output)
                    loss = self.loss(output, score)
                    test_loss += loss.item()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            if test_losses[-1] < min_loss:
                min_loss = test_losses[-1]
                torch.save(self.state_dict(), os.path.join(self.output_directory, 'best_regressor.pth'))
        torch.save(self.state_dict(), os.path.join(self.output_directory, 'last_regressor.pth'))
        plot_regressor_loss(self.epochs, train_losses, test_losses,self.output_directory)
    
    def square_plot(self,pred_test,valid_y):
        plt.figure(figsize = (8,8))
        plt.subplot(2,1,1)

        plt.plot(pred_test,'s', color='red', label='Prediction', linestyle='None', alpha = 0.5, markersize=6)
        plt.plot(valid_y,'o', color='green',label='True Score', alpha = 0.4, markersize=6)
        plt.title('Testing Set',fontsize=18)
        # plt.ylim([-0.1,1.1])
        plt.xlabel('sample Number',fontsize=16)
        plt.ylabel('Score',fontsize=16)
        plt.legend(loc=3, prop={'size':14}) # loc:position
       
        title='score comparaison'
        plt.savefig(os.path.join(self.output_directory, title+'.pdf'))
        plt.close()
    
    def plot_train_scores(self,device,train_loader):
        device = self.device
        self.to(device)
        num_samples = train_loader.dataset.__len__()
        true_scores = []
        predicted_scores = []
        self.load_state_dict(torch.load(self.output_directory + 'best_regressor.pth', map_location=device))
        self.eval()
        with torch.no_grad():
            for i in range(num_samples):
                input_tensor = train_loader.dataset[i]
                data = input_tensor[0].unsqueeze(0).to(device)
                true_score = input_tensor[2].item()
                prediction = self(data)
                predicted_score = prediction.item()
                true_scores.append(true_score)
                predicted_scores.append(predicted_score)
                print(f'Sample: {i}/{num_samples}, True Score: {true_score:.4f}, Predicted Score: {predicted_score:.4f}')
            plot_true_pred_scores(predicted_scores,true_scores,self.output_directory,title='train_scores')

    def predict_scores(self, data_loader,device):
        device = self.device
        self.to(device)
        num_samples = data_loader.dataset.__len__()
        true_scores = []
        predicted_scores = []
        self.load_state_dict(torch.load(self.output_directory + 'best_regressor.pth', map_location=device))
        self.eval()
        with torch.no_grad():
            for i in range(num_samples):
                input_tensor = data_loader.dataset[i]
                data = input_tensor[0].unsqueeze(0).to(self.device)
                true_score = input_tensor[2].item()
                prediction = self(data)
                predicted_score = prediction.item()
                true_scores.append(true_score*100)
                predicted_scores.append(predicted_score*100)

                print(f'Sample: {i+1}/{num_samples}, True Score: {true_score:.4f}, Predicted Score: {predicted_score:.4f}')
        rmse = mean_squared_error(true_scores,predicted_scores)
        mae = mean_absolute_error(true_scores,predicted_scores)
        mape = mean_absolute_percentage_error(true_scores,predicted_scores)
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'MAPE: {mape:.4f}')
        
        plot_true_pred_scores(predicted_scores,true_scores,self.output_directory,title='test_scores')
        self.square_plot(predicted_scores,true_scores)
        return true_scores, predicted_scores


