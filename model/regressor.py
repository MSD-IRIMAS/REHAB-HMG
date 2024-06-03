import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import pandas as pd
from torch.autograd import Variable

class Regressor(nn.Module):
        def __init__(self,output_directory,epochs,device,filters=128,lr=1e-4):
            super(Regressor,self).__init__()
            self.output_directory = output_directory
            self.epochs = epochs
            self.filters = filters
            self.lr = lr
            self.conv1 = nn.Conv1d(54, self.filters, kernel_size=120)
            self.conv2 = nn.Conv1d(self.filters,self.filters,kernel_size=60)
            self.conv3 = nn.Conv1d(self.filters,self.filters,kernel_size=30)
            self.conv4 = nn.Conv1d(self.filters,self.filters,kernel_size=15)
            self.conv5 = nn.Conv1d(self.filters,self.filters,kernel_size=5)
            self.relu = nn.ReLU()
            self.linear0 = nn.Linear(self.filters*523,256)
            self.linear1 = nn.Linear(256,64)
            self.linear2 = nn.Linear(64,1)

        def forward(self,x):
                x = x.view(x.size(0),x.size(1),-1)
                x = x.permute(0,2,1)
                x = self.relu(self.conv1(x))  
                x = self.relu(self.conv2(x))   
                x = self.relu(self.conv3(x))
                x = self.relu(self.conv4(x))
                x = self.relu(self.conv5(x))
                x = x.view(x.size(0), -1)
                x = self.relu(self.linear0(x))
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x

        def train_regressor(self,optimizer, epochs, device, train_loader,test_loader):
            train_losses = []
            test_losses = []
            criterion = nn.MSELoss()
            min_loss = float('inf')
            for epoch in range(epochs):
                self.train()

                train_loss = 0.0
                for batch_idx, (data, score) in enumerate(train_loader):
                    data = data.to(self.device)
                    score = score.to(self.device)
                    optimizer.zero_grad()
                    output = self(data)
                    score = score.view_as(output)
                    loss = criterion(output, score)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)
                self.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for data, score in test_loader:
                        data = data.to(self.device)
                        score = score.to(self.device)    
                        output = self(data)
                        score = score.view_as(output)
                        loss = criterion(output, score)
                        test_loss += loss.item()

                test_loss /= len(test_loader.dataset)
                test_losses.append(test_loss)
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
                if test_loss < min_loss:
                    min_loss = test_loss
                    torch.save(self.state_dict(), os.path.join(self.output_directory, 'best_regressor.pth'))


            plot_regressor_loss(self.epochs,train_loss,test_loss,self,output_directory)

        