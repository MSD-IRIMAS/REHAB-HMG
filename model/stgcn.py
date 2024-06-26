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
sys.path.append('../')
from utils.graph_layers.graph import Graph
from utils.graph_layers.temporal_conv import TemporalConvGraph
from utils.plot import plot_regressor_loss,plot_true_pred_scores
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dropout=0,residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = TemporalConvGraph(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A



class STGCN(nn.Module):
    def __init__(self,device,output_directory,epochs,edge_importance_weighting,score=1,lr=1e-4, **kwargs):
        super(STGCN,self).__init__()
        self.criterion = nn.MSELoss()
        self.score=1
        self.device=device
        self.output_directory = output_directory
        self.epochs = epochs
        self.lr=lr
       
        # load graph
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # self.data_bn = nn.BatchNorm1d(3 * A.size(1))
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.stgcn_network = nn.ModuleList(
       [    st_gcn(3, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs)]
            
            )
        
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.stgcn_network
            ])
        else:
            self.edge_importance = [1] * len(self.stgcn_network)
        # regression layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.lin1 = nn.Linear(256, 64)
        self.lin2= nn.Linear(64, 16)
        self.lin3 = nn.Linear(16, score)
    
    def forward(self,x):
        x =x.permute(0,3,1,2).unsqueeze(4)
        N, C, T, V, M = x.size()
        x = x.view(N * M, C, T, V)

        # x =x.permute(0,3,1,2).unsqueeze(4)
        # N, C, T, V, M = x.size()
        # # x = x.permute(0, 4, 3, 1, 2).contiguous()
        # # x = x.view(N * M, V * C, T)
        # # x = self.data_bn(x)
        # # x = x.view(N, M, V, C, T)
        # # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)
        


        # for gcn in self.stgcn_network:
        #     x, _ = gcn(x, self.A)
        for gcn, importance in zip(self.stgcn_network, self.edge_importance):
            x, _ = gcn(x, self.A * importance)


        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = x.view(x.size(0), -1)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)
        output=self.sigmoid(x)
        return output
    
    def loss(self,score,output):
        mse_loss=self.criterion(score,output)
        return mse_loss
    

   
    def train_stgcn(self, device, train_loader,test_loader):
        
        self.device=device
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
        train_losses = []
        test_losses = []
        min_loss = float('inf')
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            for batch_idx, (data,_, score) in enumerate(train_loader):
                data = data.to(device)
                score = score.to(device)
                optimizer.zero_grad()
                output = self(data)
                # Ensure target tensor has the same shape as the output tensor
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
                for data,_,score in test_loader:
                    data = torch.tensor(data, dtype=torch.float32)
                    data = data.to(device)
                    score = score.to(device)
                    output = self(data)
                    score = score.view_as(output)
                    loss = self.loss( score,output)
                    test_loss += loss.item()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            scheduler.step(test_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            if train_losses[-1] < min_loss:
                min_loss = train_losses[-1]
                torch.save(self.state_dict(), os.path.join(self.output_directory, 'best_stgcn.pth'))


            torch.save(self.state_dict(), os.path.join(self.output_directory, 'last_stgcn.pth'))
        plot_regressor_loss(self.epochs, train_losses, test_losses,self.output_directory)
    def score_error(self,true_scores,predicted_scores):
        rmse = mean_squared_error(true_scores,predicted_scores)
        mae = mean_absolute_error(true_scores,predicted_scores)
        mape = mean_absolute_percentage_error(true_scores,predicted_scores)
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'MAPE: {mape:.4f}')

    def square_plot(self,pred_test,valid_y):
        plt.figure(figsize = (8,8))
        plt.subplot(2,1,1)

        plt.plot(pred_test,'s', color='red', label='Prediction', linestyle='None', alpha = 0.5, markersize=6)
        plt.plot(valid_y,'o', color='green',label='True Score', alpha = 0.4, markersize=6)
        plt.title('Testing Set',fontsize=18)
        plt.ylim([-0.1,1.1])
        plt.xlabel('sample Number',fontsize=16)
        plt.ylabel('Score',fontsize=16)
        plt.legend(loc=3, prop={'size':14}) # loc:position
        plt.tight_layout()
        title='score comparaison'
        plt.savefig(os.path.join(self.output_directory, title+'.pdf'))
        plt.show()

    
    def predict_scores(self, test_loader,device):
        device = self.device
        self.to(device)
        num_samples = test_loader.dataset.__len__()
        true_scores = []
        predicted_scores = []
        self.load_state_dict(torch.load(self.output_directory + 'best_stgcn.pth', map_location=device))
        self.eval()
        with torch.no_grad():
            for i in range(num_samples):
                input_tensor = test_loader.dataset[i]
                data = input_tensor[0].unsqueeze(0).to(device)
                true_score = input_tensor[2].item()
                prediction = self(data)
                predicted_score = prediction.item()
                true_scores.append(true_score)
                predicted_scores.append(predicted_score)
            
                print(f'Sample: {i}/{num_samples}, True Score: {true_score:.4f}, Predicted Score: {predicted_score:.4f}')
            plot_true_pred_scores(predicted_scores,true_scores,self.output_directory)
            self.square_plot(predicted_scores,true_scores)
            self.score_error(true_scores,predicted_scores)

    




  #####################" REFACTOR THIS FUNCTION ##########################"  

    def test_predictions(self,device):
        self.device=device
        self.to(device)
        self.load_state_dict(torch.load(self.output_directory + 'best_stgcn.pth', map_location=device))
        self.eval()
        
        data = np.load('./results/run_0/cross_validation/class_0/fold_1/generated_samples/generated_samples_prior.npy')
        scores=np.load('./results/run_0/cross_validation/class_0/fold_1/generated_samples/scores.npy')
        data = torch.squeeze(torch.tensor(data).float(),1)
        scores = torch.tensor(scores).float()
        data=data.to(device)
  
        prediction= self(data)
        
        print(prediction)
        print('-------------------------------------')
        print(scores / 100)

       

