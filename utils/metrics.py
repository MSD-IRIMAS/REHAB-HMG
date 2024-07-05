
import os
import sys
import torch
import numpy as np
import scipy as sc
import torch.nn as nn
sys.path.append('../')
from model.stgcn import STGCN
from scipy.linalg import sqrtm 
from model.regressor import REG
from numpy.lib import scimath as sc
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from dataset.dataset import load_class, Kimore
from sklearn.model_selection import train_test_split


class FeatureExtractor:
    def __init__(self, model_name, model_path, device):
        self.device = torch.device(device)
        self.model_name = model_name
        
        if model_name == 'STGCN':
            self.model = STGCN(device=device, output_directory='./')
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            # self.features_extractor = nn.Sequential(*list(self.model.children())[:-3])
        elif model_name == 'REG':
            self.model = REG(device=device, output_directory='./')
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        
        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, data_loader):
        features = []
        # self.features_extractor.eval()
        with torch.no_grad():
            for  batch_idx, (data,_, score) in enumerate(data_loader):
                data = data.to(self.device)
                if self.model_name == 'STGCN':
                    output = self.model(data, feature_extractor=True)
                elif self.model_name =='REG':
                    # 'REG'
                    self.model.eval()
                    output = self.model(data,extract_feature=True)
                    
                features.append(output.cpu().detach().numpy())
        features = np.concatenate(features, axis=0)
        return features


class FID:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def calculate_fid(self, data_loader_x, data_loader_y):
        
        features_x = self.feature_extractor.extract_features(data_loader_x)
        features_y = self.feature_extractor.extract_features(data_loader_y)
        print(features_x.shape)
        mean_x = np.mean(features_x, axis=0)
        mean_y = np.mean(features_y, axis=0)

        cov_x = np.cov(features_x, rowvar=False)
        cov_y = np.cov(features_y, rowvar=False)

        mean_diff = mean_x - mean_y
        mean_squared_norm = np.dot(mean_diff, mean_diff)

        cov_sqrt_product = sqrtm(np.dot(cov_x, cov_y))
        if np.iscomplexobj(cov_sqrt_product):
            cov_sqrt_product = cov_sqrt_product.real
        trace_sqrt = np.trace(cov_sqrt_product)
        fid = mean_squared_norm + np.trace(cov_x) + np.trace(cov_y) - 2 * trace_sqrt
        return fid
class Coverage:
    def __init__(self, feature_extractor, n_neighbors=5):
        self.feature_extractor = feature_extractor
        self.n_neighbors = n_neighbors

    def get_distances_k_neighbors(self, x: np.ndarray, k: int):
        nn_model = NearestNeighbors(n_neighbors=k)
        nn_model.fit(X=x)
        distances_neighbors, _ = nn_model.kneighbors(X=x)
        return distances_neighbors[:, k - 1]

    def calculate(self, xgenerated_loader=None, xreal_loader=None):
        real_latent = self.feature_extractor.extract_features(xreal_loader)
        gen_latent = self.feature_extractor.extract_features(xgenerated_loader)

        real_gen_distance_matrix = pairwise_distances(X=real_latent, Y=gen_latent)
        real_distances_k_neighbors = self.get_distances_k_neighbors(x=real_latent, k=self.n_neighbors + 1)

        distances_nearest_neighbor_real_to_gen = np.min(real_gen_distance_matrix, axis=1)
        exists_inside_neighborhood = distances_nearest_neighbor_real_to_gen < real_distances_k_neighbors
        coverage = np.mean(exists_inside_neighborhood)

        return coverage

class Density:
    def __init__(self, feature_extractor, n_neighbors=5):
        self.feature_extractor = feature_extractor
        self.n_neighbors = n_neighbors

    def get_distances_k_neighbors(self, x: np.ndarray, k: int) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X=x)
        distances_neighbors, _ = nn.kneighbors(X=x)
        return distances_neighbors[:, k - 1]

    def calculate_density(self, xreal_loader, xgenerated_loader ):
        real_latent = self.feature_extractor.extract_features(xreal_loader)
        gen_latent = self.feature_extractor.extract_features(xgenerated_loader)
        real_gen_distance_matrix = pairwise_distances(X=real_latent, Y=gen_latent)

        real_distances_k_neighbors = self.get_distances_k_neighbors(
            x=real_latent, k=self.n_neighbors + 1
        )

        scaler = 1 / (1.0 * self.n_neighbors)
        inside_neighborhood = (
            real_gen_distance_matrix
            < np.expand_dims(real_distances_k_neighbors, axis=1)
        ).sum(axis=0)

        density = scaler * np.mean(inside_neighborhood)

        return density


class MMS:
    def __init__(self, feature_extractor, n_neighbors=5):
        self.feature_extractor = feature_extractor
        self.n_neighbors = n_neighbors

    def calculate_mms(self, generated_data, real_data):
        real_latent = self.feature_extractor.extract_features(real_data)
        gen_latent = self.feature_extractor.extract_features(generated_data)
        
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(real_latent)
        distances, _ = nn.kneighbors(X=gen_latent, return_distance=True)
        
        if distances.shape[-1] > 1:
            distances = distances[:, 1]

        return np.mean(distances)

    
# class FID:
#     def __init__(self, model_path, device, data_loader):
#         self.device = torch.device(device)
#         self.model = STGCN(device=device, output_directory='./')
#         self.model.load_state_dict(torch.load(model_path, map_location=device))
#         self.model.to(self.device)
#         self.model.eval()
#         self.data_loader = data_loader
#         # self.features_extractor = nn.Sequential(*list(self.model.children())[:-3])  
#     def extract_features(self, data_loader):
#         features = []
#         self.features_extractor.eval()
#         with torch.no_grad():
#             for data, _, s in data_loader:
#                 data = data.to(self.device)
#                 output = self.model(data, feature_extractor=True)
               
                
#                 features.append(output.cpu().detach().numpy()) 
#         features = np.concatenate(features, axis=0)  
#         return features

#     def calculate_fid(self, data_loader_x, data_loader_y):
#         features_x = self.extract_features(data_loader_x)
#         features_y = self.extract_features(data_loader_y)

#         mean_x = np.mean(features_x, axis=0)
#         mean_y = np.mean(features_y, axis=0)

#         cov_x = np.cov(features_x, rowvar=False)
#         cov_y = np.cov(features_y, rowvar=False)

#         mean_diff = mean_x - mean_y
#         mean_squared_norm = np.dot(mean_diff, mean_diff)

#         cov_sqrt_product = sqrtm(np.dot(cov_x, cov_y))
#         if np.iscomplexobj(cov_sqrt_product):
#             cov_sqrt_product = cov_sqrt_product.real
#         trace_sqrt = np.trace(cov_sqrt_product)
#         fid = mean_squared_norm + np.trace(cov_x) + np.trace(cov_y) - 2 * trace_sqrt
#         return fid
# class Coverage:
#     def __init__(self, model_path, device, data_loader, n_neighbors=5):
#         self.device = torch.device(device)
#         self.model = STGCN(device=device, output_directory='./')
#         self.model.load_state_dict(torch.load(model_path, map_location=device))
#         self.model.to(self.device)
#         self.model.eval()
#         self.data_loader = data_loader
#         # self.features_extractor = nn.Sequential(*list(self.model.children())[:-3])
#         self.n_neighbors = n_neighbors

#     def extract_features(self, data_loader):
#         features = []
#         self.features_extractor.eval()
#         with torch.no_grad():
#             for data, _, _ in data_loader:
#                 data = data.to(self.device)
#                 output = self.model(data, feature_extractor=True)
#                 features.append(output.cpu().detach().numpy())
#         features = np.concatenate(features, axis=0)
#         return features

#     def get_distances_k_neighbors(self, x: np.ndarray, k: int):
#         nn_model = NearestNeighbors(n_neighbors=k)
#         nn_model.fit(X=x)
#         distances_neighbors, _ = nn_model.kneighbors(X=x)
#         return distances_neighbors[:, k - 1]

#     def calculate(self, xgenerated_loader=None, xreal_loader=None):
#         real_latent = self.extract_features(xreal_loader)
#         gen_latent = self.extract_features(xgenerated_loader)

#         real_gen_distance_matrix = pairwise_distances(X=real_latent, Y=gen_latent)
#         real_distances_k_neighbors = self.get_distances_k_neighbors(x=real_latent, k=self.n_neighbors + 1)

#         distances_nearest_neighbor_real_to_gen = np.min(real_gen_distance_matrix, axis=1)
#         exists_inside_neighborhood = distances_nearest_neighbor_real_to_gen < real_distances_k_neighbors
#         coverage = np.mean(exists_inside_neighborhood)

#         return coverage


# class MMS:
#     def __init__(self, model_path, device, data_loader, n_neighbors=5):
#         self.device = torch.device(device)
#         self.model = STGCN(device=device, output_directory='./')
#         self.model.load_state_dict(torch.load(model_path, map_location=device))
#         self.model.to(self.device)
#         self.model.eval()
#         self.data_loader = data_loader
#         # self.features_extractor = nn.Sequential(*list(self.model.children())[:-3])
#         self.n_neighbors = n_neighbors

#     def extract_features(self, data_loader):
#         features = []
#         self.features_extractor.eval()
#         with torch.no_grad():
#             for data, _, _ in data_loader:
#                 data = data.to(self.device)
#                 output = self.model(data, feature_extractor=True)
#                 features.append(output.cpu().detach().numpy())
#         features = np.concatenate(features, axis=0)
#         return features 
    
#     def calculate_mms(self,generated_data,real_data):
#         real_latent = self.extract_features(real_data)
#         gen_latent = self.extract_features(generated_data)
#         nn = NearestNeighbors(n_neighbors=self.n_neighbors)
#         nn.fit(real_latent)
#         distances, _ = nn.kneighbors(X=gen_latent, return_distance=True)
#         if distances.shape[-1] > 1:
           
#             distances = distances[:, 1]

#         return np.mean(distances)
