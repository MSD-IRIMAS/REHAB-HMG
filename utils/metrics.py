
import os
import sys
import torch
import numpy as np
import scipy as sc
import torch.nn as nn
sys.path.append('../')
from model.stgcn import STGCN
from scipy.linalg import sqrtm 
from numpy.lib import scimath as sc
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from dataset.dataset import load_class, Kimore
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset,Subset
from utils.normalize import normalize_skeletons, normalize_test_set
class FID:
    def __init__(self, model_path, device, data_loader):
        self.device = torch.device(device)
        self.model = STGCN(device=device, output_directory='./')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.features_extractor = nn.Sequential(*list(self.model.children())[:-3])  
    def extract_features(self, data_loader):
        features = []
        self.features_extractor.eval()
        with torch.no_grad():
            for data, _, s in data_loader:
                data = data.to(self.device)
                output = self.model(data, feature_extractor=True)
                print(output)
                features.append(output.cpu().detach().numpy()) 
        features = np.concatenate(features, axis=0)  
        return features

    def calculate_fid(self, data_loader_x, data_loader_y):
        features_x = self.extract_features(data_loader_x)
        features_y = self.extract_features(data_loader_y)

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


class Diversity:
    def __init__(self, model_path, device, data_loader):
        self.device = torch.device(device)
        self.model = STGCN(device=device, output_directory='./')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.features_extractor = nn.Sequential(*list(self.model.children())[:-3])  
    def extract_features(self, data_loader):
        features = []
        self.features_extractor.eval()
        with torch.no_grad():
            for data, _, s in data_loader:
                data = data.to(self.device)
                output = self.model(data, feature_extractor=True)
                print(output)
                features.append(output.cpu().detach().numpy()) 
        features = np.concatenate(features, axis=0)  
        return features

    def calculate_diversity(self,data_loader_x,x,Sd):
        pass


class Coverage:
    def __init__(self, model_path, device, data_loader, n_neighbors=5):
        self.device = torch.device(device)
        self.model = STGCN(device=device, output_directory='./')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.features_extractor = nn.Sequential(*list(self.model.children())[:-3])
        self.n_neighbors = n_neighbors

    def extract_features(self, data_loader):
        features = []
        self.features_extractor.eval()
        with torch.no_grad():
            for data, _, _ in data_loader:
                data = data.to(self.device)
                output = self.features_extractor(data)
                features.append(output.cpu().detach().numpy())
        features = np.concatenate(features, axis=0)
        return features

    def get_distances_k_neighbors(self, x: np.ndarray, k: int):
        nn_model = NearestNeighbors(n_neighbors=k)
        nn_model.fit(X=x)
        distances_neighbors, _ = nn_model.kneighbors(X=x)
        return distances_neighbors[:, k - 1]

    def calculate(self, xgenerated_loader=None, xreal_loader=None):
        real_latent = self.extract_features(xreal_loader)
        gen_latent = self.extract_features(xgenerated_loader)

        real_gen_distance_matrix = pairwise_distances(X=real_latent, Y=gen_latent)
        real_distances_k_neighbors = self.get_distances_k_neighbors(x=real_latent, k=self.n_neighbors + 1)

        distances_nearest_neighbor_real_to_gen = np.min(real_gen_distance_matrix, axis=1)
        exists_inside_neighborhood = distances_nearest_neighbor_real_to_gen < real_distances_k_neighbors
        coverage = np.mean(exists_inside_neighborhood)

        return coverage


class MMS:
    def __init__(self, model_path, device, data_loader, n_neighbors=5):
        self.device = torch.device(device)
        self.model = STGCN(device=device, output_directory='./')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.features_extractor = nn.Sequential(*list(self.model.children())[:-3])
        self.n_neighbors = n_neighbors

    def extract_features(self, data_loader):
        features = []
        self.features_extractor.eval()
        with torch.no_grad():
            for data, _, _ in data_loader:
                data = data.to(self.device)
                output = self.features_extractor(data)
                features.append(output.cpu().detach().numpy())
        features = np.concatenate(features, axis=0)
        return features 
    
    def calculate_mms(self,generated_data,real_data):
        real_latent = self.extract_features(xreal_loader)
        gen_latent = self.extract_features(xgenerated_loader)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(real_latent)
        distances, _ = nn.kneighbors(X=gen_latent, return_distance=True)
        if distances.shape[-1] > 1:
            print(distances.shape)
            distances = distances[:, 1]

        return np.mean(distances)
