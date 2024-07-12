
import os
import sys
import torch
import numpy as np
import scipy as sc
import torch.nn as nn
sys.path.append('../')
from model.stgcn import STGCN
from scipy.linalg import sqrtm, eig
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
                    
                features.append(output.cpu().detach())
                
               
        features = np.concatenate(features,axis=0)
  
        return features


class FID:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def calculate_fid(self, data_loader_x, data_loader_y):
        
        real_latent = self.feature_extractor.extract_features(data_loader_x)
        gen_latent = self.feature_extractor.extract_features(data_loader_y)
        print("NaNs in real features:", np.isnan(real_latent).any())
        print("Infinities in real features:", np.isinf(real_latent).any())
        print("NaNs in real features:", np.isnan(gen_latent).any())
        print("Infinities in real features:", np.isinf(gen_latent).any())

        mean_real = np.mean(real_latent, axis=0)
        mean_gen = np.mean(gen_latent, axis=0)

        print("NaNs in real mean:", np.isnan(mean_real).any())
        print("Infinities in real mean:", np.isinf(mean_real).any())
        print("NaNs in generated mean:", np.isnan(mean_gen).any())
        print("Infinities in generated mean:", np.isinf(mean_gen).any())





        cov_real = np.cov(real_latent, rowvar=False)
        cov_gen = np.cov(gen_latent, rowvar=False)
        print("NaNs in real covariance:", np.isnan(cov_real).any())
        print("Infinities in real covariance:", np.isinf(cov_real).any())
        print("NaNs in generated covariance:", np.isnan(cov_gen).any())
        print("Infinities in generated covariance:", np.isinf(cov_gen).any())

                

        cov_gen = (cov_gen + cov_gen.T) / 2
        epsilon = 1e-10
        mean_gen += epsilon 
        cov_gen += epsilon * np.eye(cov_gen.shape[0])
        eigenvalues = np.linalg.eigvals(cov_gen)

        diff_means = np.sum(np.square(mean_real - mean_gen))

        cov_prod = sqrtm(cov_real.dot(cov_gen))
        

        if np.iscomplexobj(cov_prod):
            cov_prod = cov_prod.real

        fid = diff_means + np.trace(cov_real + cov_gen - 2.0 * cov_prod)
        
        print(fid)
        print(np.all(eigenvalues >= 0))
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

    
class APD:
    def __init__(self,feature_extractor,Sapd=20):
        self.feature_extractor = feature_extractor
        self.Sapd = Sapd
    def calculate_apd(self, xgenerated):
        gen_latent = self.feature_extractor.extract_features(xgenerated)

        if self.Sapd > len(xgenerated):
            self._Sapd = len(xgenerated)
        else:
            self._Sapd = self.Sapd

  
        
        all_indices = np.arange(len(xgenerated))
        V = gen_latent[np.random.choice(a=all_indices, size=2)]
        V_prime = gen_latent[np.random.choice(a=all_indices, size=2)]
        apd = np.mean(np.linalg.norm(V - V_prime, axis=1))

        return apd