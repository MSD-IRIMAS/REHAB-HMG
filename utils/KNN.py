

import os
import sys
import torch
import random
import numpy as np
import scipy as sc
import torch.nn as nn
sys.path.append('../')
from model.stgcn import STGCN
import argparse
from scipy.linalg import sqrtm, eig
from model.regressor import REG
from model.svae import SVAE
from model.AS_CVAE import ASCVAE
from numpy.lib import scimath as sc
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from dataset.dataset import load_class, Kimore
from sklearn.model_selection import train_test_split
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from utils.visualize import create_directory
from utils.normalize import normalize_skeletons
from utils.metrics import FeatureExtractor
from torch.utils.data import DataLoader,Subset
from sklearn import neighbors
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True  # If using CUDA
torch.backends.cudnn.benchmark 
def get_args():
    parser = argparse.ArgumentParser(
    description="training STGCN to predict generated data score")

    parser.add_argument(
        '--regression_models',
        type=str,
        choices=['STGCN','REG'],
        default='REG',
    )
    parser.add_argument(
        '--generative_model',
        type=str,
        default='ASCVAE',
    )

    parser.add_argument(
        '--dataset',
        help="Which dataset to use.",
        type=str,
        default='Kimore'
    )

    parser.add_argument(
        '--output-directory',
        type=str,
        default='results/'
    )

    parser.add_argument(
        '--runs',
        help="Number of experiments to do.",
        type=int,
        default=0
    )
    parser.add_argument(
        '--device',
        help="Device to run the training on.",
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )

    parser.add_argument(
        '--class_index',
        help="which class to generate",
        type=int,
        default=0
    )

    parser.add_argument(
        '--wrec',
        help="Weight for the reconstruction loss.",
        type=float,
        default=0.999
    )

    parser.add_argument(
        '--wkl',
        help="Weight for the kl loss.",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        '--on',
        choices=['real','generated'],
        default='real',
    )

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()
    output_directory_results = args.output_directory
    create_directory(output_directory_results)

    output_directory_gen_models = output_directory_results + 'Generative_models/'
    create_directory(output_directory_gen_models)

    output_directory_dataset = output_directory_gen_models + 'Score+Action_conditioned' + '/'
    create_directory(output_directory_dataset)

    output_directory_generator = output_directory_dataset + args.generative_model + '/'
    create_directory(output_directory_generator)

    output_directory_weights_losses = output_directory_generator + 'Wrec_' + str(args.wrec) + '_Wkl_' + str(args.wkl) + '/'
    create_directory(output_directory_weights_losses)

    output_directory_run = output_directory_weights_losses + 'run_' + str(args.runs) + '/'
    create_directory(output_directory_run)

    output_directory_skeletons_class = output_directory_run + 'class_' + str(args.class_index) + '/'
    create_directory(output_directory_skeletons_class)

    dataset_dir = 'data/' + args.dataset + '/'
    data,labels,scores = load_class(args.class_index,root_dir=dataset_dir)
    xtrain,xtest,ytrain,ytest,strain,stest= train_test_split(data,labels,scores,test_size=0.2,random_state=42)
    xtrain,min_X, max_X,min_Y,max_Y, min_Z,max_Z= normalize_skeletons(xtrain)
    train_set = Kimore(xtrain,ytrain,strain)
    train_loader = DataLoader(train_set,batch_size=16,shuffle =True)
    xtest,_,_,_,_,_,_= normalize_skeletons(xtest,min_X, max_X,min_Y,max_Y, min_Z,max_Z)
    test_set = Kimore(xtest,ytest,stest)
    test_loader = DataLoader(test_set,batch_size=16,shuffle=False)

 

    output_directory_results = args.output_directory + 'regression_models/'
    create_directory(output_directory_results)


    if args.on == 'real':
        if args.regression_models == 'STGCN':
                    stgcn_directory = output_directory_results + 'STGCN/'
                    stgcn_run_dir_class = stgcn_directory + 'run_' + str(args.runs) +'/class_'+str(args.class_index)
                    stgcn_run_dir = stgcn_run_dir_class + '/best_stgcn.pth'
                    feature_extractor_stgcn = FeatureExtractor(model_name=args.regression_models, model_path=stgcn_run_dir, device=args.device)
                    train_data = feature_extractor_stgcn.extract_features(train_loader)
                    test_features= feature_extractor_stgcn.extract_features(test_loader)
                    regressor = KNeighborsTimeSeriesRegressor(distance="euclidean",n_neighbors=5)
                    strain=strain.squeeze(1)
                    regressor.fit(train_data, strain)
                    y_pred = regressor.predict(test_features)
                    stest = stest.squeeze(1)
                    for i in range(len(y_pred)): 
                        print(f'true score {stest[i]:.2f}, predicted score {y_pred[i]:.2f}')
                    print(rmse(y_pred,stest))
                    print('MAE',mae(y_pred,stest))


        elif args.regression_models == 'REG':
                    regressor_directory = output_directory_results + 'REG/'
                    reg_run_dir_class = regressor_directory + 'run_' + str(args.runs) +'/class_'+str(args.class_index)
                    reg_run_dir = reg_run_dir_class + '/best_regressor.pth'
                    feature_extractor_reg = FeatureExtractor(model_name=args.regression_models, model_path=reg_run_dir, device=args.device)
                    train_data = feature_extractor_reg.extract_features(train_loader)
                    test_features= feature_extractor_reg.extract_features(test_loader)
                    regressor = KNeighborsTimeSeriesRegressor(distance="euclidean",n_neighbors=5)
                    strain=strain.squeeze(1)
                    regressor.fit(train_data, strain)
                    
                    y_pred = regressor.predict(test_features)
                    stest = stest.squeeze(1)
                    for i in range(len(y_pred)): 
                        print(f'true score {stest[i]:.2f}, predicted score {y_pred[i]:.2f}')
                    rmse_score = rmse(stest,y_pred)
                    print(rmse_score)
                    print('MAE',mae(y_pred,stest))

                    
     
    elif args.on == 'generated':
   
        weights_loss = {
                'wrec' : args.wrec,
                'wkl' : args.wkl,}
            #generate samples 
        generator = ASCVAE(output_directory=output_directory_run,
                    device=args.device,
                    w_rec=weights_loss['wrec'],
                    w_kl=weights_loss['wkl'])
        
        if args.regression_models == 'STGCN':
                    stgcn_directory = output_directory_results + 'STGCN/'
                    stgcn_run_dir_class = stgcn_directory + 'run_' + str(args.runs) +'/class_'+str(args.class_index)
                    stgcn_run_dir = stgcn_run_dir_class + '/best_stgcn.pth'

                    generated_samples,gen_scores = generator.generate_samples_from_prior(device = args.device,gif_directory=stgcn_run_dir_class,dataloader=test_loader,class_index=args.class_index)
                    gen_set = Kimore(generated_samples,labels,gen_scores)
                    gen_loader = DataLoader(gen_set,batch_size=10,shuffle =True)
                    feature_extractor_stgcn = FeatureExtractor(model_name=args.regression_models, model_path=stgcn_run_dir, device=args.device)
                    train_data = feature_extractor_stgcn.extract_features(train_loader)
                    test_features= feature_extractor_stgcn.extract_features(gen_loader)
                    regressor = KNeighborsTimeSeriesRegressor(distance="euclidean")
                    strain=strain.squeeze(1)
                    regressor.fit(train_data, strain)
                    y_pred = regressor.predict(test_features)
                    gen_scores=np.unique(gen_scores)

                    print(len(gen_scores),len(y_pred))
                    for i in range(len(y_pred)): 
                        print(f'true score {gen_scores[i]:.2f}, predicted score {y_pred[i]:.2f}')
                    print(rmse(y_pred,gen_scores))
                    print('MAE',mae(y_pred,gen_scores))


        elif args.regression_models == 'REG':
                    regressor_directory = output_directory_results + 'REG/'
                    reg_run_dir_class = regressor_directory + 'run_' + str(args.runs) +'/class_'+str(args.class_index)
                    reg_run_dir = reg_run_dir_class + '/best_regressor.pth'
                    generated_samples,gen_scores = generator.generate_samples_from_prior(device = args.device,gif_directory=reg_run_dir_class,dataloader=test_loader,class_index=args.class_index)
                    gen_set = Kimore(generated_samples,labels,gen_scores)
                    gen_loader = DataLoader(gen_set,batch_size=16,shuffle =True)
                    feature_extractor_reg = FeatureExtractor(model_name=args.regression_models, model_path=reg_run_dir, device=args.device)
                    train_data = feature_extractor_reg.extract_features(train_loader)
                    test_features= feature_extractor_reg.extract_features(gen_loader)
                    regressor = KNeighborsTimeSeriesRegressor(distance="euclidean",n_neighbors=5)
                    strain=strain.squeeze(1)
                    regressor.fit(train_data, strain)
                    y_pred = regressor.predict(test_features)
                    gen_scores=np.unique(gen_scores)
                
                    for i in range(len(y_pred)): 
                        
                        print(f'true score {gen_scores[i]:.2f}, predicted score {y_pred[i]:.2f}')
                        
                    print(rmse(y_pred,gen_scores))
                    print('MAE',mae(y_pred,gen_scores))
































