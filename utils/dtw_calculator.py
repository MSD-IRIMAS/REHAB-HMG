

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
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
        default='CVAE',
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
        default=0.99
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

    output_directory_dataset = output_directory_gen_models + args.dataset + '/'
    create_directory(output_directory_dataset)

    output_directory_generator = output_directory_dataset + args.generative_model + '/'
    create_directory(output_directory_generator)

    output_directory_weights_losses = output_directory_generator + 'Wrec_' + str(args.wrec) + '_Wkl_' + str(args.wkl) + '/'
    create_directory(output_directory_weights_losses)

    output_directory_run = args.output_directory_weights_losses + 'run_' + str(args.runs) + '/'
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
                    regressor = KNeighborsTimeSeriesRegressor(distance="euclidean")
                    strain=strain.squeeze(1)
                    regressor.fit(train_data, strain)
                    y_pred = regressor.predict(test_features)
                    print(y_pred)
                    print('------------------------------------------------------------------------')
                    print(stest.squeeze(1))
                    print('------------------------------------------------------------------------')
                    print(mean_squared_error(stest,y_pred))


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
                    print('reg fit')
                    y_pred = regressor.predict(test_features)
                    print(y_pred)
                    print('------------------------------------------------------------------------')
                    print(stest.squeeze(1))
                    print('------------------------------------------------------------------------')
                    print(mean_squared_error(stest,y_pred))
     
    elif args.on == 'generated':
        weights_loss = {
                'wrec' : args.wrec,
                'wkl' : args.wkl,}
            #generate samples 
        generator = CVAE(output_directory=output_directory_skeletons_class,
                    device=args.device,
                    w_rec=weights_loss['wrec'],
                    w_kl=weights_loss['wkl'])
        generated_samples,gen_scores = generator.generate_samples_from_prior(device = args.device,class_index=args.class_index,gif_directory=reg_run_dir_class,dataloader=test_loader)
        gen_set = Kimore(generated_samples,labels,gen_scores)
        gen_loader = DataLoader(gen_set,batch_size=10,shuffle =True)

        if args.regression_models == 'STGCN':
                    stgcn_directory = output_directory_results + 'STGCN/'
                    stgcn_run_dir_class = stgcn_directory + 'run_' + str(args.runs) +'/class_'+str(args.class_index)
                    stgcn_run_dir = stgcn_run_dir_class + '/best_stgcn.pth'
                    feature_extractor_stgcn = FeatureExtractor(model_name=args.regression_models, model_path=stgcn_run_dir, device=args.device)
                    train_data = feature_extractor_stgcn.extract_features(train_loader)
                    test_features= feature_extractor_stgcn.extract_features(gen_loader)
                    regressor = KNeighborsTimeSeriesRegressor(distance="euclidean")
                    strain=strain.squeeze(1)
                    regressor.fit(train_data, strain)
                    y_pred = regressor.predict(test_features)
                    print(y_pred)
                    print('------------------------------------------------------------------------')
                    print(gen_scores.squeeze(1))
                    print('------------------------------------------------------------------------')
                    print(mean_squared_error(gen_scores,y_pred))


        elif args.regression_models == 'REG':
                    regressor_directory = output_directory_results + 'REG/'
                    reg_run_dir_class = regressor_directory + 'run_' + str(args.runs) +'/class_'+str(args.class_index)
                    reg_run_dir = reg_run_dir_class + '/best_regressor.pth'
                                
                    feature_extractor_reg = FeatureExtractor(model_name=args.regression_models, model_path=reg_run_dir, device=args.device)
                    train_data = feature_extractor_reg.extract_features(train_loader)
                    test_features= feature_extractor_reg.extract_features(gen_loader)
                    regressor = KNeighborsTimeSeriesRegressor(distance="euclidean",n_neighbors=5)
                    strain=strain.squeeze(1)
                    regressor.fit(train_data, strain)
                    print('reg fit')
                    y_pred = regressor.predict(test_features)
                    print(y_pred)
                    print('------------------------------------------------------------------------')
                    print(gen_scores.squeeze(1))
                    print('------------------------------------------------------------------------')
                    print(mean_squared_error(gen_scores,y_pred))







































































# def calculate_dtw_and_save_results(xtrain, xtest, output_file, ytrain, min_distance_file, class_index):
#     xtrain = np.reshape(xtrain, (xtrain.shape[0], 748, 18 * 3))
#     xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18 * 3))

#     errors = []
#     total_error =0

#     with open(output_file, 'w') as f:
#         for i in range(len(xtrain)):
#             min_distance = float('inf')
#             min_index = None
#             for j in range(len(xtest)):
#                 distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
#                 f.write(f"DTW distance for generated sample {i} and true sample {j} is: {distance} \\\ \n")
#                 if distance < min_distance:
#                     min_distance = distance
#                     min_index = j
#             error = abs(ytrain[i] - ytrain[min_index])
#             errors.append(error)
#             total_error += error
           
#             f.write(f"Minimal DTW distance for generated sample {i} with score {ytrain[i]:.2f} is {min_distance} "
#                     f"of the true sample {min_index} with score {ytrain[min_index]:.2f} ==> error {error}\\\ \n")
#             write_min_distance_info(min_distance_file, i, min_index, ytrain)

#     # Calculate mean error and standard deviation
#     mean_error = np.mean(errors)
#     print('mean',mean_error)
#     print('total/len',total_error/len(xtrain))
#     std_error = np.std(errors)
#     print('std_error',std_error)
#     # Save mean error and standard deviation
#     with open(output_file, 'a') as f:
#         f.write(f"\nMean error: {mean_error}\n")
#         f.write(f"Standard deviation of error: {std_error}\n")

# def write_min_distance_info(file_path, sample_index, min_index, ytrain):
#     with open(file_path, 'a') as f:
#         f.write(f"Minimal DTW distance for generated sample {sample_index} with score {ytrain[sample_index]:.2f}  "
#                 f"achieved with true sample {min_index} with score {ytrain[min_index]:.2f} \n")

            
            






# def calculate_dtw(xtrain,xtest):
#     xtrain = np.reshape(xtrain, (xtrain.shape[0], 748, 18*3))
#     xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18*3))
#     for i in range(len(xtrain)):
#         for j in range(len(xtest)):
#             distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
#             print("DTW distance for generated sample", i,"and true sample",j,"is: ", distance)


# def calculate_dtw_and_save_results(xtrain, xtest, output_file,ytrain,class_index):
#     with open(output_file, 'w') as f:
#         for i in range(len(xtrain)):
#             min_distance = float('inf')
#             min_index = None
#             for j in range(7):
#                 distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
#                 f.write(f"DTW distance for generated sample {i}  and true sample {j} is: {distance}\n")
#                 if distance < min_distance:
#                     min_distance = distance
#                     min_index = j
#             f.write(f"Minimal DTW distance for generated sample {i} with score {ytrain[i]:.2f} is: {min_distance} "
#                     f"(achieved with true sample {min_index}) with score {ytrain[min_index]:.2f} \n")


# # calculate_dtw_and_save_results(xtrain, data, f'../results/run_0/class_{class_index}/dtw_evaluation_results_prior.txt',scores,class_index)


# def write_min_distance_info(file_path, sample_index, min_index, ytrain):
#     with open(file_path, 'a') as f:
#         f.write(f"Minimal DTW distance for generated sample {sample_index} with score {ytrain[sample_index]:.2f}  "
#                 f"achieved with true sample {min_index} with score {ytrain[min_index]:.2f} \n")

# def calculate_dtw_and_save_results(xtrain, xtest, output_file, ytrain, min_distance_file,class_index):
#     xtrain = np.reshape(xtrain, (xtrain.shape[0], 748, 18*3))
#     xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18*3))
#     with open(output_file, 'w') as f:
#         for i in range(len(xtrain)):
#             min_distance = float('inf')
#             min_index = None
#             for j in range(len(xtest)):
#                 distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
#                 f.write(f"DTW distance for generated sample {i} and true sample {j} is: {distance} \\\ \n")
#                 if distance < min_distance:
#                     min_distance = distance
#                     min_index = j
#             f.write(f"Minimal DTW distance for generated sample {i} with score {ytrain[i]:.2f} is {min_distance} of the true sample{min_index} \\\ \n")
#             write_min_distance_info(min_distance_file, i, min_index, ytrain)


# Example usage
class_index =4
# fold = 5
# for class_index in range(0,5):
# for fold in range(1,6):
#     xtrain = np.load(f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/generated_samples_prior.npy')
#     scores= np.load(f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/scores.npy').squeeze(0).squeeze(1)
#     data= np.load(f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/true_samples_{class_index}.npy')
#     print(xtrain.shape,data.shape,scores.shape)
#     calculate_dtw_and_save_results(xtrain, data, f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/dtw_evaluation_results_prior.txt',scores, f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/min_dtw_evaluation_results_prior.txt',class_index)





# calculate_dtw_for_labels = calculate_dtw_for_labels()
# calculate_dtw_for_labels.calculate_dtw_and_save_results(xtrain,normalize_skeletons(data),f'dtw_for_label_c{class_index}.txt')




# dtw_calculator = calculate_dtw_for_scores()
# dtw_calculator.calculate_dtw(xtrain, data, scores, class_index)
