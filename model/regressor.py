import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import pandas as pd
from torch.autograd import Variable
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
import sys
sys.path.append('/home/hferrar/HMG/dataset')
sys.path.append('/home/hferrar/HMG/urils')
from dataset import load_class
from normalize import normalize_skeletons
from aeon.distances import dtw_distance
from fastdtw import fastdtw


X_train = np.load('/home/hferrar/HMG/results/run_0/generated_samples/class_0/generated_samples.npy')
print(X_train.shape)
y_train = np.load('/home/hferrar/HMG/results/run_0/generated_samples/class_0/true_scores.npy')
X_train = np.reshape(X_train, (X_train.shape[0], 748, 18*3))
data,labels,scores = load_class(0,root_dir='/home/hferrar/HMG/data/Kimore/')
data = normalize_skeletons(data)
print(data.shape)
X_test = np.reshape(data, (data.shape[0], data.shape[1], 18*3))



def calculate_dtw(xtrain,xtest):
    for i in range(len(xtrain)):
        for j in range(len(xtest)):
            distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
            print("DTW distance for generated sample", i,"and true sample",j,"is: ", distance)


def calculate_dtw_and_save_results(xtrain, xtest, output_file,ytrain):
    with open(output_file, 'w') as f:
        for i in range(len(xtrain)):
            min_distance = float('inf')
            min_index = None
            for j in range(len(xtest)):
                distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
                f.write(f"DTW distance for generated sample {i}  and true sample {j} is: {distance}\n")
                if distance < min_distance:
                    min_distance = distance
                    min_index = j
            f.write(f"Minimal DTW distance for generated sample {i} with score {ytrain[i]} is: {min_distance} "
                    f"(achieved with true sample {min_index}) with score {ytrain[min_index]} \n")


calculate_dtw_and_save_results(X_train, X_test, '/home/hferrar/HMG/results/dtw_results.txt',y_train)
# print(y_train)
# print('--------------------------------------------')
# X_train = np.transpose(X_train, (0, 2, 1))
# X_test = np.transpose(X_test, (0, 2, 1))
# reg = KNeighborsTimeSeriesRegressor(distance='dtw')
# reg.fit(X_train,y_train)
# pred=reg.predict(X_test)
# print(pred)



# calculate_dtw(X_train,X_test)

def write_min_distance_info(file_path, sample_index, min_index, ytrain):
    with open(file_path, 'a') as f:
        f.write(f"Minimal DTW distance for generated sample {sample_index} with score {ytrain[sample_index]}  "
                f"achieved with true sample {min_index} with score {ytrain[min_index]} \n")

def calculate_dtw_and_save_results(xtrain, xtest, output_file, ytrain, min_distance_file):
    with open(output_file, 'w') as f:
        for i in range(len(xtrain)):
            min_distance = float('inf')
            min_index = None
            for j in range(len(xtest)):
                distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
                f.write(f"DTW distance for generated sample {i} and true sample {j} is: {distance}\n")
                if distance < min_distance:
                    min_distance = distance
                    min_index = j
            f.write(f"Minimal DTW distance for generated sample {i} with score {ytrain[i]} is: {min_distance}\n")
            write_min_distance_info(min_distance_file, i, min_index, ytrain)

# Example usage
calculate_dtw_and_save_results(X_train, X_test, '/home/hferrar/HMG/results/dtw_results.txt',y_train, '/home/hferrar/HMG/results/min_distance_file.txt')
