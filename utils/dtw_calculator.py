import numpy as np
import pandas as pd
import sys
sys.path.append('../dataset')
from dataset import load_class
from aeon.distances import dtw_distance
from normalize import normalize_skeletons


# class calculate_dtw_for_labels:
#     # def calculate_dtw(self,xtrain,xtest):
#     #     for i in range(len(xtrain)):
#     #         for j in range(7):
#     #             distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
#     #             print("DTW distance for generated sample", i,"and true sample",j,"is: ", distance)
   

#     def calculate_dtw_and_save_results(self,xtrain, xtest, output_file):
#         xtrain = xtrain
#         xtrain = np.reshape(xtrain, (xtrain.shape[0], 748, 18*3))
#         xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18*3))
#         with open(output_file, 'w') as f:
#             for i in range(len(xtrain)):
#                 min_distance = float('inf')
#                 min_index = None
#                 for j in range(7):
#                     distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
#                     # f.write(f"DTW distance for generated sample {i} and true sample {j} is: {distance}\n")
#                     if distance < min_distance:
#                         min_distance = distance
#                         min_index = j
                    
#                 f.write(f"Minimal DTW distance for generated sample {i}  is: {min_distance} for the true sample {min_index}\n")
#                 if i == min_index: 
#                         f.write(f" generated sample {i}  matches the true sample {min_index}\n")
               

def calculate_dtw_and_save_results(xtrain, xtest, output_file, ytrain, min_distance_file, class_index):
    xtrain = np.reshape(xtrain, (xtrain.shape[0], 748, 18 * 3))
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18 * 3))

    errors = []
    total_error =0

    with open(output_file, 'w') as f:
        for i in range(len(xtrain)):
            min_distance = float('inf')
            min_index = None
            for j in range(len(xtest)):
                distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
                f.write(f"DTW distance for generated sample {i} and true sample {j} is: {distance} \\\ \n")
                if distance < min_distance:
                    min_distance = distance
                    min_index = j
            error = abs(ytrain[i] - ytrain[min_index])
            errors.append(error)
            total_error += error
           
            f.write(f"Minimal DTW distance for generated sample {i} with score {ytrain[i]:.2f} is {min_distance} "
                    f"of the true sample {min_index} with score {ytrain[min_index]:.2f} ==> error {error}\\\ \n")
            write_min_distance_info(min_distance_file, i, min_index, ytrain)

    # Calculate mean error and standard deviation
    mean_error = np.mean(errors)
    print('mean',mean_error)
    print('total/len',total_error/len(xtrain))
    std_error = np.std(errors)
    print('std_error',std_error)
    # Save mean error and standard deviation
    with open(output_file, 'a') as f:
        f.write(f"\nMean error: {mean_error}\n")
        f.write(f"Standard deviation of error: {std_error}\n")

def write_min_distance_info(file_path, sample_index, min_index, ytrain):
    with open(file_path, 'a') as f:
        f.write(f"Minimal DTW distance for generated sample {sample_index} with score {ytrain[sample_index]:.2f}  "
                f"achieved with true sample {min_index} with score {ytrain[min_index]:.2f} \n")

            
            






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
class_index =3
# fold = 5
# for class_index in range(0,5):
for fold in range(1,6):
    xtrain = np.load(f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/generated_samples_prior.npy')
    scores= np.load(f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/scores.npy').squeeze(0).squeeze(1)
    data= np.load(f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/true_samples_{class_index}.npy')
    print(xtrain.shape,data.shape,scores.shape)
    calculate_dtw_and_save_results(xtrain, data, f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/dtw_evaluation_results_prior.txt',scores, f'../results/run_0/cross_validation/class_{class_index}/fold_{fold}/generated_samples/min_dtw_evaluation_results_prior.txt',class_index)








##############################################################################################################################################################################################





# calculate_dtw_for_labels = calculate_dtw_for_labels()
# calculate_dtw_for_labels.calculate_dtw_and_save_results(xtrain,normalize_skeletons(data),f'dtw_for_label_c{class_index}.txt')




# dtw_calculator = calculate_dtw_for_scores()
# dtw_calculator.calculate_dtw(xtrain, data, scores, class_index)
