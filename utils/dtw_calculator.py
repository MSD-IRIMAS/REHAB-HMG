import numpy as np
import pandas as pd

from aeon.distances import dtw_distance


class calculate_dtw_for_scores:
    def distance_dtw(self,xtrain, xtest):
        distances = np.zeros((len(xtrain), len(xtest)))
        for i in range(len(xtrain)):
            for j in range(len(xtest)):
                distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
                distances[i, j] = distance
        return distances
    def evaluate_dtw_with_score(self,generated_samples, generated_scores, test_samples, test_scores):
        distances = self.distance_dtw(generated_samples, test_samples)
        
        total_error = 0
        score_errors = []
        closest_indices = []
        match_count = 0
        results = []
        for i in range(len(generated_samples)):
            min_index = np.argmin(distances[i])
            closest_indices.append(min_index)
            closest_score = test_scores[min_index]
            score_error = abs(closest_score - generated_scores[i])
            score_errors.append(score_error)
            total_error += score_error
            result = f"Generated sample {i} with score {generated_scores[i]:.2f} is closest to test sample {min_index} with of score of {test_scores[min_index]:.2f} ==> error: {score_error:.2f}"
            results.append(result)
            if generated_scores[i] == closest_score:
                match_count += 1
        mean_score_error = total_error / len(generated_samples)
        results.append(f"\nMean score error: {mean_score_error:.2f}")
        results.append(f"\nNumber of matches between generated samples and closest test samples: {match_count}/{len(generated_samples)}")
        return score_errors, mean_score_error, match_count, results

    def calculate_dtw(self,class_index):
        xtrain= np.load(f'../results/run_0/cross_validation/fold_{class_index+1}/generated_samples/class_0/generated_samples_prior.npy')
        scores = np.load(f'../ex1_scores.npy')
        print('-----------------------------------------------',scores)
        xtest = np.load(f'../ex1_samples.npy').squeeze(0)
        xtrain = np.reshape(xtrain, (xtrain.shape[0], 748, 18*3))
        xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18*3))
        print(scores)

        score_errors, mean_score_error, match_count, results = self.evaluate_dtw_with_score(xtrain, scores, xtest, scores)
        with open(f"../results/run_0/cross_validation/fold_{class_index+1}/generated_samples/class_0/dtw_evaluation_results_prior.txt", "w") as file:
            for line in results:
                file.write(line + "\n")

calculate_dtw_for_scores= calculate_dtw_for_scores()
calculate_dtw_for_scores.calculate_dtw(4)

#######################################################################################################################################################################################################################################################################""
        
class calculate_dtw_for_labels():
    def calculate_dtw(xtrain,xtest):
        for i in range(len(xtrain)):
            for j in range(len(xtest)):
                distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
                print("DTW distance for generated sample", i,"and true sample",j,"is: ", distance)


    def write_min_distance_info(file_path, sample_index, min_index):
        with open(file_path, 'a') as f:
            f.write(f"Minimal DTW distance for generated sample {sample_index} "
                    f"achieved with true sample {min_index}  \n")

    def calculate_dtw_and_save_results(xtrain, xtest, output_file, min_distance_file):
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
                f.write(f"Minimal DTW distance for generated sample {i}  is: {min_distance}\n")
                write_min_distance_info(min_distance_file, i, min_index)





