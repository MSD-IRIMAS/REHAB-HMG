import numpy as np
import pandas as pd
import sys
sys.path.append('../dataset')
from dataset import load_class
from aeon.distances import dtw_distance
from normalize import normalize_skeletons


#----------------------------------------------------------------DTW for datat generated based on action only------------------------------------------------------------
class calculate_dtw_for_labels:
    def calculate_dtw(self,xtrain,xtest):
        for i in range(len(xtrain)):
            for j in range(len(xtest)):
                distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
                print("DTW distance for generated sample", i,"and true sample",j,"is: ", distance)


    def calculate_dtw_and_save_results(self,xtrain, xtest, output_file):
        xtrain = xtrain
        xtrain = np.reshape(xtrain, (xtrain.shape[0], 748, 18*3))
        xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18*3))
        with open(output_file, 'w') as f:
            for i in range(len(xtrain)):
                min_distance = float('inf')
                min_index = None
                for j in range(7):
                    distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
                    f.write(f"DTW distance for generated sample {i} and true sample {j} is: {distance}\n")
                    if distance < min_distance:
                        min_distance = distance
                        min_index = j

                f.write(f"Minimal DTW distance for generated sample {i}  is: {min_distance} for the true sample {min_index}\n")
                if i == min_index: 
                        f.write(f" generated sample {i}  matches the true sample {min_index}\n")






#----------------------------------------------------------------DTW for datat generated based on action+score------------------------------------------------------------

class calculate_dtw_for_scores:
    def distance_dtw(self, xtrain, xtest):
        distances = np.zeros((len(xtrain), len(xtest)))
        for i in range(len(xtrain)):
            for j in range(len(xtest)):
                distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
                distances[i, j] = distance
        return distances

    def evaluate_dtw_with_score(self, generated_samples, generated_scores, test_samples, test_scores):
        distances = self.distance_dtw(generated_samples, test_samples)

        total_error = 0
        score_errors = []
        distance_errors = []
        closest_indices = []
        match_count = 0
        results = []
        for i in range(len(generated_samples)):
            min_index = np.argmin(distances[i])
            closest_indices.append(min_index)
            closest_score = test_scores[min_index]
            generated_score = generated_scores[i]

            score_error = abs(float(closest_score.item()) - float(generated_score.item()))  # Ensure the scores are scalars
            dtw_distance_value = distances[i, min_index]
            score_errors.append(score_error)
            distance_errors.append(dtw_distance_value)
            total_error += score_error
            result = f"Generated sample {i} with score {float(generated_score.item()):.2f} is closest to test sample {min_index} with score of {float(closest_score.item()):.2f} ==> error: {score_error:.2f}, DTW distance: {dtw_distance_value:.2f}"
            results.append(result)
            if float(generated_score.item()) == float(closest_score.item()):
                match_count += 1

        mean_score_error = total_error / len(generated_samples)
        std_score_error = np.std(score_errors)
        std_distance_error = np.std(distance_errors)

        results.append(f"\nMean score error: {mean_score_error:.2f}")
        results.append(f"Standard deviation of score errors: {std_score_error:.2f}")
        results.append(f"Standard deviation of DTW distances: {std_distance_error:.2f}")
        results.append(f"Number of matches between generated samples and closest test samples: {match_count}/{len(generated_samples)}")

        return score_errors, mean_score_error, std_score_error, std_distance_error, match_count, results

    def calculate_dtw(self, xtrain, xtest, scores, class_index):
        # Reshape the data if needed
        xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 18*3))
        xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18*3))

        score_errors, mean_score_error, std_score_error, std_distance_error, match_count, results = self.evaluate_dtw_with_score(xtrain, scores, xtest, scores)
        with open(f"../results/run_0/class_{class_index}/dtw_evaluation_results_prior.txt", "w") as file:
            for line in results:
                file.write(line + "\n")



#add the repository for the result soutput
dtw_calculator = calculate_dtw_for_scores()
dtw_calculator.calculate_dtw(xtrain, xtest, scores, class_index)


