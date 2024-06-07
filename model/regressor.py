import numpy as np
import pandas as pd
import sys
sys.path.append('/home/hferrar/HMG/dataset')
sys.path.append('/home/hferrar/HMG/urils')
from dataset import load_class
from normalize import normalize_skeletons
from aeon.distances import dtw_distance




def distance_dtw(xtrain, xtest):
    distances = np.zeros((len(xtrain), len(xtest)))

    for i in range(len(xtrain)):
        for j in range(len(xtest)):
            distance = dtw_distance(xtrain[i, :, :], xtest[j, :, :])
            distances[i, j] = distance

    return distances

def evaluate_dtw(generated_samples, generated_scores, test_samples, test_scores):
    distances = distance_dtw(generated_samples, test_samples)
    
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



def calculate_dtw(class_index):
    xtrain= np.load(f'/home/hferrar/HMG/results/run_0/generated_samples/class_{class_index}/generated_samples_prior.npy')
    scores = np.load(f'/home/hferrar/HMG/results/run_0/generated_samples/class_{class_index}/scores.npy').squeeze(0).squeeze(1)
    xtest = np.load(f'/home/hferrar/HMG/results/run_0/generated_samples/class_{class_index}/true_samples.npy')
    xtrain = np.reshape(xtrain, (xtrain.shape[0], 748, 18*3))
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 18*3))
    print(scores)

    score_errors, mean_score_error, match_count, results = evaluate_dtw(xtrain, scores, xtest, scores)
    with open(f"/home/hferrar/HMG/results/run_0/prior/dtw_evaluation_results_class_{class_index}.txt", "w") as file:
        for line in results:
            file.write(line + "\n")
    


calculate_dtw(0)
calculate_dtw(1)
calculate_dtw(2)
calculate_dtw(3)
calculate_dtw(4)


