import os
import numpy as np
def get_dirs(dir):

    for _, dirs, _ in os.walk(dir):
        return dirs



def get_weights_loss(dir):

    weights = dir.split('_')
    weights_dict = {}

    for i in range(len(weights)):

        if weights[i][0] == 'W':
            weights_dict[weights[i]] = float(weights[i+1])
            
    return weights_dict


def noise_data(data, mean=0, std_dev=0.001):
    noise = np.random.normal(mean, std_dev, data.shape)
    noisy_data = data+ noise
    return noisy_data