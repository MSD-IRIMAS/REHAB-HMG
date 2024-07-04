import os
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