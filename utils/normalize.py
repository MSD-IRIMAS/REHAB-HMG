import numpy as np
def normalize_skeletons(X, min_X=None, max_X=None, min_Y=None, max_Y=None, min_Z=None, max_Z=None):
    """to be applied on original data, not on reshaped data"""

    n_X = np.zeros(shape=X.shape)

    if min_X is None:
        min_X = np.min(X[:,:,:,0])
    
    if max_X is None:
        max_X = np.max(X[:,:,:,0])

    if min_Y is None:
        min_Y = np.min(X[:,:,:,1])
    
    if max_Y is None:
        max_Y = np.max(X[:,:,:,1])

    if min_Z is None:
        min_Z = np.min(X[:,:,:,2])
    
    if max_Z is None:
        max_Z = np.max(X[:,:,:,2])

    n_X[:,:,:,0] = (X[:,:,:,0] - min_X) / (1.0 * (max_X - min_X))
    n_X[:,:,:,1] = (X[:,:,:,1] - min_Y) / (1.0 * (max_Y - min_Y))
    n_X[:,:,:,2] = (X[:,:,:,2] - min_Z) / (1.0 * (max_Z - min_Z))

    # np.savez('data/min_max_values_train.npz', min_X=min_X, max_X=max_X, min_Y=min_Y, max_Y=max_Y, min_Z=min_Z, max_Z=max_Z)
    return n_X,min_X, max_X,min_Y,max_Y, min_Z,max_Z


def unnormalize_generated_skeletons(X_normalized,min_X=None, max_X=None, min_Y=None, max_Y=None, min_Z=None, max_Z=None):
    """Unnormalize the generated skeleton data."""
    X_unnormalized = np.zeros_like(X_normalized)
    # min_max_values = np.load('data/min_max_values.npz')
    # min_X = min_max_values['min_X']
    # max_X = min_max_values['max_X']
    # min_Y = min_max_values['min_Y']
    # max_Y = min_max_values['max_Y']
    # min_Z = min_max_values['min_Z']
    # max_Z = min_max_values['max_Z']
    X_unnormalized[:, :, :, 0] = X_normalized[:, :, :, 0] * (max_X - min_X) + min_X
    X_unnormalized[:, :, :, 1] = X_normalized[:, :, :, 1] * (max_Y - min_Y) + min_Y
    X_unnormalized[:, :, :, 2] = X_normalized[:, :, :, 2] * (max_Z - min_Z) + min_Z
    
    return X_unnormalized



def normalize_scores(scores):
    normalized_scores = np.zeros_like(scores)
   
    normalized_scores = scores / 100.0
    return normalized_scores


