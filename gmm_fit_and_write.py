from sklearn import mixture
from scipy.stats import multivariate_normal
import math
import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp

# calculating the embeddings for 5 points

def calculate_embedding(landmarks):
    
    # assuming len(landmarks) is 5
    
    embedding = np.zeros((60,2))
    ctr = 0
    
    for i in range(len(landmarks)):
        for j in range(len(landmarks)):
            for k in range(len(landmarks)):
                if i != j and j != k and i != k:
                
                    a = landmarks[i] - landmarks[j]
                    
                    b = math.sqrt( (landmarks[i][0] - landmarks[k][0]) ** 2 + 
                                   (landmarks[i][1] - landmarks[k][1]) ** 2 )
                     
                    embedding[ctr][0] = a[0] / b
                    embedding[ctr][1] = a[1] / b
                    
                    ctr += 1
                    
    return embedding.reshape(120)
                    
                    
def fit_GMM(all_landmarks):
    # all_landmarks is a (N,5,2) numpy array where N is the number of training samples
    N = all_landmarks.shape[0]
    all_embeddings = np.zeros((N,120))
    
    for i in range(N):
        all_embeddings[i,:] = calculate_embedding(all_landmarks[i])
    
    g = mixture.GMM(n_components = 1)
    
    g.fit(all_embeddings)
    
    weights = g.weights_ # (num_components)
    means = g.means_ # (num_components, 120)
    covars = g.covars_ # (num_components, 120)
    
    return weights, means, covars
