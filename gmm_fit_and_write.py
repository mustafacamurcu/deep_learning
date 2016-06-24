from sklearn import mixture
import math
import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp
import glob
import utils
import random
# calculating the embeddings for 5 points

def read_data(file_path, length):
    directories = glob.glob(file_path + "*.txt")
    num_data = len(directories)
    print num_data
    sampled_directories = random.sample(directories, 1000)
    num_data = len(sampled_directories)
    print num_data
    all_data = np.zeros((num_data, length,2))

    for i in range(num_data):
        utils.show_progress(i,num_data)
	f = open(directories[i])
        data = f.readlines()

        for j in range(len(data)):
	    if j%2 == 0:
                all_data[i][j/2][0] = data[j]
	    else:
		all_data[i][j/2][1] = data[j]

    print "\n"
    return all_data

all_data = read_data('/data/vision/torralba/health-habits/other/enes/CelebData/RSTrain/Points/', 5)
print all_data.shape

def calculate_embedding(landmarks):
    
    # assuming len(landmarks) is 5
    
    embedding = np.zeros((120))
    ctr = 0
    
    for i in range(len(landmarks)):
        for j in range(len(landmarks)):
            for k in range(len(landmarks)):
                if i != j and j != k and i != k:
                
                    a = landmarks[i] - landmarks[j]
                    
                    b = math.sqrt( (landmarks[i][0] - landmarks[k][0]) ** 2 + 
                                   (landmarks[i][1] - landmarks[k][1]) ** 2 )
                     
                    embedding[ctr] = a[0] / b
                    ctr += 1

                    embedding[ctr] = a[1] / b
                    ctr += 1
                    
    return embedding
                    
                    
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


weights, means, covars = fit_GMM(all_data)

np.save('weights.npy',weights)
np.save('means.npy',means)
np.save('covars.npy',covars)
