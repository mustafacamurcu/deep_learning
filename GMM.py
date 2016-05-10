import numpy as np
from sklearn import mixture
import glob
import math

root1 = '/data/vision/torralba/health-habits/other/enes/'
root2 = '/afs/csail.mit.edu/u/k/kocabey/Desktop/'

file_pattern = root1 + 'CelebData/SquareTest/Points/*.txt'
num_components = 10

def read_landmarks(file_pattern):
    txt = sorted( glob.glob(file_pattern))
    all_data = []
    for i in range(1000):
        f = open(txt[i],"r")
        data = f.readlines()
        temp = list(data)
        for j in range(0,10):
            data[j] = int(temp[j])
        all_data.append( data )
    return np.array(all_data)

def prepare_representation(all_data):
     n,d = all_data.shape
     representation = np.zeros((n,60,2))

     for i in range(n):
         data = np.zeros((5,2))
         for j in range(5):
             data[j][0] = all_data[i][2*j]
             data[j][1] = all_data[i][2*j+1]

         ctr = 0
         for j in range(5):
             for k in range(5):
                 for l in range(5):
                     if j != k and j != l and k != l:
                         a = data[j] - data[k]
                         b = math.sqrt( (data[k][0] - data[l][0]) ** 2 + (data[k][1] - data[l][1]) ** 2 )
                         representation[i][ctr] = a/b
                         ctr += 1
     return representation

def fit_GMM(representation,num_components):
    n = representation.shape[0]
    representation = representation.reshape((n,120))
    np.random.seed(1)
    g = mixture.GMM(num_components)
    g.fit(representation)
    return g

def save_parameters(GMM, file_path):
    f = open(file_path, 'w')

    f.write(str(len(GMM.weights_)))
    for weight in GMM.weights_:
        f.write(str(weight))

    f.write(str(len(GMM.means_)))
    for mean in GMM.means_:
        f.write(str(mean))

    f.write(str(len(GMM.covars_)))
    for covar in GMM.covars_:
        f.write(str(covar))

    f.close()
    return

all_data = read_landmarks(file_pattern)
print all_data.shape
representation = prepare_representation(all_data)
g = fit_GMM(representation,num_components)

print g.weights_
print g.means_
