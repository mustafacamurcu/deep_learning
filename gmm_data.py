import numpy as np
import glob
import gmm_utils
import math
import utils

root1 = '/data/vision/torralba/health-habits/other/enes/'
root2 = '/afs/csail.mit.edu/u/k/kocabey/Desktop/'

def read_data(file_path, length):
    directories = glob.glob(file_path + "*.txt")
    num_data = len(directories)
    print num_data
    all_data = np.zeros((num_data, length))

    for i in range(len(directories)):
        utils.show_progress(i,len(directories))
        f = open(directories[i])
        data = f.readlines()

        for j in range(len(data)):
            all_data[i][j] = data[j]

    print "\n"

    return all_data

def prepare_bird_representation(all_data):
    n,d = all_data.shape
    representation = np.zeros((n,2730,2))

    for i in range(n):
        utils.show_progress(i,n)
        data = np.zeros((15,3))
        for j in range(15):
            data[j][0] = all_data[i][3*j]
            data[j][1] = all_data[i][3*j+1]
            data[j][2] = all_data[i][3*j+2]

        ctr = 0
        for j in range(15):
            for k in range(15):
                for l in range(15):
                    if j != k and j != l and k != l:
                        a = [data[j][0] - data[k][0], data[j][1] - data[k][1]]
                        b = math.sqrt( (data[k][0] - data[l][0]) ** 2 + (data[k][1] - data[l][1]) ** 2 )

                        representation[i][ctr][0] = a[0]/b
                        representation[i][ctr][1] = a[1]/b

                        if data[j][2] == 0 or data[k][2] == 0 or data[l][2] == 0:
                            representation[i][ctr] = (0,0)

                        ctr += 1
    print "\n"
    return representation


def fit_bird_GMM(representation):
    n = representation.shape[0]
    X = representation.reshape((n,2730*2))
    K = 50

    [Mu,P,Var] = gmm_utils.init(X,K)
    return gmm_utils.mixGauss_part2(X,K,Mu,P,Var)
