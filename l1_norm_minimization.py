import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
#np.set_printoptions(threshold=np.inf)
import sys
sys.path.append(r'/Users/liutianyang/Downloads/apgpy-master')
import apgpy as apg
import datetime
from numpy.linalg import norm, svd

#DATADIR = "/Users/liutianyang/Documents/压缩闭环转录/src_method/CroppedYale/"
#DATATYP = ".pgm"
DATADIR = "/Users/liutianyang/Downloads/分类任务/分类任务1/train/"
DATATYP = ".jpg"

DATATEST = "/Users/liutianyang/Downloads/分类任务/分类任务1/val/Fruit/Apple/Golden-Delicious/Golden-Delicious_001.jpg"

# this two functions are used to normalize the data
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# this function is used to get all of the image path
# in this function the input is the directory name and the image type
def file_name(file_dir, file_type):   
    L=[]   
    for dirpath, dirnames, filenames in os.walk(file_dir):  
        for file in filenames :  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(dirpath, file))  
    return L 


# this function is used to resize the image and return the matrix array
# the matrix array is the input of the model
# in this function the input is the directory name, the image type, the row size and the column size
# attention: before each row matrix combine to the matrix array, it should be normalized
def resize_img(DATADIR, img_size_row, img_size_col):
    list = file_name(DATADIR, DATATYP)
    img_array_0 = cv2.imread(list[0])
    img_array_0 = cv2.cvtColor(img_array_0, cv2.COLOR_BGR2GRAY)
    new_array_0 = cv2.resize(img_array_0, (img_size_row, img_size_col))
    array_all = new_array_0.reshape(-1)
    array_all = normalization(array_all)
    for i in range(1, len(list)):
        img_array = cv2.imread(list[i])
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        new_array = cv2.resize(img_array, (img_size_row, img_size_col))
        array_one = new_array.reshape(-1)
        array_one = normalization(array_one)
        array_all = np.vstack((array_all, array_one))
    return array_all

def resize_img_test(DATATEST, img_size_row, img_size_col):
    img_array_0 = cv2.imread(DATATEST)
    img_array_0 = cv2.cvtColor(img_array_0, cv2.COLOR_BGR2GRAY)
    new_array_0 = cv2.resize(img_array_0, (img_size_row, img_size_col))
    array_all = new_array_0.reshape(-1)
    array_all = normalization(array_all)
    return array_all

# this function is used to random sample the data
# the input is the matrix array(the entire dataset matrix)
# the output is the train set and the test set
def random_sample(array_all):
    row_random_array = np.arange(array_all.shape[0])
    np.random.shuffle(row_random_array)
    array_train = array_all[row_random_array[:int(array_all.shape[0]*0.5)], :]
    array_test = array_all[row_random_array[int(array_all.shape[0]*0.5):], :]
    return array_train, array_test

def inexact_augmented_lagrange_multiplier(X, lmbda = 0.01, tol = 1e-7, maxIter = 1000):
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y /dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n= Y.shape[1]
    itr = 0
    while True:
        Eraw = X - A + (1/mu) * Y
        #E(k+1) = arg min(E) L(A(k), E, Y(k), u(k))
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        #SVD eigendecomposition
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(0.05 * n), n])
        #A(k+1) = arg min(A) L(A, E(k), Y(k), u(k))
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        
        A = Aupdate

        E = Eupdate
        #(D - A(k+1) - Y(K) / u(k))
        Z = X - A - E
        #Y(k+1) = Y(k) + u(k) * (D - A(k+1) - Y(K) / u(k))
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxIter):
            break
    print("IALM Finished at iteration %d" % (itr))
    return A, E

#def residuals(A, E, X):


def main():

    #data1, data2 = random_sample(resize_img(DATADIR, 12, 10))
    data1 = resize_img(DATADIR, 12*5, 10*5)
    data2 = resize_img_test(DATATEST, 12*5, 10*5)
    mu = 1.
    A = data1.T
    b = data2.T
    n = 2640

    AtA = np.dot(A.T, A)
    Atb = np.dot(A.T, b)

    def quad_grad(y):
        return np.dot(AtA, y) - Atb

    def soft_thresh(y, t):
        return np.sign(y) * np.maximum(abs(y) - t * mu, 0.)

    starttime = datetime.datetime.now()
    x = apg.solve(quad_grad, soft_thresh, np.zeros(n), eps=1e-8, gen_plots=True, quiet=True)
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    res = np.where(x > 0.05)
    print(res)
    list = file_name(DATADIR, DATATYP)
    

    for i in range(len(res[0])):
        #print(res[0][i])
        print(list[res[0][i]])
        plt.figure()
        plt.imshow(data1[res[0][i], :].reshape(12*5, 10*5), cmap='gray')
        #plt.imshow(data1[res[0][i], :].reshape(12*5, 10*5, 3))
        plt.show()
    #print(x.shape)
    #print(x.max())
    
    plt.figure()
    plt.plot(x)
    plt.show()

def get_data_information():
    print("The data information is as follows:")
    print("The data size is: ", resize_img(DATADIR, 12, 10).shape)
    #print(file_name(DATADIR, DATATYP))
    


if __name__ == '__main__':
    main()
    #get_data_information()