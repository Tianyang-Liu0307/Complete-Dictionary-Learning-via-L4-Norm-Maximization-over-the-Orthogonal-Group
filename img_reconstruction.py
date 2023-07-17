from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import matplotlib

DATADIR = "/Users/liutianyang/Downloads/apgpy-master/mnist_test/9/"
DATATYP = ".png"

def file_name(file_dir, file_type):   
    L=[]   
    for dirpath, dirnames, filenames in os.walk(file_dir):  
        for file in filenames :  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(dirpath, file))  
    return L 

def read_img():
    list = file_name(DATADIR, DATATYP)
    img_array_0 = mpimg.imread(list[0])
    array_all = img_array_0.reshape(-1)
    for i in range(1, 784):
        img_array = mpimg.imread(list[i])
        array_one = img_array.reshape(-1)
        array_all = np.vstack((array_all, array_one))
    return array_all


matrix = np.load('./test.npy', allow_pickle=True)
coder = SparseCoder(dictionary=matrix.T, transform_n_nonzero_coefs=None,
                    transform_alpha=1, transform_algorithm='lasso_lars' , positive_code=True) 
#x = coder.transform(read_img().T.conjugate())                 
x = coder.transform(read_img().T)  
for i in range(784):
    image = x[:, i].reshape(28,28) * 255    
    matplotlib.image.imsave('./mnist_test/img_reconstruction_9/'+ str(i) +'.png', image)





