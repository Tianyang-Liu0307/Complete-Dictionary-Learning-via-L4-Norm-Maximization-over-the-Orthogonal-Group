import matplotlib.pyplot as plt
import numpy as np
#np.set_printoptions(threshold=np.inf)
import time
from numpy.linalg import norm, svd
import struct
import os
import imageio
from tqdm import tqdm
from natsort import natsorted

# this two functions are used to normalize the data
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def load_images(file_name):
    ##   在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它。##
    ##   file object = open(file_name [, access_mode][, buffering])          ##
    ##   file_name是包含您要访问的文件名的字符串值。                         ##
    ##   access_mode指定该文件已被打开，即读，写，追加等方式。               ##
    ##   0表示不使用缓冲，1表示在访问一个文件时进行缓冲。                    ##
    ##   这里rb表示只能以二进制读取的方式打开一个文件                        ##
    binfile = open(file_name, 'rb') 
    ##   从一个打开的文件读取数据
    buffers = binfile.read()
    ##   读取image文件前4个整型数字
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    ##   整个images数据大小为60000*28*28
    bits = num * rows * cols
    ##   读取images数据
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    ##   关闭文件
    binfile.close()
    ##   转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    return images

def load_labels(file_name):
    ##   打开文件
    binfile = open(file_name, 'rb')
    ##   从一个打开的文件读取数据    
    buffers = binfile.read()
    ##   读取label文件前2个整形数字，label的长度为num
    magic,num = struct.unpack_from('>II', buffers, 0) 
    ##   读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    ##   关闭文件
    binfile.close()
    ##   转换为一维数组
    labels = np.reshape(labels, [num])
    return labels   

filename_train_images = '/Users/liutianyang/Downloads/mnist_dataset/train-images-idx3-ubyte'
filename_train_labels = '/Users/liutianyang/Downloads/mnist_dataset/train-labels-idx1-ubyte'
filename_test_images = '/Users/liutianyang/Downloads/mnist_dataset/t10k-images-idx3-ubyte'
filename_test_labels = '/Users/liutianyang/Downloads/mnist_dataset/t10k-labels-idx1-ubyte'
train_images=load_images(filename_train_images)
train_labels=load_labels(filename_train_labels)
test_images=load_images(filename_test_images)
test_labels=load_labels(filename_test_labels)

'''
def resize_img(img_size_row, img_size_col):
    img_array_0 = train_images[0]
    #img_array_0 = cv2.cvtColor(img_array_0, cv2.COLOR_BGR2GRAY)
    new_array_0 = cv2.resize(img_array_0, (img_size_row, img_size_col))
    array_all = new_array_0.reshape(-1)
    array_all = normalization(array_all)
    for i in range(1, len(train_images)):
        img_array = train_images[i]
        #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        new_array = cv2.resize(img_array, (img_size_row, img_size_col))
        array_one = new_array.reshape(-1)
        array_one = normalization(array_one)
        array_all = np.vstack((array_all, array_one))
    return array_all

fig=plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(30):
    images = np.reshape(train_images[i], [28,28])
    images = normalization(images)
    print(images)
    ax=fig.add_subplot(6,5,i+1,xticks=[],yticks=[])
    ax.imshow(images,cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,7,str(train_labels[i]))
plt.show()
'''

def save_loss_line_to_gif(loss_img_path: str, gif_img_path: str) -> None:
    if not os.path.exists(loss_img_path):
        print('no data to merge.')
        return
    """ get all image by nature order """
    img_list = natsorted(os.listdir(loss_img_path))
    gif_buffer = []
    for img_name in tqdm(img_list):
        """ because plt.savefig() save image's suffix is jpg """
        if img_name.split('.')[-1] != 'jpg':
            continue
        img_path = os.path.join(loss_img_path, img_name)
        gif_buffer.append(imageio.imread(img_path))
    imageio.mimsave(gif_img_path, gif_buffer, 'GIF', duration=0.1)

def loss_function(A, Y, n, p, theta):
    return abs(1 - np.sum((np.dot(A, Y)) ** 4) / (3 * n * p * theta))

def loss_function_dictionary(A, Y, n):
    return (1 - np.sum((np.dot(A, Y)) ** 4) / n)

def msp_algorithm(Y, n, p, theta, tol=0.0035, epoch_num: int = 150):

    A_derivative = normalization(np.random.rand(n,n))
    epoch = 0

    epoch_num = epoch_num
    """ create dynamic figure """
    loss_list = []
    epoch_list = []
    # open interactive
    plt.ion()

    #print(loss_function(A_derivative, Y, n, p, theta))
    while loss_function(A_derivative, Y, n, p, theta) > tol:
    #while loss_function_dictionary(A_derivative, Y, n) > tol:
    #while t < 10:
        A_derivative = 4 * np.dot((np.dot(A_derivative, Y))**3, Y.T.conjugate())
        u, s, v = np.linalg.svd(A_derivative)
        A_derivative = np.dot(u, v)
        epoch = epoch + 1

        #print(epoch)
        #print(loss_function(A_derivative, Y, n, p, theta))
        #print(loss_function_dictionary(A_derivative, Y, n))
        print(A_derivative)
        #print(" ")
        loss_list.append(loss_function(A_derivative, Y, n, p, theta).item())
        epoch_list.append(epoch)

        """ dynamic show image. """
        plt.clf()		# clear figure axis
        plt.plot(epoch_list, loss_list, 'r-')
        plt.title("msp_algorithm_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.pause(0.1)	# pause 100ms
        """ save img file """
        save_img_path = "./img/{:0>4d}.jpg".format(epoch)
        plt.savefig(save_img_path)

        print('\r Epoch: {:>3.0f}%[{}->{}], loss: {}'.format(epoch * 100 / (epoch_num - 1),
                                                             int(epoch / 10) * '*',
                                                             (int(epoch_num / 10) - 1 - int(epoch / 10)) * '.',
                                                             loss_function(A_derivative, Y, n, p, theta).item()), end='')
        print(" ")
    # close interactive
    plt.ioff()

    return A_derivative


T1 = time.time()
matrix = msp_algorithm(normalization(train_images.T.conjugate()), 784, 60000, 0.8)
#matrix = msp_algorithm(train_images.T.conjugate(), 784, 60000, 0.1)
#matrix = msp_algorithm(np.identity(3), 3, 1, 1)
T2 = time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))

np.save('./test.npy', matrix)
#file = np.load('./test.npy', allow_pickle=True)

loss_image_path = "./img1"
save_gif_path = "./img1/res.gif"
save_loss_line_to_gif(loss_image_path, save_gif_path)

