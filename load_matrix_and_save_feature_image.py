import numpy as np
import matplotlib


file = np.load('./test.npy', allow_pickle=True)
print(file)
print(file.shape)


for i in range(784):
    image = file[i].reshape(28,28) * 255    
    matplotlib.image.imsave('./img_feature/'+ str(i) +'.png', image)



