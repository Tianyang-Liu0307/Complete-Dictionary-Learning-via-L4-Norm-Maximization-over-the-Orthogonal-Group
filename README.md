# Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group

This work is mainly based on the code reproduction of the article “Complete Dictionary Learning via L4-Norm Maximization over the Orthogonal Group”.

Due to my limited level, the method of reconstructing the numbers is not given in the article, so my work is limited to the part that reproduces the learned dictionary, and the formal method of reconstructing the numbers in the MNIST dataset set has not been implemented.

But I cleverly reconstructed the figure using other methods, I took the main diagonal elements in the eigenmatrix and visualized that to get the reconstructed figure. Although this method is not reasonable, I think it is very reliable proof that it has learned the characteristics of the data.

The file contains the MSP algorithm and the L1 norm minimization algorithm, respectively named after the Python file name.

The following are some pictures of experimental results, which can be generated through the provided Python files.

MSP algorithm convergence graph:
--------------

![res](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/bb20e504-4887-46ee-b036-eb5e121c0dee)

Reconstructed MNIST dataset picture:
--------------

![9](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/9644ea69-4493-43a3-aaa4-5e1fb0a96bb5)
![8](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/b39b1c9c-d680-4a0e-9c08-a86aac4bd288)
![7](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/bc105d4d-293e-4491-afb4-2ca9a626f5e3)
![6](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/4ca8b507-4bb5-4724-8952-44dd43035867)
![5](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/e85812a6-a376-4241-8341-0a79c345eb01)
![4](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/b4c1d881-8dac-4914-85a5-f04f32e2e6e8)
![3](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/f2f87a68-9d81-4a04-8a47-985020ce4856)
![2](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/4b6d9c84-25ec-434c-a64a-279e80cba615)
![1](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/32580a97-9bd8-4d55-8712-08226c62f092)
![0](https://github.com/Tianyang-Liu0307/Complete-Dictionary-Learning-via-L4-Norm-Maximization-over-the-Orthogonal-Group/assets/57581285/bbc5d9fe-03ec-447a-9efd-25f783a7d2b8)




References
=====
Zhai, Y., Yang, Z., Liao, Z., Wright, J. and Ma, Y., 2020. Complete dictionary learning via l 4-norm maximization over the orthogonal group. The Journal of Machine Learning Research, 21(1), pp.6622-6689.

The optimization algorithm uses APG (Accelerate Proximal Gradient), and the implementation of this algorithm is quoted from: https://github.com/bodono/apgpy.
