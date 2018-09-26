This is the final code for the thesis "Exploring and Comparing Visualization Techniques for Neural Networks", handed in September 26th, 2018.

am_alexnet.py is the main module that needs to be executed. It will start the optimization process automatically. Per default, a random unit in the last layer is optimized using a combination of regularizations.

All relevant parameters for activation maximization can be adjusted in am_constants.py. There, the parametrizations for the regularizers can be changed and a unit for optimization can be selected.

am_tools.py contains methods helping with the optimization process, but does not need to be accessed when you simply want to use the method.

The network file is too large to be uploaded on GitHub. Please download bvlc_alexnet.npy from this source: https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
