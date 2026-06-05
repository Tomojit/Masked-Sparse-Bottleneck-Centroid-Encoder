# Masked-Sparse-Bottleneck-Centroid-Encoder
A L2-L1 based nonlinear feature selection technique with adaptive centroid update along with class-separation constraint in bottleneck layer
Instruction to use the code package.

Requirements:
1. Python: 3.9.7
2. PyTorch: 2.0.0+cu117
3. Numpy: 1.21.5
4. ipython: 8.12.10
5. sklearn: 1.2.2


Notes: 
1. Two datasets: ALLAML, and COIL20 are given with the package.

2. Seperate script for each data sets:
		ALLAML: testSABCCE_ALLAML.py
		COIL20: testSABCECE_COIL20.py

How to run the code:
1. Download the code.

2. Unzip the package. The code was compressed in MacOS BigSur Version 11.6.2.

3. Make sure all the requirements are satisfied.

4. To run the script from ipython use the commands:
    a. ipython
	b. run testSCE_ALLAML.py ==>>for ALLAMML data
	c. run testSCE_COIL20.py ==>>for COIL20 data

5. To run the script directly from python:
    a. python testSCE_ALLAML.py ==>>for ALLAMML data
    b. python testSCE_COIL20.py ==>>for COIL20 data

