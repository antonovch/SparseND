# SparseND
Sparse ND arrays in Matlab

Behavior of the SparseND objects is designed to mimic the one of built-in sparse matrix class of Matlab, which in term should be indistinguishable from numerical or logical matrices from the user's point of view.

There are two ways of crating a SparseND object, S:
1. From a full matrix F, by simply passing it to the constructor: `S = SparseND(F)`,
2. From non-zero values vector, `v`, their positions, specified as a separate vector for each dimension, and scalar dimension sizes: `S = SparseND(i1,i2,i3...,v,dim1,dim2,dim3,...)`.

The code works with Matlab R2020a and up. To test it, one can run the `test_SparseND` before use. See inside for the parameters, such as the number of tests to perform, max length of the arrays (memory load), sparseness etc. 

All commonly used methods of numerical/logical arrays are overloaded for SparseND. Many use built-in functions and sparse class through correct reshaping of data into a vector and back. The ones that are implemented fully, as well as a couple of the ones that leverage built-in methods, are included in `test_SparseND`. 
