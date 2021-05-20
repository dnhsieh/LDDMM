mex -R2018a computeKernelITF.cpp computeKernel.cpp polybesselk.cpp -output computeKernel
mex -R2018a multiplyKernelITF.cpp multiplyKernel.cpp polybesselk.cpp -output multiplyKernel
mex -R2018a dqKernelITF.cpp dqKernel.cpp polybesselk.cpp -output dqKernel
mex -R2018a -DDIM2 varifold2DITF.cpp varifold2D.cpp -output varifold2D
mex -R2018a -DDIM3 varifold3DITF.cpp varifold3D.cpp -output varifold3D
