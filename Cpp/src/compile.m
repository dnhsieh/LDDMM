includeDir = 'include';
matlabFlag = '-R2018a';

mex(matlabFlag, ['-I', includeDir], ...
    'multiplyKernelITF.cpp', 'multiplyKernel.cpp', 'polybesselk.cpp', ...
    '-output', 'multiplyKernel')

mex(matlabFlag, '-DDIM2', ['-I', includeDir], ...
    'LDDMM.cpp', ...
    'assignStructMemory.cpp', ...
    'objgrd.cpp', ...
    'polybesselk.cpp', ...
    'computeKernel.cpp', ...
    'dqKernel.cpp', ...
    'varifoldITF.cpp', ...
    'varifold2D.cpp', ...
    'LBFGS.cpp', ...
    'lineSearch.cpp', ...
    'getDirection.cpp', ...
    '-lmwblas', '-output', 'LDDMM2D')

mex(matlabFlag, '-DDIM3', ['-I', includeDir], ...
    'LDDMM.cpp', ...
    'assignStructMemory.cpp', ...
    'objgrd.cpp', ...
    'polybesselk.cpp', ...
    'computeKernel.cpp', ...
    'dqKernel.cpp', ...
    'varifoldITF.cpp', ...
    'varifold3D.cpp', ...
    'LBFGS.cpp', ...
    'lineSearch.cpp', ...
    'getDirection.cpp', ...
    '-lmwblas', '-output', 'LDDMM3D')
