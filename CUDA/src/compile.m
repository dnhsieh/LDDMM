includeDir = 'include';
libraryDir = '/usr/local/cuda/lib64';
matlabFlag = '-R2018a';

mexcuda(matlabFlag, '-DDIM2', ['-I', includeDir], ...
        'multiplyKernelITF.cu', 'multiplyKernel2D.cu', ...
        '-output', 'multiplyKernel2D')

mexcuda(matlabFlag, '-DDIM3', ['-I', includeDir], ...
        'multiplyKernelITF.cu', 'multiplyKernel3D.cu', ...
        '-output', 'multiplyKernel3D')

mexcuda(matlabFlag, '-DDIM2', ['-L', libraryDir], ['-I', includeDir], ...
        'LDDMM.cu', ...
        'assignStructMemory.cu', ...
        'objgrd.cu', ...
	     'xpby.cu', ...
	     'scaleVector.cu', ...
        'computeKernel2D.cu', ...
        'dqKernel2D.cu', ...
        'varifoldITF.cu', ...
        'varifold2D.cu', ...
        'dsum.cu', ...
        'LBFGS.cu', ...
        'lineSearch.cu', ...
        'getDirection.cu', ...
        '-lcublas', '-output', 'LDDMM2D')

mexcuda(matlabFlag, '-DDIM3', ['-L', libraryDir], ['-I', includeDir], ...
        'LDDMM.cu', ...
        'assignStructMemory.cu', ...
        'objgrd.cu', ...
	     'xpby.cu', ...
	     'scaleVector.cu', ...
        'computeKernel3D.cu', ...
        'dqKernel3D.cu', ...
        'varifoldITF.cu', ...
        'varifold3D.cu', ...
        'dsum.cu', ...
        'LBFGS.cu', ...
        'lineSearch.cu', ...
        'getDirection.cu', ...
        '-lcublas', '-output', 'LDDMM3D')

