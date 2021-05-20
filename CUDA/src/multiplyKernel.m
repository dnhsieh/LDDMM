function d_vlcMat = multiplyKernel(d_ndeMat, d_alpMat, knlOrder, knlWidth)

dimNum = size(d_ndeMat, 2);

if dimNum == 2
	d_vlcMat = multiplyKernel2D(d_ndeMat, d_alpMat, knlOrder, knlWidth);
elseif dimNum == 3
	d_vlcMat = multiplyKernel3D(d_ndeMat, d_alpMat, knlOrder, knlWidth);
else
	error('multiplyKernel:dim', 'MULTIPLYKERNEL only supports 2D and 3D.')
end
