function [d_ndeStk, d_vlcStk] = computeNodePosition(d_iniNdeMat, d_alpStk, ...
                                                    knlOrder, knlWidth, timeStp, timeNum)

[ndeNum, dimNum] = size(d_iniNdeMat);

d_ndeStk = zeros(ndeNum, dimNum, timeNum    , 'gpuArray');
d_vlcStk = zeros(ndeNum, dimNum, timeNum - 1, 'gpuArray');

d_ndeStk(:, :, 1) = d_iniNdeMat;
for timeIdx = 1 : (timeNum - 1)

	d_ndeMat = d_ndeStk(:, :, timeIdx);
	d_alpMat = d_alpStk(:, :, timeIdx);

	d_vlcMat = multiplyKernel(d_ndeMat, d_alpMat, knlOrder, knlWidth);

	d_ndeStk(:, :, timeIdx + 1) = d_ndeMat + timeStp * d_vlcMat;
	d_vlcStk(:, :, timeIdx    ) = d_vlcMat;

end
