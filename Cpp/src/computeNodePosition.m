function [ndeStk, vlcStk] = computeNodePosition(iniNdeMat, alpStk, ...
                                                knlOrder, knlWidth, timeStp, timeNum)

[dimNum, ndeNum] = size(iniNdeMat);

ndeStk = zeros(dimNum, ndeNum, timeNum    );
vlcStk = zeros(dimNum, ndeNum, timeNum - 1);

ndeStk(:, :, 1) = iniNdeMat;
for timeIdx = 1 : (timeNum - 1)

	ndeMat = ndeStk(:, :, timeIdx);
	alpMat = alpStk(:, :, timeIdx);

	vlcMat = multiplyKernel(ndeMat, alpMat, knlOrder, knlWidth);

	ndeStk(:, :, timeIdx + 1) = ndeMat + timeStp * vlcMat;
	vlcStk(:, :, timeIdx    ) = vlcMat;

end
