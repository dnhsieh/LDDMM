function [fcnVal, grdVec] = objgrd(alpVec, iniNdeMat, disHdl, ...
                                   knlOrder, knlWidth, timeStp, timeNum)

[dimNum, ndeNum] = size(iniNdeMat);

alpStk = reshape(alpVec, dimNum, ndeNum, timeNum - 1);

ndeStk = zeros(dimNum, ndeNum, timeNum);
ndeStk(:, :, 1) = iniNdeMat;

knlStk = zeros(ndeNum, ndeNum, timeNum - 1);

fcnVal = 0;
for timeIdx = 1 : (timeNum - 1)

	ndeMat = ndeStk(:, :, timeIdx);
	alpMat = alpStk(:, :, timeIdx);

	knlMat = computeKernel(ndeMat, knlOrder, knlWidth);
	vlcMat = alpMat * knlMat;

	fcnVal = fcnVal + alpMat(:)' * vlcMat(:);

	ndeStk(:, :, timeIdx + 1) = ndeMat + timeStp * vlcMat;
	knlStk(:, :, timeIdx    ) = knlMat;

end

endNdeMat = ndeStk(:, :, timeNum);
[endFcnVal, endGrdMat] = disHdl(endNdeMat);

fcnVal = timeStp * 0.5 * fcnVal + endFcnVal;

pMat   = -endGrdMat;
grdStk = zeros(dimNum, ndeNum, timeNum - 1);

for timeIdx = (timeNum - 1) : -1 : 1

	alpMat = alpStk(:, :, timeIdx);
	ndeMat = ndeStk(:, :, timeIdx);
	knlMat = knlStk(:, :, timeIdx);

	grdStk(:, :, timeIdx) = (alpMat - pMat) * knlMat;

	pDotMat = dqKernel(ndeMat, 0.5 * alpMat - pMat, alpMat, knlOrder, knlWidth);
	pMat    = pMat - timeStp * pDotMat;

end

grdVec = timeStp * grdStk(:);

