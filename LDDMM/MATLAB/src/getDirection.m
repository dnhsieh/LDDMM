function dirVec = getDirection(HIniVal, grdVec, sMat, yMat, newIdx, hisNum, memNum)
%Algorithm 7.4 in Nocedal
%
% Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
% Date  : 05/20/2020

% s   = x_next - x_now = dspVec
% y   = (grad f)_next - (grad f)_now = dgdVec
% rho = 1 / (s^T y)

dirVec = grdVec;

alpVec = zeros(memNum, 1);
for hisCnt = 0 : (hisNum - 1)

	hisIdx = newIdx - hisCnt;
	if hisIdx < 1
		hisIdx = hisIdx + memNum;
	end

	sVec   = sMat(:, hisIdx);
	yVec   = yMat(:, hisIdx);

	alpVal = (sVec' * dirVec) / (sVec' * yVec);
	dirVec = dirVec - alpVal * yVec;

	alpVec(hisIdx) = alpVal;

end

dirVec = HIniVal * dirVec;

if hisNum < memNum
	oldIdx = 1;
elseif newIdx == memNum
	oldIdx = 1;
else
	oldIdx = newIdx + 1;
end
	
for hisCnt = 0 : (hisNum - 1)

	hisIdx = oldIdx + hisCnt;
	if hisIdx > memNum
		hisIdx = hisIdx - memNum;
	end

	alpVal = alpVec(hisIdx);

	sVec   = sMat(:, hisIdx);
	yVec   = yMat(:, hisIdx);

	btaVal = (yVec' * dirVec) / (sVec' * yVec);
	dirVec = dirVec + (alpVal - btaVal) * sVec;

end

dirVec = -dirVec;

