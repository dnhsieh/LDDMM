function [ndeStk, vlcStk, alpStk] = LDDMM(iniNdeMat, iniAlpStk, disHdl, ...
                                          knlOrder, knlWidth, timeEnd, timeStp, optObj)

[dimNum, ndeNum] = size(iniNdeMat);
timeNum = floor(timeEnd / timeStp) + 1;

if isempty(iniAlpStk)
	iniAlpVec = zeros(dimNum * ndeNum * (timeNum - 1), 1);
else
	iniAlpVec = iniAlpStk(:);
end

objgrdHdl = @(a) objgrd(a, iniNdeMat, disHdl, knlOrder, knlWidth, timeStp, timeNum);
optAlpVec = LBFGS(objgrdHdl, iniAlpVec, optObj);

alpStk = reshape(optAlpVec, dimNum, ndeNum, timeNum - 1);
[ndeStk, vlcStk] = computeNodePosition(iniNdeMat, alpStk, ...
                                       knlOrder, knlWidth, timeStp, timeNum);
