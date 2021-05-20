function [ndeStk, vlcStk, alpStk] = LDDMM(iniNdeMat, iniAlpStk, disObj, ...
                                          knlOrder, knlWidth, timeEnd, timeStp, optObj)

[dimNum, ndeNum] = size(iniNdeMat);
timeNum = floor(timeEnd / timeStp) + 1;

d_iniNdeMat = gpuArray(iniNdeMat');

d_dfmElmIfoMat = gpuArray(getElmIfo(disObj.dfmElmVtxMat, ndeNum)');
disObj.dfmElmVtxMat = gpuArray(int32(disObj.dfmElmVtxMat - 1)');
disObj.tgtCenPosMat = gpuArray(disObj.tgtCenPosMat');
disObj.tgtElmVolVec = gpuArray(disObj.tgtElmVolVec(:));
disObj.tgtUniDirMat = gpuArray(disObj.tgtUniDirMat');

if isempty(iniAlpStk)
	d_iniAlpVec = zeros(ndeNum * dimNum * (timeNum - 1), 1, 'gpuArray');
else
	d_iniAlpVec = gpuArray(iniAlpStk(:));
end

if dimNum == 2

	d_optAlpVec = LDDMM2D(d_iniNdeMat, d_iniAlpVec, ...
	                      disObj.dfmElmVtxMat, d_dfmElmIfoMat, ...
	                      disObj.tgtCenPosMat, disObj.tgtElmVolVec, disObj.tgtUniDirMat, ...
	                      disObj.cenKnlType, disObj.cenKnlWidth, ...
	                      disObj.dirKnlType, disObj.dirKnlWidth, disObj.disWgt, ...
	                      knlOrder, knlWidth, timeStp, timeNum, ...
	                      optObj.MaxIterations, optObj.OptimalityTolerance, ...
	                      optObj.Wolfe1, optObj.Wolfe2, optObj.Columns, optObj.Verbose);

elseif dimNum == 3
	
	d_optAlpVec = LDDMM3D(d_iniNdeMat, d_iniAlpVec, ...
	                      disObj.dfmElmVtxMat, d_dfmElmIfoMat, ...
	                      disObj.tgtCenPosMat, disObj.tgtElmVolVec, disObj.tgtUniDirMat, ...
	                      disObj.cenKnlType, disObj.cenKnlWidth, ...
	                      disObj.dirKnlType, disObj.dirKnlWidth, disObj.disWgt, ...
	                      knlOrder, knlWidth, timeStp, timeNum, ...
	                      optObj.MaxIterations, optObj.OptimalityTolerance, ...
	                      optObj.Wolfe1, optObj.Wolfe2, optObj.Columns, optObj.Verbose);

else

	error('LDDMM:dim', 'LDDMM only supports 2D and 3D.');

end

d_alpStk = reshape(d_optAlpVec, ndeNum, dimNum, timeNum - 1);
[d_ndeStk, d_vlcStk] = computeNodePosition(d_iniNdeMat, d_alpStk, ...
                                           knlOrder, knlWidth, timeStp, timeNum);

ndeStk = permute(gather(d_ndeStk), [2, 1, 3]);
vlcStk = permute(gather(d_vlcStk), [2, 1, 3]);
alpStk = permute(gather(d_alpStk), [2, 1, 3]);

