function [fcnVal, grdMat] = varifold(dfmNdePosMat, dfmElmVtxMat, ...
                                     tgtCenPosMat, tgtUniDirMat, tgtElmVolVec, ...
                                     cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth, wgtVal)

[dimNum, dfmNdeNum] = size(dfmNdePosMat);
dfmElmIfoMat = getElmIfo(dfmElmVtxMat, dfmNdeNum);
dfmElmVtxMat = int32(dfmElmVtxMat - 1);

if dimNum == 2

	[fcnVal, grdMat] = varifold2D(dfmNdePosMat, dfmElmVtxMat, dfmElmIfoMat, ...
	                              tgtCenPosMat, tgtUniDirMat, tgtElmVolVec, ...
	                              cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth);

elseif dimNum == 3

	[fcnVal, grdMat] = varifold3D(dfmNdePosMat, dfmElmVtxMat, dfmElmIfoMat, ...
	                              tgtCenPosMat, tgtUniDirMat, tgtElmVolVec, ...
	                              cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth);

else

	error('varifold:dim', 'VARIFOLD only supports 2D and 3D.')

end

fcnVal = wgtVal * fcnVal;
grdMat = wgtVal * grdMat;
