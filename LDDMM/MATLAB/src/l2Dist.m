function [fcnVal, grdMat] = l2Dist(ndeMat, tgtMat, wgtVal)

difMat = ndeMat - tgtMat;
fcnVal = wgtVal * 0.5 * (difMat(:)' * difMat(:));
grdMat = wgtVal * difMat;
