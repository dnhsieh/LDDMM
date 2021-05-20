#include "struct.h"

void varifold(double *, double *, double *, int *, int *, double *, double *, double *,
              char, double, char, double, double *, double *, double *,
              double *, double *, double *, double *, int, int, int);

void varifoldITF(double *h_vfdPtr, double *d_dqVfdMat, double *d_ndeMat, fcndata &fcnObj)
{
	varifold(h_vfdPtr, d_dqVfdMat,
	         d_ndeMat, fcnObj.dis.d_dfmElmVtxMat, fcnObj.dis.d_dfmElmIfoMat,
	         fcnObj.dis.d_tgtCenPosMat, fcnObj.dis.d_tgtUniDirMat, fcnObj.dis.d_tgtElmVolVec,
	         fcnObj.dis.cenKnlType, fcnObj.dis.cenKnlWidth,
	         fcnObj.dis.dirKnlType, fcnObj.dis.dirKnlWidth,
	         fcnObj.dis.d_dfmCenPosMat, fcnObj.dis.d_dfmUniDirMat, fcnObj.dis.d_dfmElmVolVec,
	         fcnObj.dis.d_vfdVec, fcnObj.dis.d_sumBufVec, fcnObj.dis.d_dcVfdMat, fcnObj.dis.d_ddVfdMat,
	         fcnObj.dis.dfmNdeNum, fcnObj.dis.dfmElmNum, fcnObj.dis.tgtElmNum);

	return;
}
