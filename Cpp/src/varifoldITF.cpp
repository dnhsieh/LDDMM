#include "struct.h"

void varifold(double *, double *, double *, int *, int *, double *, double *, double *,
              char, double, char, double, double *,
              double *, double *, double *, double *, int, int, int, int);

void varifoldITF(double *vfdVal, double *dqVfdMat, double *ndeMat, fcndata &fcnObj)
{
	varifold(vfdVal, dqVfdMat,
	         ndeMat, fcnObj.dis.dfmElmVtxMat, fcnObj.dis.dfmElmIfoMat,
	         fcnObj.dis.tgtCenPosMat, fcnObj.dis.tgtUniDirMat, fcnObj.dis.tgtElmVolVec,
	         fcnObj.dis.cenKnlType, fcnObj.dis.cenKnlWidth,
	         fcnObj.dis.dirKnlType, fcnObj.dis.dirKnlWidth,
	         fcnObj.dis.dfmCenPosMat, fcnObj.dis.dfmUniDirMat, fcnObj.dis.dfmElmVolVec,
	         fcnObj.dis.dcVfdMat, fcnObj.dis.ddVfdMat,
	         fcnObj.dis.dfmNdeNum, fcnObj.dis.dfmElmNum, fcnObj.dis.dfmElmIfoNum, fcnObj.dis.tgtElmNum);

	return;
}
