#include "struct.h"
#include "constants.h"

long long assignOptStructMemory(optdata &optObj, double *d_optWorkspace)
{
	long long memCnt = 0;

	// ---

	int varNum = optObj.varNum;
	int memNum = optObj.memNum;

	double *d_nowPtr = d_optWorkspace;

	optObj.d_dspMat = d_nowPtr;
	d_nowPtr += varNum * memNum;
	memCnt   += varNum * memNum;

	optObj.d_dgdMat = d_nowPtr;
	d_nowPtr += varNum * memNum;
	memCnt   += varNum * memNum;

	optObj.d_dirVec = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_posNxt = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_grdNxt = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_dspVec = d_nowPtr;
	d_nowPtr += varNum;
	memCnt   += varNum;

	optObj.d_dgdVec = d_nowPtr;
	memCnt   += varNum;

	return memCnt;
}

long long assignFcnStructMemory(fcndata &fcnObj, double *d_fcnWorkspace)
{
	long long memCnt = 0;

	// ---

	int ndeNum    = fcnObj.prm.ndeNum;
	int dfmElmNum = fcnObj.dis.dfmElmNum;
	int timeNum   = fcnObj.prm.timeNum;

	double *d_nowPtr = d_fcnWorkspace;

	fcnObj.dis.d_vfdVec = d_nowPtr;
	d_nowPtr += dfmElmNum + fcnObj.dis.tgtElmNum;
	memCnt   += dfmElmNum + fcnObj.dis.tgtElmNum;

	fcnObj.dis.d_sumBufVec = d_nowPtr;
	d_nowPtr += SUMBLKDIM;
	memCnt   += SUMBLKDIM;

	fcnObj.dis.d_dfmCenPosMat = d_nowPtr;
	d_nowPtr += DIMNUM * dfmElmNum;
	memCnt   += DIMNUM * dfmElmNum;

	fcnObj.dis.d_dfmUniDirMat = d_nowPtr;
	d_nowPtr += DIMNUM * dfmElmNum;
	memCnt   += DIMNUM * dfmElmNum;

	fcnObj.dis.d_dfmElmVolVec = d_nowPtr;
	d_nowPtr += dfmElmNum;
	memCnt   += dfmElmNum;

	fcnObj.dis.d_dqVfdMat = d_nowPtr;
	d_nowPtr += DIMNUM * ndeNum;
	memCnt   += DIMNUM * ndeNum;

	fcnObj.dis.d_dcVfdMat = d_nowPtr;
	d_nowPtr += DIMNUM * dfmElmNum;
	memCnt   += DIMNUM * dfmElmNum;

	fcnObj.dis.d_ddVfdMat = d_nowPtr;
	d_nowPtr += DIMNUM * dfmElmNum;
	memCnt   += DIMNUM * dfmElmNum;

	fcnObj.d_ndeStk = d_nowPtr;
	d_nowPtr += DIMNUM * ndeNum * timeNum;
	memCnt   += DIMNUM * ndeNum * timeNum;

	fcnObj.d_knlMat = d_nowPtr;
	d_nowPtr += ndeNum * ndeNum;
	memCnt   += ndeNum * ndeNum;

	fcnObj.d_vlcMat = d_nowPtr;
	d_nowPtr += DIMNUM * ndeNum;
	memCnt   += DIMNUM * ndeNum;

	fcnObj.d_pMat = d_nowPtr;
	d_nowPtr += DIMNUM * ndeNum;
	memCnt   += DIMNUM * ndeNum;

	fcnObj.d_ampMat = d_nowPtr;
	d_nowPtr += DIMNUM * ndeNum;
	memCnt   += DIMNUM * ndeNum;

	fcnObj.d_pDotMat = d_nowPtr;
	memCnt   += DIMNUM * ndeNum;

	return memCnt;
}
