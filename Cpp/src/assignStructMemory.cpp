#include "struct.h"
#include "constants.h"

long long assignOptStructMemory(optdata &optObj, double *optWorkspace)
{
	long long memCnt = 0;

	// ---

	int varNum = optObj.varNum;
	int memNum = optObj.memNum;

	double *nowPtr = optWorkspace;

	optObj.dspMat = nowPtr;
	nowPtr += varNum * memNum;
	memCnt += varNum * memNum;

	optObj.dgdMat = nowPtr;
	nowPtr += varNum * memNum;
	memCnt += varNum * memNum;

	optObj.dirVec = nowPtr;
	nowPtr += varNum;
	memCnt += varNum;

	optObj.posNxt = nowPtr;
	nowPtr += varNum;
	memCnt += varNum;

	optObj.grdNxt = nowPtr;
	nowPtr += varNum;
	memCnt += varNum;

	optObj.dspVec = nowPtr;
	nowPtr += varNum;
	memCnt += varNum;

	optObj.dgdVec = nowPtr;
	nowPtr += varNum;
	memCnt += varNum;

	optObj.recVec = nowPtr;
	memCnt += memNum;

	return memCnt;
}

long long assignFcnStructMemory(fcndata &fcnObj, double *fcnWorkspace)
{
	long long memCnt = 0;

	// ---

	int ndeNum    = fcnObj.prm.ndeNum;
	int dfmElmNum = fcnObj.dis.dfmElmNum;
	int timeNum   = fcnObj.prm.timeNum;

	double *nowPtr = fcnWorkspace;

	fcnObj.dis.dfmCenPosMat = nowPtr;
	nowPtr += DIMNUM * dfmElmNum;
	memCnt += DIMNUM * dfmElmNum;

	fcnObj.dis.dfmUniDirMat = nowPtr;
	nowPtr += DIMNUM * dfmElmNum;
	memCnt += DIMNUM * dfmElmNum;

	fcnObj.dis.dfmElmVolVec = nowPtr;
	nowPtr += dfmElmNum;
	memCnt += dfmElmNum;

	fcnObj.dis.dqVfdMat = nowPtr;
	nowPtr += DIMNUM * ndeNum;
	memCnt += DIMNUM * ndeNum;

	fcnObj.dis.dcVfdMat = nowPtr;
	nowPtr += DIMNUM * dfmElmNum;
	memCnt += DIMNUM * dfmElmNum;

	fcnObj.dis.ddVfdMat = nowPtr;
	nowPtr += DIMNUM * dfmElmNum;
	memCnt += DIMNUM * dfmElmNum;

	fcnObj.ndeStk = nowPtr;
	nowPtr += DIMNUM * ndeNum * timeNum;
	memCnt += DIMNUM * ndeNum * timeNum;

	fcnObj.knlStk = nowPtr;
	nowPtr += ndeNum * ndeNum * (timeNum - 1);
	memCnt += ndeNum * ndeNum * (timeNum - 1);

	fcnObj.vlcMat = nowPtr;
	nowPtr += DIMNUM * ndeNum;
	memCnt += DIMNUM * ndeNum;

	fcnObj.pMat = nowPtr;
	nowPtr += DIMNUM * ndeNum;
	memCnt += DIMNUM * ndeNum;

	fcnObj.ampMat = nowPtr;
	nowPtr += DIMNUM * ndeNum;
	memCnt += DIMNUM * ndeNum;

	fcnObj.pDotMat = nowPtr;
	memCnt += DIMNUM * ndeNum;

	return memCnt;
}
