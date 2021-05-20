#include <cstring>
#include "mex.h"
#include "struct.h"
#include "constants.h"

long long assignFcnStructMemory(fcndata &, double *);
long long assignOptStructMemory(optdata &, double *);

int LBFGS(double *, double *, double *, optdata &, fcndata &);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	fcndata fcnObj;
	optdata optObj;

	fcnObj.prm.iniNdeMat    =      mxGetDoubles(prhs[ 0]);
	double    *iniAlpVec    =      mxGetDoubles(prhs[ 1]);
	fcnObj.dis.dfmElmVtxMat = (int *) mxGetData(prhs[ 2]);
	fcnObj.dis.dfmElmIfoMat = (int *) mxGetData(prhs[ 3]);
	fcnObj.dis.tgtCenPosMat =      mxGetDoubles(prhs[ 4]);
	fcnObj.dis.tgtElmVolVec =      mxGetDoubles(prhs[ 5]);
	fcnObj.dis.tgtUniDirMat =      mxGetDoubles(prhs[ 6]);
	fcnObj.dis.cenKnlType   =       mxGetScalar(prhs[ 7]);
	fcnObj.dis.cenKnlWidth  =       mxGetScalar(prhs[ 8]); 
	fcnObj.dis.dirKnlType   =       mxGetScalar(prhs[ 9]);
	fcnObj.dis.dirKnlWidth  =       mxGetScalar(prhs[10]);
	fcnObj.dis.disWgt       =       mxGetScalar(prhs[11]);
	fcnObj.prm.knlOrder     =       mxGetScalar(prhs[12]);
	fcnObj.prm.knlWidth     =       mxGetScalar(prhs[13]);
	fcnObj.prm.timeStp      =       mxGetScalar(prhs[14]);
	fcnObj.prm.timeNum      =       mxGetScalar(prhs[15]);
	optObj.itrMax           =       mxGetScalar(prhs[16]);
	optObj.tolVal           =       mxGetScalar(prhs[17]);
	optObj.wolfe1           =       mxGetScalar(prhs[18]);
	optObj.wolfe2           =       mxGetScalar(prhs[19]);
	optObj.memNum           =       mxGetScalar(prhs[20]);
	optObj.vbsFlg           =       mxGetScalar(prhs[21]);

	// ---

	fcnObj.prm.ndeNum = mxGetN(prhs[0]);
	fcnObj.varNum     = DIMNUM * fcnObj.prm.ndeNum * (fcnObj.prm.timeNum - 1);
	optObj.varNum     = fcnObj.varNum;

	plhs[0] = mxCreateDoubleMatrix(fcnObj.varNum, 1, mxREAL);
	double *optAlpVec = mxGetDoubles(plhs[0]);

	// ---

	fcnObj.dis.dfmNdeNum    = mxGetN(prhs[3]);
	fcnObj.dis.dfmElmNum    = mxGetN(prhs[2]);
	fcnObj.dis.dfmElmIfoNum = mxGetM(prhs[3]);
	fcnObj.dis.tgtElmNum    = mxGetN(prhs[4]);

	// ---

	long long fcnMemCnt = (long long) fcnObj.dis.dfmElmNum * (DIMNUM * 4 + 1)
	                                + fcnObj.prm.ndeNum * DIMNUM * (fcnObj.prm.timeNum + 5)
	                                + fcnObj.prm.ndeNum * fcnObj.prm.ndeNum * (fcnObj.prm.timeNum - 1);
	long long optMemCnt = (long long) optObj.varNum * (5 + optObj.memNum * 2) + optObj.memNum;

	double *fcnWorkspace = (double *) calloc(fcnMemCnt, sizeof(double));
	double *optWorkspace = (double *) calloc(optMemCnt, sizeof(double));

	long long fcnAsgMemCnt = assignFcnStructMemory(fcnObj, fcnWorkspace);
	long long optAsgMemCnt = assignOptStructMemory(optObj, optWorkspace);
	
	if ( fcnAsgMemCnt != fcnMemCnt )
	{
		mexErrMsgIdAndTxt("LDDMM:fcnMemAssign",
		                  "Assigned memory(%d) for function evaluation mismatched the allocated memory (%d).",
		                  fcnAsgMemCnt, fcnMemCnt);
	}

	if ( optAsgMemCnt != optMemCnt )
	{
		mexErrMsgIdAndTxt("LDDMM:optMemAssign",
		                  "Assigned memory(%d) for optimization mismatched the allocated memory (%d).",
		                  optAsgMemCnt, optMemCnt);
	}

	// ---

	double  fcnVal;
	double *grdVec = (double *) calloc(fcnObj.varNum, sizeof(double));

	memcpy(optAlpVec, iniAlpVec, sizeof(double) * fcnObj.varNum);
	LBFGS(&fcnVal, grdVec, optAlpVec, optObj, fcnObj);

	// ---

	free(fcnWorkspace);
	free(optWorkspace);

	free(grdVec);

	return;
}
