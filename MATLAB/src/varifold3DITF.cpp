// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 12/05/2020

#include "mex.h"
#include "constants.h"

void varifold(double *, double *, int *, double *, double *, double *, char, double, char, double,
              double *, double *, double *, int, int, int);

void varifold(double *, double *, double *, int *, int *, double *, double *, double *,
              char, double, char, double, double *,
              double *, double *, double *, double *, int, int, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	if ( nlhs == 1 )
	{
		double *dfmLmkPosMat =      mxGetDoubles(prhs[0]);
		int    *dfmElmVtxMat = (int *) mxGetData(prhs[1]);
		double *tgtCenPosMat =      mxGetDoubles(prhs[2]);
		double *tgtUniDirMat =      mxGetDoubles(prhs[3]);
		double *tgtElmVolVec =      mxGetDoubles(prhs[4]);
		char    cenKnlType   =       mxGetScalar(prhs[5]);
		double  cenKnlWidth  =       mxGetScalar(prhs[6]);
		char    dirKnlType   =       mxGetScalar(prhs[7]);
		double  dirKnlWidth  =       mxGetScalar(prhs[8]);
	
		// ---
	
		int dfmLmkNum = mxGetN(prhs[0]);
		int dfmElmNum = mxGetN(prhs[1]);
		int tgtElmNum = mxGetN(prhs[2]);
	
		double *dfmCenPosMat = (double *) mxMalloc(sizeof(double) * DIMNUM * dfmElmNum);
		double *dfmUniDirMat = (double *) mxMalloc(sizeof(double) * DIMNUM * dfmElmNum);
		double *dfmElmVolVec = (double *) mxMalloc(sizeof(double)          * dfmElmNum);

		double vfdVal;
		varifold(&vfdVal,
		         dfmLmkPosMat, dfmElmVtxMat,
		         tgtCenPosMat, tgtUniDirMat, tgtElmVolVec,
		         cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
		         dfmCenPosMat, dfmUniDirMat, dfmElmVolVec,
		         dfmLmkNum, dfmElmNum, tgtElmNum);

		plhs[0] = mxCreateDoubleScalar(vfdVal);
	
		// ---
	
		mxFree(dfmCenPosMat);
		mxFree(dfmUniDirMat);
		mxFree(dfmElmVolVec);
	}
	else
	{
		double *dfmLmkPosMat =      mxGetDoubles(prhs[0]);
		int    *dfmElmVtxMat = (int *) mxGetData(prhs[1]);
		int    *dfmElmIfoMat = (int *) mxGetData(prhs[2]);
		double *tgtCenPosMat =      mxGetDoubles(prhs[3]);
		double *tgtUniDirMat =      mxGetDoubles(prhs[4]);
		double *tgtElmVolVec =      mxGetDoubles(prhs[5]);
		char    cenKnlType   =       mxGetScalar(prhs[6]);
		double  cenKnlWidth  =       mxGetScalar(prhs[7]);
		char    dirKnlType   =       mxGetScalar(prhs[8]);
		double  dirKnlWidth  =       mxGetScalar(prhs[9]);
	
		int dfmLmkNum = mxGetN(prhs[0]);

		plhs[1] = mxCreateDoubleMatrix(DIMNUM, dfmLmkNum, mxREAL);
		double *dqVfdMat = mxGetDoubles(plhs[1]);
	
		// ---
	
		int dfmElmNum    = mxGetN(prhs[1]);
		int dfmElmIfoNum = mxGetM(prhs[2]);
		int tgtElmNum    = mxGetN(prhs[3]);
		
		double *dfmCenPosMat = (double *) mxMalloc(sizeof(double) * DIMNUM * dfmElmNum);
		double *dfmUniDirMat = (double *) mxMalloc(sizeof(double) * DIMNUM * dfmElmNum);
		double *dfmElmVolVec = (double *) mxMalloc(sizeof(double)          * dfmElmNum);
		double *dcVfdMat     = (double *) mxMalloc(sizeof(double) * DIMNUM * dfmElmNum);
		double *ddVfdMat     = (double *) mxMalloc(sizeof(double) * DIMNUM * dfmElmNum);
		
		double vfdVal;
		varifold(&vfdVal, dqVfdMat,
		         dfmLmkPosMat, dfmElmVtxMat, dfmElmIfoMat,
		         tgtCenPosMat, tgtUniDirMat, tgtElmVolVec,
		         cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
		         dfmCenPosMat, dfmUniDirMat, dfmElmVolVec,
		         dcVfdMat, ddVfdMat,
		         dfmLmkNum, dfmElmNum, dfmElmIfoNum, tgtElmNum);
	
		plhs[0] = mxCreateDoubleScalar(vfdVal);
	
		// ---
	
		mxFree(dfmCenPosMat);
		mxFree(dfmUniDirMat);
		mxFree(dfmElmVolVec);
		mxFree(dcVfdMat);
		mxFree(ddVfdMat);
	}

	return;
}

