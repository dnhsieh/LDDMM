#include <cstring>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include "struct.h"
#include "constants.h"

long long assignFcnStructMemory(fcndata &, double *);
long long assignOptStructMemory(optdata &, double *);

int LBFGS(double *, double *, double *, optdata &, fcndata &);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mxInitGPU();

	fcndata fcnObj;
	optdata optObj;

	mxGPUArray const *iniNdeMat, *iniAlpVec;
	mxGPUArray const *dfmElmVtxMat, *dfmElmIfoMat;
	mxGPUArray const *tgtCenPosMat, *tgtElmVolVec, *tgtUniDirMat;

	iniNdeMat               = mxGPUCreateFromMxArray(prhs[ 0]);
	iniAlpVec               = mxGPUCreateFromMxArray(prhs[ 1]);
	dfmElmVtxMat            = mxGPUCreateFromMxArray(prhs[ 2]);
	dfmElmIfoMat            = mxGPUCreateFromMxArray(prhs[ 3]);
	tgtCenPosMat            = mxGPUCreateFromMxArray(prhs[ 4]);
	tgtElmVolVec            = mxGPUCreateFromMxArray(prhs[ 5]);
	tgtUniDirMat            = mxGPUCreateFromMxArray(prhs[ 6]);
	fcnObj.dis.cenKnlType   =            mxGetScalar(prhs[ 7]);
	fcnObj.dis.cenKnlWidth  =            mxGetScalar(prhs[ 8]); 
	fcnObj.dis.dirKnlType   =            mxGetScalar(prhs[ 9]);
	fcnObj.dis.dirKnlWidth  =            mxGetScalar(prhs[10]);
	fcnObj.dis.disWgt       =            mxGetScalar(prhs[11]);
	fcnObj.prm.knlOrder     =            mxGetScalar(prhs[12]);
	fcnObj.prm.knlWidth     =            mxGetScalar(prhs[13]);
	fcnObj.prm.timeStp      =            mxGetScalar(prhs[14]);
	fcnObj.prm.timeNum      =            mxGetScalar(prhs[15]);
	optObj.itrMax           =            mxGetScalar(prhs[16]);
	optObj.tolVal           =            mxGetScalar(prhs[17]);
	optObj.wolfe1           =            mxGetScalar(prhs[18]);
	optObj.wolfe2           =            mxGetScalar(prhs[19]);
	optObj.memNum           =            mxGetScalar(prhs[20]);
	optObj.vbsFlg           =            mxGetScalar(prhs[21]);

	// ---

	mwSize const *ndeDims = mxGPUGetDimensions(iniNdeMat);

	fcnObj.prm.ndeNum = ndeDims[0];
	fcnObj.varNum     = DIMNUM * fcnObj.prm.ndeNum * (fcnObj.prm.timeNum - 1);
	optObj.varNum     = fcnObj.varNum;

	mxGPUArray *optAlpVec;

	mwSize const ndim = 1;
	mwSize const outDims[1] = {(mwSize) fcnObj.varNum};
	optAlpVec = mxGPUCreateGPUArray(ndim, outDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

	double *d_optAlpVec = (double *) mxGPUGetData(optAlpVec);

	// ---

	fcnObj.prm.d_iniNdeMat    = (double *) mxGPUGetDataReadOnly(   iniNdeMat);
	double    *d_iniAlpVec    = (double *) mxGPUGetDataReadOnly(   iniAlpVec);
	fcnObj.dis.d_dfmElmVtxMat = (int    *) mxGPUGetDataReadOnly(dfmElmVtxMat);
	fcnObj.dis.d_dfmElmIfoMat = (int    *) mxGPUGetDataReadOnly(dfmElmIfoMat);
	fcnObj.dis.d_tgtCenPosMat = (double *) mxGPUGetDataReadOnly(tgtCenPosMat);
	fcnObj.dis.d_tgtElmVolVec = (double *) mxGPUGetDataReadOnly(tgtElmVolVec);
	fcnObj.dis.d_tgtUniDirMat = (double *) mxGPUGetDataReadOnly(tgtUniDirMat);

	mwSize const *dfmNdeDims = mxGPUGetDimensions(dfmElmIfoMat);
	mwSize const *dfmElmDims = mxGPUGetDimensions(dfmElmVtxMat);
	mwSize const *tgtElmDims = mxGPUGetDimensions(tgtCenPosMat);

	fcnObj.dis.dfmNdeNum = dfmNdeDims[0];
	fcnObj.dis.dfmElmNum = dfmElmDims[0];
	fcnObj.dis.tgtElmNum = tgtElmDims[0];

	// ---

	long long fcnMemCnt = (long long) fcnObj.dis.dfmElmNum + fcnObj.dis.tgtElmNum + SUMBLKDIM 
	                                + fcnObj.dis.dfmElmNum * (DIMNUM * 4 + 1)
	                                + fcnObj.prm.ndeNum * DIMNUM * (fcnObj.prm.timeNum + 5)
	                                + fcnObj.prm.ndeNum * fcnObj.prm.ndeNum;
	long long optMemCnt = (long long) optObj.varNum * (5 + optObj.memNum * 2);

	double *d_fcnWorkspace, *d_optWorkspace;
	cudaMalloc((void **) &d_fcnWorkspace, sizeof(double) * fcnMemCnt);
	cudaMalloc((void **) &d_optWorkspace, sizeof(double) * optMemCnt);

	optObj.h_recVec = (double *) calloc(optObj.memNum, sizeof(double));

	long long fcnAsgMemCnt = assignFcnStructMemory(fcnObj, d_fcnWorkspace);
	long long optAsgMemCnt = assignOptStructMemory(optObj, d_optWorkspace);
	
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

	cublasCreate(&(fcnObj.blasHdl));

	// ---

	double h_fcnVal, *d_grdVec;
	cudaMalloc((void **) &d_grdVec, sizeof(double) * fcnObj.varNum);

	cudaMemcpy(d_optAlpVec, d_iniAlpVec, sizeof(double) * fcnObj.varNum, cudaMemcpyDeviceToDevice);
	LBFGS(&h_fcnVal, d_grdVec, d_optAlpVec, optObj, fcnObj);

	plhs[0] = mxGPUCreateMxArrayOnGPU(optAlpVec);

	// ---

	mxGPUDestroyGPUArray(iniNdeMat);
	mxGPUDestroyGPUArray(iniAlpVec);
	mxGPUDestroyGPUArray(dfmElmVtxMat);
	mxGPUDestroyGPUArray(dfmElmIfoMat);
	mxGPUDestroyGPUArray(tgtCenPosMat);
	mxGPUDestroyGPUArray(tgtElmVolVec);
	mxGPUDestroyGPUArray(tgtUniDirMat);
	mxGPUDestroyGPUArray(optAlpVec);

	mxFree((void *)    ndeDims);
	mxFree((void *) dfmNdeDims);
	mxFree((void *) dfmElmDims);
	mxFree((void *) tgtElmDims);

	cudaFree(d_fcnWorkspace);
	cudaFree(d_optWorkspace);
	free(optObj.h_recVec);

	cublasDestroy(fcnObj.blasHdl);

	cudaFree(d_grdVec);

	return;
}
