// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "constants.h"

void multiplyKernel(double *, double *, double *, int, double, int);
void multiplyKernel(double *, double *, double *, double *, int, double, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	mxInitGPU();

	if ( nrhs == 4 )
	{
		mxGPUArray const *lmkMat, *alpMat;
		mxGPUArray       *vlcMat;
	
		int    knlOrder;
		double knlWidth;
	
		lmkMat   = mxGPUCreateFromMxArray(prhs[0]);
		alpMat   = mxGPUCreateFromMxArray(prhs[1]);
		knlOrder =            mxGetScalar(prhs[2]);
		knlWidth =            mxGetScalar(prhs[3]);
	
		mwSize const *lmkDims = mxGPUGetDimensions(lmkMat);
		int lmkNum = lmkDims[0];
	
		mwSize const ndim = 2;
		mwSize const vlcDims[2] = {(mwSize) lmkNum, (mwSize) DIMNUM};
		vlcMat = mxGPUCreateGPUArray(ndim, vlcDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
		// ---
	
		double *d_lmkMat = (double *) mxGPUGetDataReadOnly(lmkMat);
		double *d_alpMat = (double *) mxGPUGetDataReadOnly(alpMat);
		double *d_vlcMat = (double *) mxGPUGetData(vlcMat);
	
		// ---
	
		multiplyKernel(d_vlcMat, d_lmkMat, d_alpMat, knlOrder, knlWidth, lmkNum);
		plhs[0] = mxGPUCreateMxArrayOnGPU(vlcMat);
	
		// ---
	
		mxGPUDestroyGPUArray(lmkMat);
		mxGPUDestroyGPUArray(alpMat);
		mxGPUDestroyGPUArray(vlcMat);
	
		mxFree((void *) lmkDims);
	}
	else if ( nrhs == 5 )
	{
		mxGPUArray const *lmkiMat, *lmkjMat, *alpMat;
		mxGPUArray       *vlcMat;
	
		int    knlOrder;
		double knlWidth;
	
		lmkiMat  = mxGPUCreateFromMxArray(prhs[0]);
		lmkjMat  = mxGPUCreateFromMxArray(prhs[1]);
		alpMat   = mxGPUCreateFromMxArray(prhs[2]);
		knlOrder =            mxGetScalar(prhs[3]);
		knlWidth =            mxGetScalar(prhs[4]);
	
		mwSize const *lmkiDims = mxGPUGetDimensions(lmkiMat);
		mwSize const *lmkjDims = mxGPUGetDimensions(lmkjMat);
		int lmkiNum = lmkiDims[0];
		int lmkjNum = lmkjDims[0];
	
		mwSize const ndim = 2;
		mwSize const vlcDims[2] = {(mwSize) lmkiNum, (mwSize) DIMNUM};
		vlcMat = mxGPUCreateGPUArray(ndim, vlcDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
		// ---
	
		double *d_lmkiMat = (double *) mxGPUGetDataReadOnly(lmkiMat);
		double *d_lmkjMat = (double *) mxGPUGetDataReadOnly(lmkjMat);
		double *d_alpMat  = (double *) mxGPUGetDataReadOnly(alpMat);
		double *d_vlcMat  = (double *) mxGPUGetData(vlcMat);
	
		// ---
	
		multiplyKernel(d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat, knlOrder, knlWidth, lmkiNum, lmkjNum);
		plhs[0] = mxGPUCreateMxArrayOnGPU(vlcMat);
	
		// ---
	
		mxGPUDestroyGPUArray(lmkiMat);
		mxGPUDestroyGPUArray(lmkjMat);
		mxGPUDestroyGPUArray(alpMat);
		mxGPUDestroyGPUArray(vlcMat);
	
		mxFree((void *) lmkiDims);
		mxFree((void *) lmkjDims);
	}
	else
		mexErrMsgIdAndTxt("multiplyKernel:nrhs", "The number of inputs must be 4 or 5.");

	return;
}

