// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cstring>
#include <cmath>
#include "mex.h"

void multiplyKernel(double *, double *, double *, int, double, int, int);
void multiplyKernel(double *, double *, double *, double *, int, double, int, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	if ( nrhs == 4 )
	{
		double *lmkMat, *alpMat;
		int     knlOrder;
		double  knlWidth;
	
		double *vlcMat;
	
		lmkMat   = mxGetDoubles(prhs[0]);
		alpMat   = mxGetDoubles(prhs[1]);
		knlOrder =  mxGetScalar(prhs[2]);
		knlWidth =  mxGetScalar(prhs[3]);
	
		int dimNum = mxGetM(prhs[0]);
		int lmkNum = mxGetN(prhs[0]);
	
		plhs[0] = mxCreateDoubleMatrix(dimNum, lmkNum, mxREAL);
		vlcMat  = mxGetDoubles(plhs[0]);
	
		multiplyKernel(vlcMat, lmkMat, alpMat, knlOrder, knlWidth, lmkNum, dimNum);
	}
	else if ( nrhs == 5 )
	{
		double *lmkiMat, *lmkjMat, *alpMat;
		int     knlOrder;
		double  knlWidth;
	
		double *vlcMat;
	
		lmkiMat  = mxGetDoubles(prhs[0]);
		lmkjMat  = mxGetDoubles(prhs[1]);
		alpMat   = mxGetDoubles(prhs[2]);
		knlOrder =  mxGetScalar(prhs[3]);
		knlWidth =  mxGetScalar(prhs[4]);
	
		int  dimNum = mxGetM(prhs[0]);
		int lmkiNum = mxGetN(prhs[0]);
		int lmkjNum = mxGetN(prhs[1]);
	
		plhs[0] = mxCreateDoubleMatrix(dimNum, lmkiNum, mxREAL);
		vlcMat  = mxGetDoubles(plhs[0]);
	
		multiplyKernel(vlcMat, lmkiMat, lmkjMat, alpMat, knlOrder, knlWidth, lmkiNum, lmkjNum, dimNum);
	}
	else
		mexErrMsgIdAndTxt("multiplyKernel:nrhs", "The number of inputs must be 4 or 5.");

	return;
}
