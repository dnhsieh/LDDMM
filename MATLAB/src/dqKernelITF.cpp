// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cstring>
#include <cmath>
#include "mex.h"

void dqKernel(double *, double *, double *, double *, int, double, int, int);
void dqKernel(double *, double *, double *, double *, double *, double *, int, double, int, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	if ( nrhs == 5 )
	{
		double *lmkMat, *lftMat, *rgtMat;
		int     knlOrder;
		double  knlWidth;
	
		double *dqKMat;
	
		lmkMat   = mxGetDoubles(prhs[0]);
		lftMat   = mxGetDoubles(prhs[1]);
		rgtMat   = mxGetDoubles(prhs[2]);
		knlOrder =  mxGetScalar(prhs[3]);
		knlWidth =  mxGetScalar(prhs[4]);
	
		if ( knlOrder == 0 )
			mexErrMsgIdAndTxt("dqKernel:order", "Matern kernel of order 0 is not differentiable.");
	
		int dimNum = mxGetM(prhs[0]);
		int lmkNum = mxGetN(prhs[0]);
	
		plhs[0] = mxCreateDoubleMatrix(dimNum, lmkNum, mxREAL);
		dqKMat  = mxGetDoubles(plhs[0]);
	
		dqKernel(dqKMat, lmkMat, lftMat, rgtMat, knlOrder, knlWidth, lmkNum, dimNum);
	}
	else if ( nrhs == 6 )
	{
		double *lmkiMat, *lmkjMat, *lftMat, *rgtMat;
		int     knlOrder;
		double  knlWidth;
	
		double *dqiKMat, *dqjKMat;
	
		lmkiMat  = mxGetDoubles(prhs[0]);
		lmkjMat  = mxGetDoubles(prhs[1]);
		lftMat   = mxGetDoubles(prhs[2]);
		rgtMat   = mxGetDoubles(prhs[3]);
		knlOrder =  mxGetScalar(prhs[4]);
		knlWidth =  mxGetScalar(prhs[5]);
	
		if ( knlOrder == 0 )
			mexErrMsgIdAndTxt("dqKernel:order", "Matern kernel of order 0 is not differentiable.");
	
		int  dimNum = mxGetM(prhs[0]);
		int lmkiNum = mxGetN(prhs[0]);
		int lmkjNum = mxGetN(prhs[1]);
	
		plhs[0] = mxCreateDoubleMatrix(dimNum, lmkiNum, mxREAL);
		plhs[1] = mxCreateDoubleMatrix(dimNum, lmkjNum, mxREAL);
		dqiKMat = mxGetDoubles(plhs[0]);
		dqjKMat = mxGetDoubles(plhs[1]);
	
		dqKernel(dqiKMat, dqjKMat, lmkiMat, lmkjMat, lftMat, rgtMat,
		         knlOrder, knlWidth, lmkiNum, lmkjNum, dimNum);
	}
	else
		mexErrMsgIdAndTxt("dqKernel:nrhs", "The number of inputs must be 5 or 6.");

	return;
}
