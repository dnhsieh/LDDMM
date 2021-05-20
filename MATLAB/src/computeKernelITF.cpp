// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cmath>
#include "mex.h"

void computeKernel(double *, double *, int, double, int, int);
void computeKernel(double *, double *, double *, int, double, int, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	if ( nrhs == 3 )
	{
		double *lmkMat;
		int     knlOrder;
		double  knlWidth;

		double *knlMat;

		lmkMat   = mxGetDoubles(prhs[0]);
		knlOrder =  mxGetScalar(prhs[1]);
		knlWidth =  mxGetScalar(prhs[2]);

		int dimNum = mxGetM(prhs[0]);
		int lmkNum = mxGetN(prhs[0]);

		plhs[0] = mxCreateDoubleMatrix(lmkNum, lmkNum, mxREAL);
		knlMat  = mxGetDoubles(plhs[0]);

		computeKernel(knlMat, lmkMat, knlOrder, knlWidth, lmkNum, dimNum);
	}
	else if ( nrhs == 4 )
	{
		double *lmkiMat;
		double *lmkjMat;
		int     knlOrder;
		double  knlWidth;

		double *knlMat;

		lmkiMat  = mxGetDoubles(prhs[0]);
		lmkjMat  = mxGetDoubles(prhs[1]);
		knlOrder =  mxGetScalar(prhs[2]);
		knlWidth =  mxGetScalar(prhs[3]);

		int  dimNum = mxGetM(prhs[0]);
		int lmkiNum = mxGetN(prhs[0]);
		int lmkjNum = mxGetN(prhs[1]);

		plhs[0] = mxCreateDoubleMatrix(lmkiNum, lmkjNum, mxREAL);
		knlMat  = mxGetDoubles(plhs[0]);

		computeKernel(knlMat, lmkiMat, lmkjMat, knlOrder, knlWidth, lmkiNum, lmkjNum, dimNum);
	}
	else
		mexErrMsgIdAndTxt("computeKernel:nrhs", "The number of inputs must be 3 or 4.");

	return;
}
