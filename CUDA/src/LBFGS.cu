// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 06/10/2020

#include <cstdio>
#include <cmath>
#include <cublas_v2.h>
#include "struct.h"
#include "constants.h"

void objgrd(double *, double *, double *, fcndata &);

int  lineSearch(double *, double *, double *, double *, double, double, double, double *, double *, double *,
                double &, int &, fcndata &);
void getDirection(double *, double, double *, double *, double *, 
                  int, int, int, double *, int, cublasHandle_t);

__global__ void vectorSubtractKernel(double *d_v12Vec, double *d_v1Vec, double *d_v2Vec, int varNum)
{
	int varIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( varIdx < varNum )
		d_v12Vec[varIdx] = d_v1Vec[varIdx] - d_v2Vec[varIdx];

	return;
}

void vectorSubtract(double *d_v12Vec, double *d_v1Vec, double *d_v2Vec, int varNum)
{
	int blkNum = (varNum - 1) / BLKDIM + 1;
	vectorSubtractKernel <<<blkNum, BLKDIM>>> (d_v12Vec, d_v1Vec, d_v2Vec, varNum);

	return;
}

int LBFGS(double *h_fcnNow, double *d_grdNow, double *d_posNow, optdata &optObj, fcndata &fcnObj)
{
	int     varNum = fcnObj.varNum;
	int     itrMax = optObj.itrMax;
	double  tolVal = optObj.tolVal;
	int     memNum = optObj.memNum;
	double  wolfe1 = optObj.wolfe1;
	double  wolfe2 = optObj.wolfe2;
	bool    vbsFlg = optObj.vbsFlg;

	double *d_dspMat = optObj.d_dspMat;
	double *d_dgdMat = optObj.d_dgdMat;
	double *d_dirVec = optObj.d_dirVec;
	double *d_posNxt = optObj.d_posNxt;
	double *d_grdNxt = optObj.d_grdNxt;
	double *d_dspVec = optObj.d_dspVec;
	double *d_dgdVec = optObj.d_dgdVec;
	double *h_recVec = optObj.h_recVec;
	double  h_fcnNxt;

	cudaMemset(d_dspMat, 0, sizeof(double) * varNum * memNum);
	cudaMemset(d_dgdMat, 0, sizeof(double) * varNum * memNum);

	objgrd(h_fcnNow, d_grdNow, d_posNow, fcnObj);

	double h_grdSqu;
	cublasDdot(fcnObj.blasHdl, varNum, d_grdNow, 1, d_grdNow, 1, &h_grdSqu);
	double h_grdLen = sqrt(h_grdSqu);

	if ( vbsFlg )
	{
		printf("%5s   %13s  %13s  %13s  %9s\n", "iter", "f", "|grad f|", "step length", "fcn eval");
		char sepStr[65] = {0};
		memset(sepStr, '-', 62);
		printf("%s\n", sepStr);
		printf("%5d:  %13.6e  %13.6e\n", 0, *h_fcnNow, h_grdLen);
	}

	int newIdx = -1;
	for ( int itrIdx = 1; itrIdx <= itrMax; ++itrIdx )
	{
		if ( h_grdLen < tolVal )
			break;

		double HIniVal;
		if ( newIdx == -1 )
			HIniVal = 1.0;
		else
		{
			double *d_dspPtr = d_dspMat + newIdx * varNum;
			double *d_dgdPtr = d_dgdMat + newIdx * varNum;

			double h_dspDgd, h_dgdDgd;
			cublasDdot(fcnObj.blasHdl, varNum, d_dspPtr, 1, d_dgdPtr, 1, &h_dspDgd);
			cublasDdot(fcnObj.blasHdl, varNum, d_dgdPtr, 1, d_dgdPtr, 1, &h_dgdDgd);
			
			HIniVal = h_dspDgd / h_dgdDgd;
		}

		if ( itrIdx <= memNum )
		{
			getDirection(d_dirVec, HIniVal, d_grdNow, d_dspMat, d_dgdMat, 
			             newIdx, itrIdx - 1, memNum, h_recVec, varNum, fcnObj.blasHdl);
		}
		else
		{
			getDirection(d_dirVec, HIniVal, d_grdNow, d_dspMat, d_dgdMat, 
			             newIdx, memNum, memNum, h_recVec, varNum, fcnObj.blasHdl);
		}

		double stpLen;
		int    objCnt;
		int    lineErr = lineSearch(d_posNow, d_grdNow, d_dirVec, h_fcnNow, wolfe1, wolfe2, tolVal, 
		                            d_posNxt, d_grdNxt, &h_fcnNxt, stpLen, objCnt, fcnObj);
		if ( lineErr != 0 ) return 1;

		vectorSubtract(d_dspVec, d_posNxt, d_posNow, varNum);
		vectorSubtract(d_dgdVec, d_grdNxt, d_grdNow, varNum);

		newIdx = (newIdx == memNum - 1) ? 0 : (newIdx + 1);
		cudaMemcpy(d_dspMat + newIdx * varNum, d_dspVec, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_dgdMat + newIdx * varNum, d_dgdVec, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);
	
		cudaMemcpy(d_posNow, d_posNxt, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_grdNow, d_grdNxt, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);
		*h_fcnNow = h_fcnNxt;
		cublasDdot(fcnObj.blasHdl, varNum, d_grdNow, 1, d_grdNow, 1, &h_grdSqu);
		h_grdLen = sqrt(h_grdSqu);

		if ( vbsFlg )
			printf("%5d:  %13.6e  %13.6e  %13.6e  %9d\n", itrIdx, *h_fcnNow, h_grdLen, stpLen, objCnt);
	}

	return 0;
}
