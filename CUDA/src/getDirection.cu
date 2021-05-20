// Algorithm 7.4 in Nocedal
//
// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 05/20/2020

#include <cublas_v2.h>
#include "constants.h"

__global__ void vectorxpay(double *d_xVec, double aVal, double *d_yVec, int varNum)
{
	int varIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( varIdx < varNum )
		d_xVec[varIdx] += aVal * d_yVec[varIdx];

	return;
}

__global__ void vectorScale(double *d_xVec, double scale, int varNum)
{
	int varIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( varIdx < varNum )
		d_xVec[varIdx] *= scale;

	return;
}

void getDirection(double *d_dirVec, double HIniVal, double *d_grdNow, double *d_sMat, double *d_yMat, 
                  int newIdx, int hisNum, int memNum, double *h_alpVec, int varNum, cublasHandle_t blasHdl)
{
	// s   = x_next - x_now = dspVec
	// y   = (grad f)_next - (grad f)_now = dgdVec
	// rho = 1 / (s^T y)

	int blkNum = (varNum - 1) / BLKDIM + 1;
	
	cudaMemcpy(d_dirVec, d_grdNow, sizeof(double) * varNum, cudaMemcpyDeviceToDevice);

	for ( int hisCnt = 0; hisCnt < hisNum; ++hisCnt )
	{
		int hisIdx = newIdx - hisCnt;	
		if ( hisIdx < 0 ) hisIdx += memNum;

		double *d_sVec = d_sMat + hisIdx * varNum;
		double *d_yVec = d_yMat + hisIdx * varNum;

		double h_sTq, h_sTy;
		cublasDdot(blasHdl, varNum, d_sVec, 1, d_dirVec, 1, &h_sTq);
		cublasDdot(blasHdl, varNum, d_sVec, 1, d_yVec,   1, &h_sTy);
		double h_alpVal = h_sTq / h_sTy;

		vectorxpay <<<blkNum, BLKDIM>>> (d_dirVec, -h_alpVal, d_yVec, varNum);

		h_alpVec[hisIdx] = h_alpVal;
	}

	vectorScale <<<blkNum, BLKDIM>>> (d_dirVec, HIniVal, varNum);

	int oldIdx = (hisNum < memNum ? 0 : newIdx + 1);
	if ( oldIdx == memNum ) oldIdx = 0;

	for ( int hisCnt = 0; hisCnt < hisNum; ++hisCnt )
	{
		int hisIdx = oldIdx + hisCnt;
		if ( hisIdx >= memNum ) hisIdx -= memNum;

		double h_alpVal = h_alpVec[hisIdx];

		double *d_sVec = d_sMat + hisIdx * varNum;
		double *d_yVec = d_yMat + hisIdx * varNum;

		double h_yTr, h_sTy;
		cublasDdot(blasHdl, varNum, d_yVec, 1, d_dirVec, 1, &h_yTr);
		cublasDdot(blasHdl, varNum, d_sVec, 1, d_yVec,   1, &h_sTy);
		double h_btaVal = h_yTr / h_sTy;

		vectorxpay <<<blkNum, BLKDIM>>> (d_dirVec, h_alpVal - h_btaVal, d_sVec, varNum);
	}

	vectorScale <<<blkNum, BLKDIM>>> (d_dirVec, -1.0, varNum);

	return;
}
