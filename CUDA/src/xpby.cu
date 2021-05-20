#include "constants.h"

__global__ void xpbyKernel(double *d_outVec, double *d_xVec, double bVal, double *d_yVec, int varNum)
{
	int varIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( varIdx < varNum )
		d_outVec[varIdx] = d_xVec[varIdx] + bVal * d_yVec[varIdx];

	return;
}

void xpby(double *d_outVec, double *d_xVec, double bVal, double *d_yVec, int varNum)
{
	int blkNum = (varNum - 1) / BLKDIM + 1;
	xpbyKernel <<<blkNum, BLKDIM>>> (d_outVec, d_xVec, bVal, d_yVec, varNum);

	return;
}
