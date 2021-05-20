#include "constants.h"

__global__ void scaleVectorKernel(double *d_outVec, double scale, double *d_inpVec, int varNum)
{
	int varIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( varIdx < varNum )
		d_outVec[varIdx] = scale * d_inpVec[varIdx];

	return;
}

void scaleVector(double *d_outVec, double scale, double *d_inpVec, int varNum)
{
	int blkNum = (varNum - 1) / BLKDIM + 1;
	scaleVectorKernel <<<blkNum, BLKDIM>>> (d_outVec, scale, d_inpVec, varNum);

	return;
}
