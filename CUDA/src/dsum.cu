// Date: 07/08/2020

#include <cstdio>
#include "constants.h"

__device__ double d_sumVal;

__global__ void dsumSmallGPU(double *d_vec, int len)
{
	__shared__ double s_vec[SUMBLKDIM];

	int trdIdx = threadIdx.x;

	s_vec[trdIdx] = (trdIdx < len ? d_vec[trdIdx] : 0.0);
	__syncthreads();

	for ( int stride = blockDim.x >> 1; stride > 0; stride >>= 1 )
	{
		if ( trdIdx < stride )
			s_vec[trdIdx] += s_vec[trdIdx + stride];
		__syncthreads();
	}

	if ( trdIdx == 0 )
		d_sumVal = s_vec[0];

	return;
}

__global__ void dsumGPU(double *d_buf, double *d_vec, int len)
{
	__shared__ double s_vec[SUMBLKDIM];

	int trdIdx = threadIdx.x;
	int glbIdx = blockIdx.x * blockDim.x + threadIdx.x;

	s_vec[trdIdx] = (glbIdx < len ? d_vec[glbIdx] : 0.0);
	__syncthreads();

	for ( int stride = blockDim.x >> 1; stride > 0; stride >>= 1 )
	{
		if ( trdIdx < stride )
			s_vec[trdIdx] += s_vec[trdIdx + stride];
		__syncthreads();
	}

	if ( trdIdx == 0 )
		d_buf[blockIdx.x] += s_vec[0];

	return;
}

void dsum(double *h_sumVal, double *d_vec, double *d_buf, int len)
{
	if ( len < SUMBLKDIM )
	{
		dsumSmallGPU <<<1, SUMBLKDIM>>> (d_vec, len);	
	}
	else
	{
		cudaMemset(d_buf, 0, sizeof(double) * SUMBLKDIM);

		for ( int runIdx = 0, runNum = (len - 1) / (SUMBLKDIM * SUMBLKDIM) + 1; runIdx < runNum; ++runIdx )
		{
			int lenNow = (runIdx == runNum - 1 ? len - runIdx * SUMBLKDIM * SUMBLKDIM : SUMBLKDIM * SUMBLKDIM);
			int blkNum = (lenNow - 1) / SUMBLKDIM + 1;
			dsumGPU <<<blkNum, SUMBLKDIM>>> (d_buf, d_vec + runIdx * SUMBLKDIM * SUMBLKDIM, lenNow);
		}
		dsumSmallGPU <<<1, SUMBLKDIM>>> (d_buf, SUMBLKDIM);
	}

	cudaMemcpyFromSymbol(h_sumVal, d_sumVal, sizeof(double), 0, cudaMemcpyDeviceToHost);
}

