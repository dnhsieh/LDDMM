// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/18/2020

#include <cmath>
#include "matvec.h"
#include "constants.h"

__global__ void gaussian(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);

		double dijSqu = eucdistSqu(qiVec, qjVec) / (knlWidth * knlWidth);
		double knlVal = exp(-dijSqu);

		d_knlMat[colIdx * lmkNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void gaussian(double *d_knlMat, double *d_lmkiMat, double *d_lmkjMat,
                         double knlWidth, int lmkiNum, int lmkjNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkiNum && colIdx < lmkjNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkiMat, rowIdx, lmkiNum);
		getVector(qjVec, d_lmkjMat, colIdx, lmkjNum);

		double dijSqu = eucdistSqu(qiVec, qjVec) / (knlWidth * knlWidth);
		double knlVal = exp(-dijSqu);

		d_knlMat[colIdx * lmkiNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern0(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = exp(-dijVal);

		d_knlMat[colIdx * lmkNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern0(double *d_knlMat, double *d_lmkiMat, double *d_lmkjMat,
                        double knlWidth, int lmkiNum, int lmkjNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkiNum && colIdx < lmkjNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkiMat, rowIdx, lmkiNum);
		getVector(qjVec, d_lmkjMat, colIdx, lmkjNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = exp(-dijVal);

		d_knlMat[colIdx * lmkiNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern1(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = (1.0 + dijVal) * exp(-dijVal);

		d_knlMat[colIdx * lmkNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern1(double *d_knlMat, double *d_lmkiMat, double *d_lmkjMat,
                        double knlWidth, int lmkiNum, int lmkjNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkiNum && colIdx < lmkjNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkiMat, rowIdx, lmkiNum);
		getVector(qjVec, d_lmkjMat, colIdx, lmkjNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = (1.0 + dijVal) * exp(-dijVal);

		d_knlMat[colIdx * lmkiNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern2(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = (3.0 + dijVal * (3.0 + dijVal)) / 3.0 * exp(-dijVal);

		d_knlMat[colIdx * lmkNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern2(double *d_knlMat, double *d_lmkiMat, double *d_lmkjMat,
                        double knlWidth, int lmkiNum, int lmkjNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkiNum && colIdx < lmkjNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkiMat, rowIdx, lmkiNum);
		getVector(qjVec, d_lmkjMat, colIdx, lmkjNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = (3.0 + dijVal * (3.0 + dijVal)) / 3.0 * exp(-dijVal);

		d_knlMat[colIdx * lmkiNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern3(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / 15.0 * exp(-dijVal);

		d_knlMat[colIdx * lmkNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern3(double *d_knlMat, double *d_lmkiMat, double *d_lmkjMat,
                        double knlWidth, int lmkiNum, int lmkjNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkiNum && colIdx < lmkjNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkiMat, rowIdx, lmkiNum);
		getVector(qjVec, d_lmkjMat, colIdx, lmkjNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / 15.0 * exp(-dijVal);

		d_knlMat[colIdx * lmkiNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern4(double *d_knlMat, double *d_lmkMat, double knlWidth, int lmkNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkNum && colIdx < lmkNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkMat, rowIdx, lmkNum);
		getVector(qjVec, d_lmkMat, colIdx, lmkNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = (105.0 + dijVal * (105.0 + dijVal * (45.0 + dijVal * (10.0 + dijVal)))) / 105.0 * exp(-dijVal);

		d_knlMat[colIdx * lmkNum + rowIdx] = knlVal;
	}

	return;
}

__global__ void matern4(double *d_knlMat, double *d_lmkiMat, double *d_lmkjMat,
                        double knlWidth, int lmkiNum, int lmkjNum)
{
	int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if ( rowIdx < lmkiNum && colIdx < lmkjNum )
	{
		vector qiVec, qjVec;
		getVector(qiVec, d_lmkiMat, rowIdx, lmkiNum);
		getVector(qjVec, d_lmkjMat, colIdx, lmkjNum);

		double dijVal = eucdist(qiVec, qjVec) / knlWidth;
		double knlVal = (105.0 + dijVal * (105.0 + dijVal * (45.0 + dijVal * (10.0 + dijVal)))) / 105.0 * exp(-dijVal);

		d_knlMat[colIdx * lmkiNum + rowIdx] = knlVal;
	}

	return;
}

void computeKernel(double *d_knlMat, double *d_lmkMat, int knlOrder, double knlWidth, int lmkNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	int  gridRow = (lmkNum - 1) / BLKROW + 1;
	dim3 blkNum(gridRow, gridRow);
	dim3 blkDim( BLKROW,  BLKROW);

	switch ( knlOrder )
	{
		case -1:
			gaussian <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 0:
			matern0 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 1:
			matern1 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 2:
			matern2 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 3:
			matern3 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;

		case 4:
			matern4 <<<blkNum, blkDim>>> (d_knlMat, d_lmkMat, knlWidth, lmkNum);
			break;
	}

	return;
}

void computeKernel(double *d_knlMat, double *d_lmkiMat, double *d_lmkjMat, 
                   int knlOrder, double knlWidth, int lmkiNum, int lmkjNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	int  gridRow = (lmkiNum - 1) / BLKROW + 1;
	int  gridCol = (lmkjNum - 1) / BLKROW + 1;
	dim3 blkNum(gridRow, gridCol);
	dim3 blkDim( BLKROW,  BLKROW);

	switch ( knlOrder )
	{
		case -1:
			gaussian <<<blkNum, blkDim>>> (d_knlMat, d_lmkiMat, d_lmkjMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 0:
			matern0 <<<blkNum, blkDim>>> (d_knlMat, d_lmkiMat, d_lmkjMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 1:
			matern1 <<<blkNum, blkDim>>> (d_knlMat, d_lmkiMat, d_lmkjMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 2:
			matern2 <<<blkNum, blkDim>>> (d_knlMat, d_lmkiMat, d_lmkjMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 3:
			matern3 <<<blkNum, blkDim>>> (d_knlMat, d_lmkiMat, d_lmkjMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 4:
			matern4 <<<blkNum, blkDim>>> (d_knlMat, d_lmkiMat, d_lmkjMat, knlWidth, lmkiNum, lmkjNum);
			break;
	}

	return;
}

