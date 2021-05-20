// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cmath>
#include "besselk.h"
#include "polybesselk.h"
#include "matvec.h"
#include "constants.h"

inline void setBesselkCoefficients()
{
	cudaMemcpyToSymbol(c_P01Vec, P01Vec, sizeof(double) * (P01Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q01Vec, Q01Vec, sizeof(double) * (Q01Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P02Vec, P02Vec, sizeof(double) * (P02Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q02Vec, Q02Vec, sizeof(double) * (Q02Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P03Vec, P03Vec, sizeof(double) * (P03Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q03Vec, Q03Vec, sizeof(double) * (Q03Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P11Vec, P11Vec, sizeof(double) * (P11Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q11Vec, Q11Vec, sizeof(double) * (Q11Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P12Vec, P12Vec, sizeof(double) * (P12Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q12Vec, Q12Vec, sizeof(double) * (Q12Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P13Vec, P13Vec, sizeof(double) * (P13Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q13Vec, Q13Vec, sizeof(double) * (Q13Deg + 1), 0, cudaMemcpyHostToDevice);	

	return;
}

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

		double p1Val;
		p1Fcn(p1Val, dijVal);

		double knlVal = p1Val;

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

		double p1Val;
		p1Fcn(p1Val, dijVal);

		double knlVal = p1Val;

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

		double p0Val, p1Val;
		p0Fcn(p0Val, dijVal);
		p1Fcn(p1Val, dijVal);

		double knlVal = 0.5 * (p0Val + 2.0 * p1Val);

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

		double p0Val, p1Val;
		p0Fcn(p0Val, dijVal);
		p1Fcn(p1Val, dijVal);

		double knlVal = 0.5 * (p0Val + 2.0 * p1Val);

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
		double dijSqu = dijVal * dijVal;

		double p0Val, p1Val;
		p0Fcn(p0Val, dijVal);
		p1Fcn(p1Val, dijVal);

		double knlVal = (4.0 * p0Val + (8.0 + dijSqu) * p1Val) / 8.0;

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
		double dijSqu = dijVal * dijVal;

		double p0Val, p1Val;
		p0Fcn(p0Val, dijVal);
		p1Fcn(p1Val, dijVal);

		double knlVal = (4.0 * p0Val + (8.0 + dijSqu) * p1Val) / 8.0;

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
		double dijSqu = dijVal * dijVal;

		double p0Val, p1Val;
		p0Fcn(p0Val, dijVal);
		p1Fcn(p1Val, dijVal);

		double knlVal = ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val) / 48.0;

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
		double dijSqu = dijVal * dijVal;

		double p0Val, p1Val;
		p0Fcn(p0Val, dijVal);
		p1Fcn(p1Val, dijVal);

		double knlVal = ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val) / 48.0;

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
		double dijSqu = dijVal * dijVal;

		double p0Val, p1Val;
		p0Fcn(p0Val, dijVal);
		p1Fcn(p1Val, dijVal);

		double knlVal = (12.0 * (16.0 + dijSqu) * p0Val + (384.0 + dijSqu * (72.0 + dijSqu)) * p1Val) / 384.0;

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
		double dijSqu = dijVal * dijVal;

		double p0Val, p1Val;
		p0Fcn(p0Val, dijVal);
		p1Fcn(p1Val, dijVal);

		double knlVal = (12.0 * (16.0 + dijSqu) * p0Val + (384.0 + dijSqu * (72.0 + dijSqu)) * p1Val) / 384.0;

		d_knlMat[colIdx * lmkiNum + rowIdx] = knlVal;
	}

	return;
}

void computeKernel(double *d_knlMat, double *d_lmkMat, int knlOrder, double knlWidth, int lmkNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	setBesselkCoefficients();

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

	setBesselkCoefficients();

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

