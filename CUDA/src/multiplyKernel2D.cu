// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cmath>
#include "besselk.h"
#include "polybesselk.h"
#include "matvec.h"
#include "constants.h"

void setBesselkCoefficients()
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

__global__ void multiplyGaussian(double *d_vlcMat, double *d_lmkMat, double *d_alpMat,
                                 double knlWidth, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijSqu = eucdistSqu(qiVec, qjVec) / (knlWidth * knlWidth);
			double knlVal = exp(-dijSqu);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void multiplyGaussian(double *d_vlcMat, double *d_lmkiMat, double *d_lmkjMat, double *d_alpMat,
                                 double knlWidth, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijSqu = eucdistSqu(qiVec, qjVec) / (knlWidth * knlWidth);
			double knlVal = exp(-dijSqu);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void multiplyMatern0(double *d_vlcMat, double *d_lmkMat, double *d_alpMat,
                                double knlWidth, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;

			double f1Val;
			p1Fcn(f1Val, dijVal);

			double knlVal = f1Val;

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void multiplyMatern0(double *d_vlcMat, double *d_lmkiMat, double *d_lmkjMat, double *d_alpMat,
                                double knlWidth, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;

			double f1Val;
			p1Fcn(f1Val, dijVal);

			double knlVal = f1Val;

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void multiplyMatern1(double *d_vlcMat, double *d_lmkMat, double *d_alpMat,
                                double knlWidth, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;

			double f0Val, f1Val;
			p0Fcn(f0Val, dijVal);
			p1Fcn(f1Val, dijVal);

			double knlVal = 0.5 * (f0Val + 2.0 * f1Val);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void multiplyMatern1(double *d_vlcMat, double *d_lmkiMat, double *d_lmkjMat, double *d_alpMat,
                                double knlWidth, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;

			double f0Val, f1Val;
			p0Fcn(f0Val, dijVal);
			p1Fcn(f1Val, dijVal);

			double knlVal = 0.5 * (f0Val + 2.0 * f1Val);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void multiplyMatern2(double *d_vlcMat, double *d_lmkMat, double *d_alpMat,
                                double knlWidth, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double f0Val, f1Val;
			p0Fcn(f0Val, dijVal);
			p1Fcn(f1Val, dijVal);

			double knlVal = (4.0 * f0Val + (8.0 + dijSqu) * f1Val) / 8.0;

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void multiplyMatern2(double *d_vlcMat, double *d_lmkiMat, double *d_lmkjMat, double *d_alpMat,
                                double knlWidth, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double f0Val, f1Val;
			p0Fcn(f0Val, dijVal);
			p1Fcn(f1Val, dijVal);

			double knlVal = (4.0 * f0Val + (8.0 + dijSqu) * f1Val) / 8.0;

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void multiplyMatern3(double *d_vlcMat, double *d_lmkMat, double *d_alpMat,
                                double knlWidth, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double f0Val, f1Val;
			p0Fcn(f0Val, dijVal);
			p1Fcn(f1Val, dijVal);

			double knlVal = ((24.0 + dijSqu) * f0Val + 8.0 * (6.0 + dijSqu) * f1Val) / 48.0;

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void multiplyMatern3(double *d_vlcMat, double *d_lmkiMat, double *d_lmkjMat, double *d_alpMat,
                                double knlWidth, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double f0Val, f1Val;
			p0Fcn(f0Val, dijVal);
			p1Fcn(f1Val, dijVal);

			double knlVal = ((24.0 + dijSqu) * f0Val + 8.0 * (6.0 + dijSqu) * f1Val) / 48.0;

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void multiplyMatern4(double *d_vlcMat, double *d_lmkMat, double *d_alpMat,
                                double knlWidth, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double f0Val, f1Val;
			p0Fcn(f0Val, dijVal);
			p1Fcn(f1Val, dijVal);

			double knlVal = (12.0 * (16.0 + dijSqu) * f0Val + (384.0 + dijSqu * (72.0 + dijSqu)) * f1Val) / 384.0;

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void multiplyMatern4(double *d_vlcMat, double *d_lmkiMat, double *d_lmkjMat, double *d_alpMat,
                                double knlWidth, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
	
		vector viVec = {0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double f0Val, f1Val;
			p0Fcn(f0Val, dijVal);
			p1Fcn(f1Val, dijVal);

			double knlVal = (12.0 * (16.0 + dijSqu) * f0Val + (384.0 + dijSqu * (72.0 + dijSqu)) * f1Val) / 384.0;

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
		}

		setVector(d_vlcMat, viVec, lmkiIdx, lmkiNum);
	}

	return;
}

void multiplyKernel(double *d_vlcMat, double *d_lmkMat, double *d_alpMat, 
                    int knlOrder, double knlWidth, int lmkNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	setBesselkCoefficients();

	int blkNum = (lmkNum - 1) / BLKDIM + 1;

	switch ( knlOrder )
	{
		case -1:
			multiplyGaussian <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkMat, d_alpMat, knlWidth, lmkNum);
			break;

		case 0:
			multiplyMatern0 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkMat, d_alpMat, knlWidth, lmkNum);
			break;

		case 1:
			multiplyMatern1 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkMat, d_alpMat, knlWidth, lmkNum);
			break;

		case 2:
			multiplyMatern2 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkMat, d_alpMat, knlWidth, lmkNum);
			break;

		case 3:
			multiplyMatern3 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkMat, d_alpMat, knlWidth, lmkNum);
			break;

		case 4:
			multiplyMatern4 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkMat, d_alpMat, knlWidth, lmkNum);
			break;
	}

	return;
}

void multiplyKernel(double *d_vlcMat, double *d_lmkiMat, double *d_lmkjMat, double *d_alpMat, 
                    int knlOrder, double knlWidth, int lmkiNum, int lmkjNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	setBesselkCoefficients();

	int blkNum = (lmkiNum - 1) / BLKDIM + 1;

	switch ( knlOrder )
	{
		case -1:
			multiplyGaussian <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 0:
			multiplyMatern0 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 1:
			multiplyMatern1 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 2:
			multiplyMatern2 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 3:
			multiplyMatern3 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat, knlWidth, lmkiNum, lmkjNum);
			break;

		case 4:
			multiplyMatern4 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat, knlWidth, lmkiNum, lmkjNum);
			break;
	}

	return;
}
