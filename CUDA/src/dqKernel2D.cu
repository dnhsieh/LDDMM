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

__global__ void dqGaussian(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijSqu = eucnormSqu(qijVec) / knlWidthSqu;

			double dqKVal = -2.0 / knlWidthSqu * exp(-dijSqu);
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void dqiGaussian(double *d_dqiKMat, double *d_lmkiMat, double *d_lmkjMat,
                            double *d_lftMat, double *d_rgtMat,
                            double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijSqu = eucnormSqu(qijVec) / knlWidthSqu;

			double dqKVal = -2.0 / knlWidthSqu * exp(-dijSqu);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqiKMat, dqKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void dqjGaussian(double *d_dqjKMat, double *d_lmkiMat, double *d_lmkjMat,
                            double *d_lftMat, double *d_rgtMat,
                            double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qjVec, rjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijSqu = eucnormSqu(qjiVec) / knlWidthSqu;

			double dqKVal = -2.0 / knlWidthSqu * exp(-dijSqu);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
		}

		setVector(d_dqjKMat, dqKVec, lmkjIdx, lmkjNum);
	}

	return;
}

__global__ void dqMatern1(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat,
                          double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;

			double p1Val;
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (2.0 * knlWidthSqu) * p1Val;
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void dqiMatern1(double *d_dqiKMat, double *d_lmkiMat, double *d_lmkjMat,
                           double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;

			double p1Val;
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (2.0 * knlWidthSqu) * p1Val;
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqiKMat, dqKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void dqjMatern1(double *d_dqjKMat, double *d_lmkiMat, double *d_lmkjMat,
                           double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qjVec, rjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijVal = eucnorm(qjiVec) / knlWidth;

			double p1Val;
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (2.0 * knlWidthSqu) * p1Val;
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
		}

		setVector(d_dqjKMat, dqKVec, lmkjIdx, lmkjNum);
	}

	return;
}

__global__ void dqMatern2(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat,
                          double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (8.0 * knlWidthSqu) * (p0Val + 2.0 * p1Val);
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void dqiMatern2(double *d_dqiKMat, double *d_lmkiMat, double *d_lmkjMat,
                           double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (8.0 * knlWidthSqu) * (p0Val + 2.0 * p1Val);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqiKMat, dqKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void dqjMatern2(double *d_dqjKMat, double *d_lmkiMat, double *d_lmkjMat,
                           double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qjVec, rjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijVal = eucnorm(qjiVec) / knlWidth;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (8.0 * knlWidthSqu) * (p0Val + 2.0 * p1Val);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
		}

		setVector(d_dqjKMat, dqKVec, lmkjIdx, lmkjNum);
	}

	return;
}

__global__ void dqMatern3(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat,
                          double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (48.0 * knlWidthSqu) * (4.0 * p0Val + (8.0 + dijSqu) * p1Val);
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void dqiMatern3(double *d_dqiKMat, double *d_lmkiMat, double *d_lmkjMat,
                           double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (48.0 * knlWidthSqu) * (4.0 * p0Val + (8.0 + dijSqu) * p1Val);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqiKMat, dqKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void dqjMatern3(double *d_dqjKMat, double *d_lmkiMat, double *d_lmkjMat,
                           double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qjVec, rjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijVal = eucnorm(qjiVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (48.0 * knlWidthSqu) * (4.0 * p0Val + (8.0 + dijSqu) * p1Val);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
		}

		setVector(d_dqjKMat, dqKVec, lmkjIdx, lmkjNum);
	}

	return;
}

__global__ void dqMatern4(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat,
                          double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (384.0 * knlWidthSqu)
			               * ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void dqiMatern4(double *d_dqiKMat, double *d_lmkiMat, double *d_lmkjMat,
                           double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (384.0 * knlWidthSqu)
			               * ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
		}

		setVector(d_dqiKMat, dqKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void dqjMatern4(double *d_dqjKMat, double *d_lmkiMat, double *d_lmkjMat,
                           double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qjVec, rjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijVal = eucnorm(qjiVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double dqKVal = -1.0 / (384.0 * knlWidthSqu)
			               * ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
		}

		setVector(d_dqjKMat, dqKVec, lmkjIdx, lmkjNum);
	}

	return;
}

void dqKernel(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat,
              int knlOrder, double knlWidth, int lmkNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	setBesselkCoefficients();

	double knlWidthSqu = knlWidth * knlWidth;

	int blkNum = (lmkNum - 1) / BLKDIM + 1;

	switch ( knlOrder )
	{
		case -1:
			dqGaussian <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat, 
			                                 knlWidth, knlWidthSqu, lmkNum);
			break;

		// Matern0 is not differentiable
	
		case 1:
			dqMatern1 <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat,
			                                knlWidth, knlWidthSqu, lmkNum);
			break;
	
		case 2:
			dqMatern2 <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat,
			                                knlWidth, knlWidthSqu, lmkNum);
			break;
	
		case 3:
			dqMatern3 <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat,
			                                knlWidth, knlWidthSqu, lmkNum);
			break;
	
		case 4:
			dqMatern4 <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat,
			                                knlWidth, knlWidthSqu, lmkNum);
			break;
	}	

	return;
}

void dqKernel(double *d_dqiKMat, double *d_dqjKMat, double *d_lmkiMat, double *d_lmkjMat,
              double *d_lftMat, double *d_rgtMat, int knlOrder, double knlWidth, int lmkiNum, int lmkjNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	setBesselkCoefficients();

	double knlWidthSqu = knlWidth * knlWidth;

	int blkiNum = (lmkiNum - 1) / BLKDIM + 1;
	int blkjNum = (lmkjNum - 1) / BLKDIM + 1;

	switch ( knlOrder )
	{
		case -1:
			dqiGaussian <<<blkiNum, BLKDIM>>> (d_dqiKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat, 
			                                   knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			dqjGaussian <<<blkjNum, BLKDIM>>> (d_dqjKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat, 
			                                   knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;

		// Matern0 is not differentiable
	
		case 1:
			dqiMatern1 <<<blkiNum, BLKDIM>>> (d_dqiKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
			                                  knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			dqjMatern1 <<<blkjNum, BLKDIM>>> (d_dqjKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
			                                  knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;
	
		case 2:
			dqiMatern2 <<<blkiNum, BLKDIM>>> (d_dqiKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
			                                  knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			dqjMatern2 <<<blkjNum, BLKDIM>>> (d_dqjKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
			                                  knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;
	
		case 3:
			dqiMatern3 <<<blkiNum, BLKDIM>>> (d_dqiKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
			                                  knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			dqjMatern3 <<<blkjNum, BLKDIM>>> (d_dqjKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
			                                  knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;
	
		case 4:
			dqiMatern4 <<<blkiNum, BLKDIM>>> (d_dqiKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
			                                  knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			dqjMatern4 <<<blkjNum, BLKDIM>>> (d_dqjKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
			                                  knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;
	}	

	return;
}
