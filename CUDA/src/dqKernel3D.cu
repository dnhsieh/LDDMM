// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/18/2020

#include <cmath>
#include "matvec.h"
#include "constants.h"

__global__ void dqGaussian(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat,
                           double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0, 0.0};

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
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			dqKVec.z += lrVal * dqKVal * qjiVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -exp(-dijVal) / knlWidthSqu;
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -exp(-dijVal) / knlWidthSqu;
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -exp(-dijVal) / knlWidthSqu;
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
			dqKVec.z += lrVal * dqKVal * qjiVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(1.0 + dijVal) / (3.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(1.0 + dijVal) / (3.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(1.0 + dijVal) / (3.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
			dqKVec.z += lrVal * dqKVal * qjiVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(3.0 + dijVal * (3.0 + dijVal)) / (15.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(3.0 + dijVal * (3.0 + dijVal)) / (15.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(3.0 + dijVal * (3.0 + dijVal)) / (15.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
			dqKVec.z += lrVal * dqKVal * qjiVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / (105.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / (105.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qijVec.x;
			dqKVec.y += lrVal * dqKVal * qijVec.y;
			dqKVec.z += lrVal * dqKVal * qijVec.z;
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
		vector dqKVec = {0.0, 0.0, 0.0};

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
			double dqKVal = -(15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / (105.0 * knlWidthSqu) * exp(-dijVal);
			double  lrVal = dotProduct(liVec, rjVec);

			dqKVec.x += lrVal * dqKVal * qjiVec.x;
			dqKVec.y += lrVal * dqKVal * qjiVec.y;
			dqKVec.z += lrVal * dqKVal * qjiVec.z;
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
