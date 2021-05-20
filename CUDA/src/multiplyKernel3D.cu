// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/18/2020

#include <cmath>
#include "matvec.h"
#include "constants.h"

__global__ void multiplyGaussian(double *d_vlcMat, double *d_lmkMat, double *d_alpMat,
                                 double knlWidth, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector qiVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijSqu = eucdistSqu(qiVec, qjVec) / (knlWidth * knlWidth);
			double knlVal = exp(-dijSqu);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijSqu = eucdistSqu(qiVec, qjVec) / (knlWidth * knlWidth);
			double knlVal = exp(-dijSqu);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = (1.0 + dijVal) * exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = (1.0 + dijVal) * exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = (3.0 + dijVal * (3.0 + dijVal)) / 3.0 * exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = (3.0 + dijVal * (3.0 + dijVal)) / 3.0 * exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / 15.0 * exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / 15.0 * exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ajVec, d_alpMat, lmkjIdx, lmkNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = (105.0 + dijVal * (105.0 + dijVal * (45.0 + dijVal * (10.0 + dijVal)))) / 105.0 * exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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
	
		vector viVec = {0.0, 0.0, 0.0};
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, ajVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(ajVec, d_alpMat,  lmkjIdx, lmkjNum);

			double dijVal = eucdist(qiVec, qjVec) / knlWidth;
			double knlVal = (105.0 + dijVal * (105.0 + dijVal * (45.0 + dijVal * (10.0 + dijVal)))) / 105.0 * exp(-dijVal);

			viVec.x += knlVal * ajVec.x;
			viVec.y += knlVal * ajVec.y;
			viVec.z += knlVal * ajVec.z;
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

	int blkNum = (lmkiNum - 1) / BLKDIM + 1;

	switch ( knlOrder )
	{
		case -1:
			multiplyGaussian <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat,
			                                       knlWidth, lmkiNum, lmkjNum);
			break;

		case 0:
			multiplyMatern0 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat,
			                                      knlWidth, lmkiNum, lmkjNum);
			break;

		case 1:
			multiplyMatern1 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat,
			                                      knlWidth, lmkiNum, lmkjNum);
			break;

		case 2:
			multiplyMatern2 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat,
			                                      knlWidth, lmkiNum, lmkjNum);
			break;

		case 3:
			multiplyMatern3 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat,
			                                      knlWidth, lmkiNum, lmkjNum);
			break;

		case 4:
			multiplyMatern4 <<<blkNum, BLKDIM>>> (d_vlcMat, d_lmkiMat, d_lmkjMat, d_alpMat,
			                                      knlWidth, lmkiNum, lmkjNum);
			break;
	}

	return;
}
