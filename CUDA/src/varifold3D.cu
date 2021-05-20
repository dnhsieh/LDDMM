// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 07/12/2020

#include "matvec.h"
#include "constants.h"

void dsum(double *, double *, double *, int);

__global__ void landmarksToVarifoldKernel(double *d_cenPosMat, double *d_uniDirMat, double *d_elmVolVec,
                                          double *d_lmkPosMat, int *d_elmVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		int q0Idx = d_elmVtxMat[             elmIdx];
		int q1Idx = d_elmVtxMat[    elmNum + elmIdx];
		int q2Idx = d_elmVtxMat[2 * elmNum + elmIdx];

		vector q0Vec, q1Vec, q2Vec;
		getVector(q0Vec, d_lmkPosMat, q0Idx, lmkNum);
		getVector(q1Vec, d_lmkPosMat, q1Idx, lmkNum);
		getVector(q2Vec, d_lmkPosMat, q2Idx, lmkNum);

		vector cenVec;
		vectorAverage(cenVec, q0Vec, q1Vec, q2Vec);

		vector q10Vec, q20Vec, dirVec;
		vectorSubtract(q10Vec, q1Vec, q0Vec);
		vectorSubtract(q20Vec, q2Vec, q0Vec);
		crossProduct(dirVec, q10Vec, q20Vec);

		double elmVol = eucnorm(dirVec);
		dirVec.x /= elmVol;
		dirVec.y /= elmVol;
		dirVec.z /= elmVol;
		
		setVector(d_cenPosMat, cenVec, elmIdx, elmNum);
		setVector(d_uniDirMat, dirVec, elmIdx, elmNum);
		d_elmVolVec[elmIdx] = 0.5 * elmVol;
	}

	return;
}

__device__ void geometricFunction(double &knlVal, vector c1Vec, vector c2Vec,
                                  char knlType, double knlWidth)
{
	if ( knlType == 'G' )   // gaussian
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = exp(-dstSqu / (knlWidth * knlWidth));

		return;
	}

	if ( knlType == 'C' )   // cauchy
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = 1.0 / (1.0 + dstSqu / (knlWidth * knlWidth));

		return;
	}

	return;
}

__device__ void geometricFunction(double &knlVal, vector &d1KVec, vector c1Vec, vector c2Vec, 
                                  char knlType, double knlWidth)
{
	if ( knlType == 'G' )   // gaussian
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = exp(-dstSqu / (knlWidth * knlWidth));

		double d1KVal = -2.0 * knlVal / (knlWidth * knlWidth);
		d1KVec.x = d1KVal * (c1Vec.x - c2Vec.x);
		d1KVec.y = d1KVal * (c1Vec.y - c2Vec.y);
		d1KVec.z = d1KVal * (c1Vec.z - c2Vec.z);

		return;
	}

	if ( knlType == 'C' )   // cauchy
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = 1.0 / (1.0 + dstSqu / (knlWidth * knlWidth));

		double d1KVal = -2.0 * knlVal * knlVal / (knlWidth * knlWidth);
		d1KVec.x = d1KVal * (c1Vec.x - c2Vec.x);
		d1KVec.y = d1KVal * (c1Vec.y - c2Vec.y);
		d1KVec.z = d1KVal * (c1Vec.z - c2Vec.z);

		return;
	}

	return;
}

__device__ void grassmanFunction(double &knlVal, vector v1Vec, vector v2Vec,
                                 char knlType, double knlWidth)
{
	if ( knlType == 'B' )   // binet
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal * angVal;

		return;
	}

	if ( knlType == 'L' )   // linear
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal;

		return;
	}

	if ( knlType == 'O' )   // gaussian oriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal - 1.0) / (knlWidth * knlWidth));

		return;
	}

	if ( knlType == 'U' )   // gaussian unoriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal * angVal - 1.0) / (knlWidth * knlWidth));

		return;
	}

	return;
}

__device__ void grassmanFunction(double &knlVal, vector &d1KVec, vector v1Vec, vector v2Vec,
                                 char knlType, double knlWidth, double n1Len)
{
	if ( knlType == 'B' )   // binet
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal * angVal;

		double d1KVal = 2.0 * angVal;
		d1KVec.x = d1KVal / n1Len * (-angVal * v1Vec.x + v2Vec.x);
		d1KVec.y = d1KVal / n1Len * (-angVal * v1Vec.y + v2Vec.y);
		d1KVec.z = d1KVal / n1Len * (-angVal * v1Vec.z + v2Vec.z);

		return;
	}

	if ( knlType == 'L' )   // linear
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal;

		d1KVec.x = 1.0 / n1Len * (-angVal * v1Vec.x + v2Vec.x);
		d1KVec.y = 1.0 / n1Len * (-angVal * v1Vec.y + v2Vec.y);
		d1KVec.z = 1.0 / n1Len * (-angVal * v1Vec.z + v2Vec.z);

		return;
	}

	if ( knlType == 'O' )   // gaussian oriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal - 1.0) / (knlWidth * knlWidth));

		double d1KVal = 2.0 * knlVal / (knlWidth * knlWidth);
		d1KVec.x = d1KVal / n1Len * (-angVal * v1Vec.x + v2Vec.x);
		d1KVec.y = d1KVal / n1Len * (-angVal * v1Vec.y + v2Vec.y);
		d1KVec.z = d1KVal / n1Len * (-angVal * v1Vec.z + v2Vec.z);

		return;
	}

	if ( knlType == 'U' )   // gaussian unoriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal * angVal - 1.0) / (knlWidth * knlWidth));

		double d1KVal = 4.0 * angVal * knlVal / (knlWidth * knlWidth);
		d1KVec.x = d1KVal / n1Len * (-angVal * v1Vec.x + v2Vec.x);
		d1KVec.y = d1KVal / n1Len * (-angVal * v1Vec.y + v2Vec.y);
		d1KVec.z = d1KVal / n1Len * (-angVal * v1Vec.z + v2Vec.z);

		return;
	}

	return;
}

__global__ void vfd_DD_DT_Kernel(double *d_vfdVec,
                                 double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec, 
                                 double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec, 
                                 char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth, 
                                 int dfmElmNum, int tgtElmNum)
{
	int dfmElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( dfmElmiIdx < dfmElmNum )
	{
		double vfdVal = 0.0;

		vector dfmCeniVec, dfmDiriVec;
		getVector(dfmCeniVec, d_dfmCenPosMat, dfmElmiIdx, dfmElmNum);
		getVector(dfmDiriVec, d_dfmUniDirMat, dfmElmiIdx, dfmElmNum);

		double dfmElmiVol = d_dfmElmVolVec[dfmElmiIdx];

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			vector dfmCenjVec, dfmDirjVec;
			getVector(dfmCenjVec, d_dfmCenPosMat, dfmElmjIdx, dfmElmNum);
			getVector(dfmDirjVec, d_dfmUniDirMat, dfmElmjIdx, dfmElmNum);

			double dfmElmjVol = d_dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth);
			 grassmanFunction(dirKnlVal, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;
		}

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, dfmCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, dfmDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			vfdVal -= 2.0 * cenKnlVal * dirKnlVal * dfmElmiVol * tgtElmjVol;
		}

		d_vfdVec[dfmElmiIdx] = vfdVal;
	}

	return;
}

__global__ void vfd_TT_Kernel(double *d_vfdVec, double *d_tgtCenPosMat, double *d_tgtUniDirMat, 
                              double *d_tgtElmVolVec, char cenKnlType, double cenKnlWidth,
                              char dirKnlType, double dirKnlWidth, int tgtElmNum)
{
	int tgtElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tgtElmiIdx < tgtElmNum )
	{
		double vfdVal = 0.0;
	
		vector tgtCeniVec, tgtDiriVec;
		getVector(tgtCeniVec, d_tgtCenPosMat, tgtElmiIdx, tgtElmNum);
		getVector(tgtDiriVec, d_tgtUniDirMat, tgtElmiIdx, tgtElmNum);

		double tgtElmiVol = d_tgtElmVolVec[tgtElmiIdx];

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, tgtDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * tgtElmiVol * tgtElmjVol;
		}

		d_vfdVec[tgtElmiIdx] = vfdVal;
	}

	return;
}

__global__ void vfd_TT_TD_Kernel(double *d_vfdVec,
                                 double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec, 
                                 double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec, 
                                 char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth, 
                                 int dfmElmNum, int tgtElmNum)
{
	int tgtElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tgtElmiIdx < tgtElmNum )
	{
		double vfdVal = 0.0;

		vector tgtCeniVec, tgtDiriVec;
		getVector(tgtCeniVec, d_tgtCenPosMat, tgtElmiIdx, tgtElmNum);
		getVector(tgtDiriVec, d_tgtUniDirMat, tgtElmiIdx, tgtElmNum);

		double tgtElmiVol = d_tgtElmVolVec[tgtElmiIdx];

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, tgtDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * tgtElmiVol * tgtElmjVol;
		}

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			vector dfmCenjVec, dfmDirjVec;
			getVector(dfmCenjVec, d_dfmCenPosMat, dfmElmjIdx, dfmElmNum);
			getVector(dfmDirjVec, d_dfmUniDirMat, dfmElmjIdx, dfmElmNum);

			double dfmElmjVol = d_dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth);
			 grassmanFunction(dirKnlVal, tgtDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth);

			vfdVal -= 2.0 * cenKnlVal * dirKnlVal * tgtElmiVol * dfmElmjVol;
		}

		d_vfdVec[tgtElmiIdx] = vfdVal;
	}

	return;
}

__global__ void vfd_DD_Kernel(double *d_vfdVec, double *d_dfmCenPosMat, double *d_dfmUniDirMat,
                              double *d_dfmElmVolVec, char cenKnlType, double cenKnlWidth, 
                              char dirKnlType, double dirKnlWidth, int dfmElmNum)
{
	int dfmElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( dfmElmiIdx < dfmElmNum )
	{
		double vfdVal = 0.0;

		vector dfmCeniVec, dfmDiriVec;
		getVector(dfmCeniVec, d_dfmCenPosMat, dfmElmiIdx, dfmElmNum);
		getVector(dfmDiriVec, d_dfmUniDirMat, dfmElmiIdx, dfmElmNum);

		double dfmElmiVol = d_dfmElmVolVec[dfmElmiIdx];

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			vector dfmCenjVec, dfmDirjVec;
			getVector(dfmCenjVec, d_dfmCenPosMat, dfmElmjIdx, dfmElmNum);
			getVector(dfmDirjVec, d_dfmUniDirMat, dfmElmjIdx, dfmElmNum);

			double dfmElmjVol = d_dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth);
			 grassmanFunction(dirKnlVal, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;
		}

		d_vfdVec[dfmElmiIdx] = vfdVal;
	}

	return;
}

__global__ void dqVfd_DD_DT_Kernel(double *d_vfdVec, double *d_dcVfdMat, double *d_ddVfdMat,
                                   double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec, 
                                   double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec, 
                                   char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth, 
                                   int dfmElmNum, int tgtElmNum)
{
	int dfmElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( dfmElmiIdx < dfmElmNum )
	{
		double vfdVal = 0.0;
		vector dciVfdVec = {0.0, 0.0, 0.0};
		vector ddiVfdVec = {0.0, 0.0, 0.0};

		vector dfmCeniVec, dfmDiriVec;
		getVector(dfmCeniVec, d_dfmCenPosMat, dfmElmiIdx, dfmElmNum);
		getVector(dfmDiriVec, d_dfmUniDirMat, dfmElmiIdx, dfmElmNum);

		double dfmElmiVol = d_dfmElmVolVec[dfmElmiIdx];

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			vector dfmCenjVec, dfmDirjVec;
			getVector(dfmCenjVec, d_dfmCenPosMat, dfmElmjIdx, dfmElmNum);
			getVector(dfmDirjVec, d_dfmUniDirMat, dfmElmjIdx, dfmElmNum);

			double dfmElmjVol = d_dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			vector dciKnlVec, ddiKnlVec;
			geometricFunction(cenKnlVal, dciKnlVec, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth                  );
			 grassmanFunction(dirKnlVal, ddiKnlVec, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth, 2.0 * dfmElmiVol);

			vfdVal += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;

			dciVfdVec.x += 2.0 * dciKnlVec.x * dirKnlVal * dfmElmiVol * dfmElmjVol;
			dciVfdVec.y += 2.0 * dciKnlVec.y * dirKnlVal * dfmElmiVol * dfmElmjVol;
			dciVfdVec.z += 2.0 * dciKnlVec.z * dirKnlVal * dfmElmiVol * dfmElmjVol;

			ddiVfdVec.x += 2.0 * cenKnlVal * (  ddiKnlVec.x *       dfmElmiVol
			                                  + dirKnlVal   * 0.5 * dfmDiriVec.x ) * dfmElmjVol;

			ddiVfdVec.y += 2.0 * cenKnlVal * (  ddiKnlVec.y *       dfmElmiVol
			                                  + dirKnlVal   * 0.5 * dfmDiriVec.y ) * dfmElmjVol;

			ddiVfdVec.z += 2.0 * cenKnlVal * (  ddiKnlVec.z *       dfmElmiVol
			                                  + dirKnlVal   * 0.5 * dfmDiriVec.z ) * dfmElmjVol;
		}

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			vector dciKnlVec, ddiKnlVec;
			geometricFunction(cenKnlVal, dciKnlVec, dfmCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth                  );
			 grassmanFunction(dirKnlVal, ddiKnlVec, dfmDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth, 2.0 * dfmElmiVol);

			vfdVal -= 2.0 * cenKnlVal * dirKnlVal * dfmElmiVol * tgtElmjVol;

			dciVfdVec.x -= 2.0 * dciKnlVec.x * dirKnlVal * dfmElmiVol * tgtElmjVol;
			dciVfdVec.y -= 2.0 * dciKnlVec.y * dirKnlVal * dfmElmiVol * tgtElmjVol;
			dciVfdVec.z -= 2.0 * dciKnlVec.z * dirKnlVal * dfmElmiVol * tgtElmjVol;

			ddiVfdVec.x -= 2.0 * cenKnlVal * (  ddiKnlVec.x *       dfmElmiVol 
			                                  + dirKnlVal   * 0.5 * dfmDiriVec.x ) * tgtElmjVol;

			ddiVfdVec.y -= 2.0 * cenKnlVal * (  ddiKnlVec.y *       dfmElmiVol
			                                  + dirKnlVal   * 0.5 * dfmDiriVec.y ) * tgtElmjVol;

			ddiVfdVec.z -= 2.0 * cenKnlVal * (  ddiKnlVec.z *       dfmElmiVol
			                                  + dirKnlVal   * 0.5 * dfmDiriVec.z ) * tgtElmjVol;
		}

		d_vfdVec[dfmElmiIdx] = vfdVal;
		setVector(d_dcVfdMat, dciVfdVec, dfmElmiIdx, dfmElmNum);
		setVector(d_ddVfdMat, ddiVfdVec, dfmElmiIdx, dfmElmNum);
	}

	return;
}

__global__ void dqVfd_TT_Kernel(double *d_vfdVec, double *d_tgtCenPosMat, double *d_tgtUniDirMat, 
                                double *d_tgtElmVolVec, char cenKnlType, double cenKnlWidth,
                                char dirKnlType, double dirKnlWidth, int tgtElmNum)
{
	int tgtElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tgtElmiIdx < tgtElmNum )
	{
		double vfdVal = 0.0;
	
		vector tgtCeniVec, tgtDiriVec;
		getVector(tgtCeniVec, d_tgtCenPosMat, tgtElmiIdx, tgtElmNum);
		getVector(tgtDiriVec, d_tgtUniDirMat, tgtElmiIdx, tgtElmNum);

		double tgtElmiVol = d_tgtElmVolVec[tgtElmiIdx];

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, tgtDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * tgtElmiVol * tgtElmjVol;
		}

		d_vfdVec[tgtElmiIdx] = vfdVal;
	}

	return;
}

__global__ void dqVfdGatherKernel(double *d_dqVfdMat, double *d_dcVfdMat, double *d_ddVfdMat,
                                  int *d_dfmElmIfoMat, double *d_dfmLmkPosMat,
                                  int dfmElmNum, int dfmLmkNum)
{
	int dfmLmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( dfmLmkIdx < dfmLmkNum )
	{
		vector dqVfdVec = {0.0, 0.0, 0.0};

		int adjNum = d_dfmElmIfoMat[dfmLmkIdx];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int elmIdx = d_dfmElmIfoMat[(1 + 3 * adjIdx    ) * dfmLmkNum + dfmLmkIdx];
			int  qaIdx = d_dfmElmIfoMat[(1 + 3 * adjIdx + 1) * dfmLmkNum + dfmLmkIdx];
			int  qbIdx = d_dfmElmIfoMat[(1 + 3 * adjIdx + 2) * dfmLmkNum + dfmLmkIdx];

			vector dcVfdVec, ddVfdVec;
			getVector(dcVfdVec, d_dcVfdMat, elmIdx, dfmElmNum);
			getVector(ddVfdVec, d_ddVfdMat, elmIdx, dfmElmNum);

			vector qaDfmVec, qbDfmVec;
			getVector(qaDfmVec, d_dfmLmkPosMat, qaIdx, dfmLmkNum);
			getVector(qbDfmVec, d_dfmLmkPosMat, qbIdx, dfmLmkNum);

			vector qabDfmVec, ddVfdxqabVec;
			vectorSubtract(qabDfmVec, qaDfmVec, qbDfmVec);
			crossProduct(ddVfdxqabVec, ddVfdVec, qabDfmVec);

			dqVfdVec.x += dcVfdVec.x / 3.0 + ddVfdxqabVec.x;
			dqVfdVec.y += dcVfdVec.y / 3.0 + ddVfdxqabVec.y;
			dqVfdVec.z += dcVfdVec.z / 3.0 + ddVfdxqabVec.z;
		}

		setVector(d_dqVfdMat, dqVfdVec, dfmLmkIdx, dfmLmkNum);
	}

	return;
}

void varifold(double *h_vfdPtr, double *d_dfmLmkPosMat, int *d_dfmElmVtxMat,
              double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec,
              char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth,
              double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec,
              double *d_vfdVec, double *d_sumBufVec,
              int dfmLmkNum, int dfmElmNum, int tgtElmNum)
{
	int blkNum = (dfmElmNum - 1) / BLKDIM + 1;
	landmarksToVarifoldKernel <<<blkNum, BLKDIM>>> (d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec, 
	                                                d_dfmLmkPosMat, d_dfmElmVtxMat, dfmLmkNum, dfmElmNum);
	
	if ( dfmElmNum >= tgtElmNum )
	{
		blkNum = (dfmElmNum - 1) / BLKDIM + 1;
		vfd_DD_DT_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec,
		                                       d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
		                                       d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
		                                       cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
		                                       dfmElmNum, tgtElmNum);

		blkNum = (tgtElmNum - 1) / BLKDIM + 1;
		vfd_TT_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec + dfmElmNum, 
		                                    d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
		                                    cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth, tgtElmNum);
	}
	else
	{
		blkNum = (tgtElmNum - 1) / BLKDIM + 1;
		vfd_TT_TD_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec,
		                                       d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
		                                       d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
		                                       cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
		                                       dfmElmNum, tgtElmNum);

		blkNum = (dfmElmNum - 1) / BLKDIM + 1;
		vfd_DD_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec + tgtElmNum,
		                                    d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
		                                    cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth, dfmElmNum);
	}

	dsum(h_vfdPtr, d_vfdVec, d_sumBufVec, dfmElmNum + tgtElmNum);

	return;
}

void varifold(double *h_vfdPtr, double *d_dqVfdMat,
              double *d_dfmLmkPosMat, int *d_dfmElmVtxMat, int *d_dfmElmIfoMat,
              double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec,
              char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth,
              double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec,
              double *d_vfdVec, double *d_sumBufVec, double *d_dcVfdMat, double *d_ddVfdMat,
              int dfmLmkNum, int dfmElmNum, int tgtElmNum)
{
	int blkNum = (dfmElmNum - 1) / BLKDIM + 1;
	landmarksToVarifoldKernel <<<blkNum, BLKDIM>>> (d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec, 
	                                                d_dfmLmkPosMat, d_dfmElmVtxMat, dfmLmkNum, dfmElmNum);

	blkNum = (dfmElmNum - 1) / BLKDIM + 1;
	dqVfd_DD_DT_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec, d_dcVfdMat, d_ddVfdMat,
	                                         d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
	                                         d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
	                                         cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
	                                         dfmElmNum, tgtElmNum);

	blkNum = (tgtElmNum - 1) / BLKDIM + 1;
	dqVfd_TT_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec + dfmElmNum,
	                                      d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
		                                   cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth, tgtElmNum);

	dsum(h_vfdPtr, d_vfdVec, d_sumBufVec, dfmElmNum + tgtElmNum);

	blkNum = (dfmLmkNum - 1) / BLKDIM + 1;
	dqVfdGatherKernel <<<blkNum, BLKDIM>>> (d_dqVfdMat, d_dcVfdMat, d_ddVfdMat,
	                                        d_dfmElmIfoMat, d_dfmLmkPosMat, dfmElmNum, dfmLmkNum);

	return;
}
