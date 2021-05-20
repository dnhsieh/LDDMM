// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 12/05/2020

#include <cstring>
#include "matvec.h"
#include "constants.h"

void landmarksToVarifoldKernel(double *cenPosMat, double *uniDirMat, double *elmVolVec,
                               double *lmkPosMat, int *elmVtxMat, int elmNum)
{
	for ( int elmIdx = 0; elmIdx < elmNum; ++elmIdx )
	{
		int q0Idx = elmVtxMat[elmIdx * VFDVTXNUM    ];
		int q1Idx = elmVtxMat[elmIdx * VFDVTXNUM + 1];

		double *q0Vec = lmkPosMat + q0Idx * DIMNUM;
		double *q1Vec = lmkPosMat + q1Idx * DIMNUM;

		double *cenVec = cenPosMat + elmIdx * DIMNUM;
		double *dirVec = uniDirMat + elmIdx * DIMNUM;
		 vectorAverage(cenVec, q0Vec, q1Vec);
		vectorSubtract(dirVec, q1Vec, q0Vec);

		double elmVol = eucnorm(dirVec);
		dirVec[0] /= elmVol;
		dirVec[1] /= elmVol;
		
		elmVolVec[elmIdx] = elmVol;
	}

	return;
}

void geometricFunction(double &knlVal, double *c1Vec, double *c2Vec,
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

void geometricFunction(double &knlVal, double *d1KVec, double *c1Vec, double *c2Vec, 
                       char knlType, double knlWidth)
{
	if ( knlType == 'G' )   // gaussian
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = exp(-dstSqu / (knlWidth * knlWidth));

		double d1KVal = -2.0 * knlVal / (knlWidth * knlWidth);
		d1KVec[0] = d1KVal * (c1Vec[0] - c2Vec[0]);
		d1KVec[1] = d1KVal * (c1Vec[1] - c2Vec[1]);

		return;
	}

	if ( knlType == 'C' )   // cauchy
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = 1.0 / (1.0 + dstSqu / (knlWidth * knlWidth));

		double d1KVal = -2.0 * knlVal * knlVal / (knlWidth * knlWidth);
		d1KVec[0] = d1KVal * (c1Vec[0] - c2Vec[0]);
		d1KVec[1] = d1KVal * (c1Vec[1] - c2Vec[1]);

		return;
	}

	return;
}

void grassmanFunction(double &knlVal, double *v1Vec, double *v2Vec,
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

void grassmanFunction(double &knlVal, double *d1KVec, double *v1Vec, double *v2Vec,
                      char knlType, double knlWidth, double v1Vol)
{
	if ( knlType == 'B' )   // binet
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal * angVal;

		double d1KVal = 2.0 * angVal;
		d1KVec[0] = d1KVal / v1Vol * (-angVal * v1Vec[0] + v2Vec[0]);
		d1KVec[1] = d1KVal / v1Vol * (-angVal * v1Vec[1] + v2Vec[1]);

		return;
	}

	if ( knlType == 'L' )   // linear
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal;

		d1KVec[0] = 1.0 / v1Vol * (-angVal * v1Vec[0] + v2Vec[0]);
		d1KVec[1] = 1.0 / v1Vol * (-angVal * v1Vec[1] + v2Vec[1]);

		return;
	}

	if ( knlType == 'O' )   // gaussian oriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal - 1.0) / (knlWidth * knlWidth));

		double d1KVal = 2.0 * knlVal / (knlWidth * knlWidth);
		d1KVec[0] = d1KVal / v1Vol * (-angVal * v1Vec[0] + v2Vec[0]);
		d1KVec[1] = d1KVal / v1Vol * (-angVal * v1Vec[1] + v2Vec[1]);

		return;
	}

	if ( knlType == 'U' )   // gaussian unoriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal * angVal - 1.0) / (knlWidth * knlWidth));

		double d1KVal = 4.0 * angVal * knlVal / (knlWidth * knlWidth);
		d1KVec[0] = d1KVal / v1Vol * (-angVal * v1Vec[0] + v2Vec[0]);
		d1KVec[1] = d1KVal / v1Vol * (-angVal * v1Vec[1] + v2Vec[1]);

		return;
	}

	return;
}

void vf_Kernel(double *vfdPtr, double *dfmCenPosMat, double *dfmUniDirMat, double *dfmElmVolVec, 
               double *tgtCenPosMat, double *tgtUniDirMat, double *tgtElmVolVec, 
               char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth, 
               int dfmElmNum, int tgtElmNum)
{
	*vfdPtr = 0.0;

	for ( int dfmElmiIdx = 0; dfmElmiIdx < dfmElmNum; ++dfmElmiIdx )
	{
		double *dfmCeniVec = dfmCenPosMat + dfmElmiIdx * DIMNUM;
		double *dfmDiriVec = dfmUniDirMat + dfmElmiIdx * DIMNUM;

		double dfmElmiVol = dfmElmVolVec[dfmElmiIdx];

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			double *dfmCenjVec = dfmCenPosMat + dfmElmjIdx * DIMNUM;
			double *dfmDirjVec = dfmUniDirMat + dfmElmjIdx * DIMNUM;

			double dfmElmjVol = dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth);
			 grassmanFunction(dirKnlVal, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth);

			*vfdPtr += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;
		}

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			double *tgtCenjVec = tgtCenPosMat + tgtElmjIdx * DIMNUM;
			double *tgtDirjVec = tgtUniDirMat + tgtElmjIdx * DIMNUM;

			double tgtElmjVol = tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, dfmCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, dfmDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			*vfdPtr -= 2.0 * cenKnlVal * dirKnlVal * dfmElmiVol * tgtElmjVol;
		}
	}

	for ( int tgtElmiIdx = 0; tgtElmiIdx < tgtElmNum; ++tgtElmiIdx )
	{
		double *tgtCeniVec = tgtCenPosMat + tgtElmiIdx * DIMNUM;
		double *tgtDiriVec = tgtUniDirMat + tgtElmiIdx * DIMNUM;

		double tgtElmiVol = tgtElmVolVec[tgtElmiIdx];

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			double *tgtCenjVec = tgtCenPosMat + tgtElmjIdx * DIMNUM;
			double *tgtDirjVec = tgtUniDirMat + tgtElmjIdx * DIMNUM;

			double tgtElmjVol = tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, tgtDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			*vfdPtr += cenKnlVal * dirKnlVal * tgtElmiVol * tgtElmjVol;
		}
	}

	return;
}

void dqVf_Kernel(double *vfdPtr, double *dcVfdMat, double *ddVfdMat,
                 double *dfmCenPosMat, double *dfmUniDirMat, double *dfmElmVolVec, 
                 double *tgtCenPosMat, double *tgtUniDirMat, double *tgtElmVolVec, 
                 char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth, 
                 int dfmElmNum, int tgtElmNum)
{
	*vfdPtr = 0.0;

	for ( int dfmElmiIdx = 0; dfmElmiIdx < dfmElmNum; ++dfmElmiIdx )
	{
		double *dciVfdVec = dcVfdMat + dfmElmiIdx * DIMNUM;
		double *ddiVfdVec = ddVfdMat + dfmElmiIdx * DIMNUM;

		double *dfmCeniVec = dfmCenPosMat + dfmElmiIdx * DIMNUM;
		double *dfmDiriVec = dfmUniDirMat + dfmElmiIdx * DIMNUM;

		double dfmElmiVol = dfmElmVolVec[dfmElmiIdx];

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			double *dfmCenjVec = dfmCenPosMat + dfmElmjIdx * DIMNUM;
			double *dfmDirjVec = dfmUniDirMat + dfmElmjIdx * DIMNUM;

			double dfmElmjVol = dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			double dciKnlVec[DIMNUM], ddiKnlVec[DIMNUM];
			geometricFunction(cenKnlVal, dciKnlVec, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth            );
			 grassmanFunction(dirKnlVal, ddiKnlVec, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth, dfmElmiVol);

			*vfdPtr += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;

			dciVfdVec[0] += 2.0 * dciKnlVec[0] * dirKnlVal * dfmElmiVol * dfmElmjVol;
			dciVfdVec[1] += 2.0 * dciKnlVec[1] * dirKnlVal * dfmElmiVol * dfmElmjVol;

			ddiVfdVec[0] += 2.0 * cenKnlVal * (  ddiKnlVec[0] * dfmElmiVol
			                                   + dirKnlVal   * dfmDiriVec[0] ) * dfmElmjVol;

			ddiVfdVec[1] += 2.0 * cenKnlVal * (  ddiKnlVec[1] * dfmElmiVol
			                                   + dirKnlVal   * dfmDiriVec[1] ) * dfmElmjVol;
		}

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			double *tgtCenjVec = tgtCenPosMat + tgtElmjIdx * DIMNUM;
			double *tgtDirjVec = tgtUniDirMat + tgtElmjIdx * DIMNUM;

			double tgtElmjVol = tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			double dciKnlVec[DIMNUM], ddiKnlVec[DIMNUM];
			geometricFunction(cenKnlVal, dciKnlVec, dfmCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth            );			
			 grassmanFunction(dirKnlVal, ddiKnlVec, dfmDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth, dfmElmiVol);

			*vfdPtr -= 2.0 * cenKnlVal * dirKnlVal * dfmElmiVol * tgtElmjVol;

			dciVfdVec[0] -= 2.0 * dciKnlVec[0] * dirKnlVal * dfmElmiVol * tgtElmjVol;
			dciVfdVec[1] -= 2.0 * dciKnlVec[1] * dirKnlVal * dfmElmiVol * tgtElmjVol;

			ddiVfdVec[0] -= 2.0 * cenKnlVal * (  ddiKnlVec[0] * dfmElmiVol 
			                                   + dirKnlVal   * dfmDiriVec[0] ) * tgtElmjVol;

			ddiVfdVec[1] -= 2.0 * cenKnlVal * (  ddiKnlVec[1] * dfmElmiVol
			                                   + dirKnlVal   * dfmDiriVec[1] ) * tgtElmjVol;
		}
	}

	for ( int tgtElmiIdx = 0; tgtElmiIdx < tgtElmNum; ++tgtElmiIdx )
	{
		double *tgtCeniVec = tgtCenPosMat + tgtElmiIdx * DIMNUM;
		double *tgtDiriVec = tgtUniDirMat + tgtElmiIdx * DIMNUM;

		double tgtElmiVol = tgtElmVolVec[tgtElmiIdx];

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			double *tgtCenjVec = tgtCenPosMat + tgtElmjIdx * DIMNUM;
			double *tgtDirjVec = tgtUniDirMat + tgtElmjIdx * DIMNUM;

			double tgtElmjVol = tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);	
			 grassmanFunction(dirKnlVal, tgtDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			*vfdPtr += cenKnlVal * dirKnlVal * tgtElmiVol * tgtElmjVol;
		}
	}

	return;
}

void dqVfdGatherKernel(double *dqVfdMat, double *dcVfdMat, double *ddVfdMat,
                       int *dfmElmIfoMat, int dfmElmIfoNum, int dfmLmkNum)
{
	for ( int dfmLmkIdx = 0; dfmLmkIdx < dfmLmkNum; ++dfmLmkIdx )
	{
		double *dqVfdVec = dqVfdMat + dfmLmkIdx * DIMNUM;

		int adjNum = dfmElmIfoMat[dfmLmkIdx * dfmElmIfoNum];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int elmIdx = dfmElmIfoMat[dfmLmkIdx * dfmElmIfoNum + 1 + 2 * adjIdx    ];
			int sgnInt = dfmElmIfoMat[dfmLmkIdx * dfmElmIfoNum + 1 + 2 * adjIdx + 1];

			double *dcVfdVec = dcVfdMat + elmIdx * DIMNUM;
			double *ddVfdVec = ddVfdMat + elmIdx * DIMNUM;

			dqVfdVec[0] += 0.5 * dcVfdVec[0] + sgnInt * ddVfdVec[0];
			dqVfdVec[1] += 0.5 * dcVfdVec[1] + sgnInt * ddVfdVec[1];
		}
	}

	return;
}

void varifold(double *vfdPtr, double *dfmLmkPosMat, int *dfmElmVtxMat,
              double *tgtCenPosMat, double *tgtUniDirMat, double *tgtElmVolVec,
              char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth,
              double *dfmCenPosMat, double *dfmUniDirMat, double *dfmElmVolVec,
              int dfmLmkNum, int dfmElmNum, int tgtElmNum)
{
	landmarksToVarifoldKernel(dfmCenPosMat, dfmUniDirMat, dfmElmVolVec, 
	                          dfmLmkPosMat, dfmElmVtxMat, dfmElmNum);

	vf_Kernel(vfdPtr, dfmCenPosMat, dfmUniDirMat, dfmElmVolVec,
	          tgtCenPosMat, tgtUniDirMat, tgtElmVolVec,
	          cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
	          dfmElmNum, tgtElmNum);
	
	return;
}

void varifold(double *vfdPtr, double *dqVfdMat,
              double *dfmLmkPosMat, int *dfmElmVtxMat, int *dfmElmIfoMat,
              double *tgtCenPosMat, double *tgtUniDirMat, double *tgtElmVolVec,
              char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth,
              double *dfmCenPosMat, double *dfmUniDirMat, double *dfmElmVolVec,
              double *dcVfdMat, double *ddVfdMat,
              int dfmLmkNum, int dfmElmNum, int dfmElmIfoNum, int tgtElmNum)
{
	memset(dcVfdMat, 0, sizeof(double) * DIMNUM * dfmElmNum);
	memset(ddVfdMat, 0, sizeof(double) * DIMNUM * dfmElmNum);
	memset(dqVfdMat, 0, sizeof(double) * DIMNUM * dfmLmkNum);

	landmarksToVarifoldKernel(dfmCenPosMat, dfmUniDirMat, dfmElmVolVec, 
	                          dfmLmkPosMat, dfmElmVtxMat, dfmElmNum);
	
	dqVf_Kernel(vfdPtr, dcVfdMat, ddVfdMat, 
	            dfmCenPosMat, dfmUniDirMat, dfmElmVolVec,
	            tgtCenPosMat, tgtUniDirMat, tgtElmVolVec,
	            cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
	            dfmElmNum, tgtElmNum);

	dqVfdGatherKernel(dqVfdMat, dcVfdMat, ddVfdMat,
	                  dfmElmIfoMat, dfmElmIfoNum, dfmLmkNum);

	return;
}
