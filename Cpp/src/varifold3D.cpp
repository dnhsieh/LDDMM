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
		int q2Idx = elmVtxMat[elmIdx * VFDVTXNUM + 2];

		double *q0Vec = lmkPosMat + q0Idx * DIMNUM;
		double *q1Vec = lmkPosMat + q1Idx * DIMNUM;
		double *q2Vec = lmkPosMat + q2Idx * DIMNUM;

		double *cenVec = cenPosMat + elmIdx * DIMNUM;
		vectorAverage(cenVec, q0Vec, q1Vec, q2Vec);

		double q10Vec[DIMNUM], q20Vec[DIMNUM];
		double *dirVec = uniDirMat + elmIdx * DIMNUM;
		vectorSubtract(q10Vec, q1Vec, q0Vec);
		vectorSubtract(q20Vec, q2Vec, q0Vec);
		crossProduct(dirVec, q10Vec, q20Vec);

		double elmVol = eucnorm(dirVec);
		dirVec[0] /= elmVol;
		dirVec[1] /= elmVol;
		dirVec[2] /= elmVol;
		
		elmVolVec[elmIdx] = 0.5 * elmVol;
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
		d1KVec[2] = d1KVal * (c1Vec[2] - c2Vec[2]);

		return;
	}

	if ( knlType == 'C' )   // cauchy
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = 1.0 / (1.0 + dstSqu / (knlWidth * knlWidth));

		double d1KVal = -2.0 * knlVal * knlVal / (knlWidth * knlWidth);
		d1KVec[0] = d1KVal * (c1Vec[0] - c2Vec[0]);
		d1KVec[1] = d1KVal * (c1Vec[1] - c2Vec[1]);
		d1KVec[2] = d1KVal * (c1Vec[2] - c2Vec[2]);

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
                      char knlType, double knlWidth, double n1Len)
{
	if ( knlType == 'B' )   // binet
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal * angVal;

		double d1KVal = 2.0 * angVal;
		d1KVec[0] = d1KVal / n1Len * (-angVal * v1Vec[0] + v2Vec[0]);
		d1KVec[1] = d1KVal / n1Len * (-angVal * v1Vec[1] + v2Vec[1]);
		d1KVec[2] = d1KVal / n1Len * (-angVal * v1Vec[2] + v2Vec[2]);

		return;
	}

	if ( knlType == 'L' )   // linear
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal;

		d1KVec[0] = 1.0 / n1Len * (-angVal * v1Vec[0] + v2Vec[0]);
		d1KVec[1] = 1.0 / n1Len * (-angVal * v1Vec[1] + v2Vec[1]);
		d1KVec[2] = 1.0 / n1Len * (-angVal * v1Vec[2] + v2Vec[2]);

		return;
	}

	if ( knlType == 'O' )   // gaussian oriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal - 1.0) / (knlWidth * knlWidth));

		double d1KVal = 2.0 * knlVal / (knlWidth * knlWidth);
		d1KVec[0] = d1KVal / n1Len * (-angVal * v1Vec[0] + v2Vec[0]);
		d1KVec[1] = d1KVal / n1Len * (-angVal * v1Vec[1] + v2Vec[1]);
		d1KVec[2] = d1KVal / n1Len * (-angVal * v1Vec[2] + v2Vec[2]);

		return;
	}

	if ( knlType == 'U' )   // gaussian unoriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal * angVal - 1.0) / (knlWidth * knlWidth));

		double d1KVal = 4.0 * angVal * knlVal / (knlWidth * knlWidth);
		d1KVec[0] = d1KVal / n1Len * (-angVal * v1Vec[0] + v2Vec[0]);
		d1KVec[1] = d1KVal / n1Len * (-angVal * v1Vec[1] + v2Vec[1]);
		d1KVec[2] = d1KVal / n1Len * (-angVal * v1Vec[2] + v2Vec[2]);

		return;
	}

	return;
}

void vf_Kernel(double *vfdPtr,
               double *dfmCenPosMat, double *dfmUniDirMat, double *dfmElmVolVec, 
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
			geometricFunction(cenKnlVal, dciKnlVec, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth                  );
			 grassmanFunction(dirKnlVal, ddiKnlVec, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth, 2.0 * dfmElmiVol);

			*vfdPtr += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;

			dciVfdVec[0] += 2.0 * dciKnlVec[0] * dirKnlVal * dfmElmiVol * dfmElmjVol;
			dciVfdVec[1] += 2.0 * dciKnlVec[1] * dirKnlVal * dfmElmiVol * dfmElmjVol;
			dciVfdVec[2] += 2.0 * dciKnlVec[2] * dirKnlVal * dfmElmiVol * dfmElmjVol;

			ddiVfdVec[0] += 2.0 * cenKnlVal * (  ddiKnlVec[0] *       dfmElmiVol
			                                   + dirKnlVal   * 0.5 * dfmDiriVec[0] ) * dfmElmjVol;

			ddiVfdVec[1] += 2.0 * cenKnlVal * (  ddiKnlVec[1] *       dfmElmiVol
			                                   + dirKnlVal   * 0.5 * dfmDiriVec[1] ) * dfmElmjVol;

			ddiVfdVec[2] += 2.0 * cenKnlVal * (  ddiKnlVec[2] *       dfmElmiVol
			                                   + dirKnlVal   * 0.5 * dfmDiriVec[2] ) * dfmElmjVol;
		}

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			double *tgtCenjVec = tgtCenPosMat + tgtElmjIdx * DIMNUM;
			double *tgtDirjVec = tgtUniDirMat + tgtElmjIdx * DIMNUM;

			double tgtElmjVol = tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			double dciKnlVec[DIMNUM], ddiKnlVec[DIMNUM];
			geometricFunction(cenKnlVal, dciKnlVec, dfmCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth                  );
			 grassmanFunction(dirKnlVal, ddiKnlVec, dfmDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth, 2.0 * dfmElmiVol);

			*vfdPtr -= 2.0 * cenKnlVal * dirKnlVal * dfmElmiVol * tgtElmjVol;

			dciVfdVec[0] -= 2.0 * dciKnlVec[0] * dirKnlVal * dfmElmiVol * tgtElmjVol;
			dciVfdVec[1] -= 2.0 * dciKnlVec[1] * dirKnlVal * dfmElmiVol * tgtElmjVol;
			dciVfdVec[2] -= 2.0 * dciKnlVec[2] * dirKnlVal * dfmElmiVol * tgtElmjVol;

			ddiVfdVec[0] -= 2.0 * cenKnlVal * (  ddiKnlVec[0] *       dfmElmiVol 
			                                   + dirKnlVal   * 0.5 * dfmDiriVec[0] ) * tgtElmjVol;

			ddiVfdVec[1] -= 2.0 * cenKnlVal * (  ddiKnlVec[1] *       dfmElmiVol
			                                   + dirKnlVal   * 0.5 * dfmDiriVec[1] ) * tgtElmjVol;

			ddiVfdVec[2] -= 2.0 * cenKnlVal * (  ddiKnlVec[2] *       dfmElmiVol
			                                   + dirKnlVal   * 0.5 * dfmDiriVec[2] ) * tgtElmjVol;
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
                       int *dfmElmIfoMat, double *dfmLmkPosMat,
                       int dfmElmNum, int dfmElmIfoNum, int dfmLmkNum)
{
	for ( int dfmLmkIdx = 0; dfmLmkIdx < dfmLmkNum; ++dfmLmkIdx )
	{
		double *dqVfdVec = dqVfdMat + dfmLmkIdx * DIMNUM;

		int adjNum = dfmElmIfoMat[dfmLmkIdx * dfmElmIfoNum];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int elmIdx = dfmElmIfoMat[dfmLmkIdx * dfmElmIfoNum + 1 + 3 * adjIdx    ];
			int  qaIdx = dfmElmIfoMat[dfmLmkIdx * dfmElmIfoNum + 1 + 3 * adjIdx + 1];
			int  qbIdx = dfmElmIfoMat[dfmLmkIdx * dfmElmIfoNum + 1 + 3 * adjIdx + 2];

			double *dcVfdVec = dcVfdMat + elmIdx * DIMNUM;
			double *ddVfdVec = ddVfdMat + elmIdx * DIMNUM;

			double *qaDfmVec = dfmLmkPosMat + qaIdx * DIMNUM;
			double *qbDfmVec = dfmLmkPosMat + qbIdx * DIMNUM;

			double qabDfmVec[DIMNUM], ddVfdxqabVec[DIMNUM];
			vectorSubtract(qabDfmVec, qaDfmVec, qbDfmVec);
			crossProduct(ddVfdxqabVec, ddVfdVec, qabDfmVec);

			dqVfdVec[0] += dcVfdVec[0] / 3.0 + ddVfdxqabVec[0];
			dqVfdVec[1] += dcVfdVec[1] / 3.0 + ddVfdxqabVec[1];
			dqVfdVec[2] += dcVfdVec[2] / 3.0 + ddVfdxqabVec[2];
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
	                  dfmElmIfoMat, dfmLmkPosMat, dfmElmNum, dfmElmIfoNum, dfmLmkNum);

	return;
}
