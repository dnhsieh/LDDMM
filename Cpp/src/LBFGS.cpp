// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 06/10/2020

#include <cstdio>
#include <cstring>
#include <cmath>
#include "blas.h"
#include "struct.h"

void objgrd(double *, double *, double *, fcndata &);

int  lineSearch(double *, double *, double *, double *, double, double, double, double *, double *, double *,
                double &, int &, fcndata &);
void getDirection(double *, double, double *, double *, double *, 
                  int, int, int, double *, int);

void vectorSubtract(double *v12Vec, double *v1Vec, double *v2Vec, int varNum)
{
	for ( int varIdx = 0; varIdx < varNum; ++varIdx )
		v12Vec[varIdx] = v1Vec[varIdx] - v2Vec[varIdx];

	return;
}

int LBFGS(double *fcnNow, double *grdNow, double *posNow, optdata &optObj, fcndata &fcnObj)
{
	int     varNum = fcnObj.varNum;
	int     itrMax = optObj.itrMax;
	double  tolVal = optObj.tolVal;
	int     memNum = optObj.memNum;
	double  wolfe1 = optObj.wolfe1;
	double  wolfe2 = optObj.wolfe2;
	bool    vbsFlg = optObj.vbsFlg;

	double *dspMat = optObj.dspMat;
	double *dgdMat = optObj.dgdMat;
	double *dirVec = optObj.dirVec;
	double *posNxt = optObj.posNxt;
	double *grdNxt = optObj.grdNxt;
	double *dspVec = optObj.dspVec;
	double *dgdVec = optObj.dgdVec;
	double *recVec = optObj.recVec;
	double  fcnNxt;

	ptrdiff_t dotDim = varNum, incNum = 1;

	memset(dspMat, 0, sizeof(double) * varNum * memNum);
	memset(dgdMat, 0, sizeof(double) * varNum * memNum);

	objgrd(fcnNow, grdNow, posNow, fcnObj);
	double grdSqu = ddot(&dotDim, grdNow, &incNum, grdNow, &incNum);
	double grdLen = sqrt(grdSqu);

	if ( vbsFlg )
	{
		printf("%5s   %13s  %13s  %13s  %9s\n", "iter", "f", "|grad f|", "step length", "fcn eval");
		char sepStr[65] = {0};
		memset(sepStr, '-', 62);
		printf("%s\n", sepStr);
		printf("%5d:  %13.6e  %13.6e\n", 0, *fcnNow, grdLen);
	}

	int newIdx = -1;
	for ( int itrIdx = 1; itrIdx <= itrMax; ++itrIdx )
	{
		if ( grdLen < tolVal )
			break;

		double HIniVal;
		if ( newIdx == -1 )
			HIniVal = 1.0;
		else
		{
			double *dspPtr = dspMat + newIdx * varNum;
			double *dgdPtr = dgdMat + newIdx * varNum;

			double dspDgd = ddot(&dotDim, dspPtr, &incNum, dgdPtr, &incNum);
			double dgdDgd = ddot(&dotDim, dgdPtr, &incNum, dgdPtr, &incNum);

			HIniVal = dspDgd / dgdDgd;
		}

		if ( itrIdx <= memNum )
			getDirection(dirVec, HIniVal, grdNow, dspMat, dgdMat, newIdx, itrIdx - 1, memNum, recVec, varNum);
		else                                                             
			getDirection(dirVec, HIniVal, grdNow, dspMat, dgdMat, newIdx, memNum, memNum, recVec, varNum);

		double stpLen;
		int    objCnt;
		int    lineErr = lineSearch(posNow, grdNow, dirVec, fcnNow, wolfe1, wolfe2, tolVal,
		                            posNxt, grdNxt, &fcnNxt, stpLen, objCnt, fcnObj);
		if ( lineErr != 0 ) return 1;

		vectorSubtract(dspVec, posNxt, posNow, varNum);
		vectorSubtract(dgdVec, grdNxt, grdNow, varNum);

		newIdx = (newIdx == memNum - 1) ? 0 : (newIdx + 1);
		memcpy(dspMat + newIdx * varNum, dspVec, sizeof(double) * varNum);
		memcpy(dgdMat + newIdx * varNum, dgdVec, sizeof(double) * varNum);
	
		memcpy(posNow, posNxt, sizeof(double) * varNum);
		memcpy(grdNow, grdNxt, sizeof(double) * varNum);
		*fcnNow = fcnNxt;
		grdSqu  = ddot(&dotDim, grdNow, &incNum, grdNow, &incNum);
		grdLen  = sqrt(grdSqu);

		if ( vbsFlg )
			printf("%5d:  %13.6e  %13.6e  %13.6e  %9d\n", itrIdx, *fcnNow, grdLen, stpLen, objCnt);
	}

	return 0;
}
