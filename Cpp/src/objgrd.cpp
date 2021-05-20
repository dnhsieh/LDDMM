#include <cstring>
#include "blas.h"
#include "struct.h"
#include "constants.h"

void computeKernel(double *, double *, int, double, int, int);
void varifoldITF(double *, double *, double *, fcndata &);
void dqKernel(double *, double *, double *, double *, int, double, int, int);

void xpby(double *outVec, double *xVec, double bVal, double *yVec, int varNum)
{
	for ( int varIdx = 0; varIdx < varNum; ++varIdx )
		outVec[varIdx] = xVec[varIdx] + bVal * yVec[varIdx];

	return;
}

void scaleVector(double *outVec, double scale, double *inpVec, int varNum)
{
	for ( int varIdx = 0; varIdx < varNum; ++varIdx )
		outVec[varIdx] = scale * inpVec[varIdx];

	return;
}

void objgrd(double *fcnVal, double *grdStk, double *alpStk, fcndata &fcnObj)
{
	int    ndeNum  = fcnObj.prm.ndeNum;
	int    timeNum = fcnObj.prm.timeNum;
	double timeStp = fcnObj.prm.timeStp;

	int    knlOrder = fcnObj.prm.knlOrder;
	double knlWidth = fcnObj.prm.knlWidth;

	char      transA = 'N', transK = 'N';
	ptrdiff_t rowNum = DIMNUM, colNum = ndeNum, innNum = ndeNum, incNum = 1;
	double    oneVal = 1.0, zroVal = 0.0;

	ptrdiff_t dotDim = DIMNUM * ndeNum;

	memcpy(fcnObj.ndeStk, fcnObj.prm.iniNdeMat, sizeof(double) * DIMNUM * ndeNum);
	memset(grdStk, 0, sizeof(double) * DIMNUM * ndeNum * (timeNum - 1));

	*fcnVal = 0.0;
	for ( int timeIdx = 0; timeIdx < timeNum - 1; ++timeIdx )
	{
		double *ndeNowMat = fcnObj.ndeStk +  timeIdx      * DIMNUM * ndeNum;
		double *ndeNxtMat = fcnObj.ndeStk + (timeIdx + 1) * DIMNUM * ndeNum;
		double *knlMat    = fcnObj.knlStk +  timeIdx      * ndeNum * ndeNum;
		double *alpMat    =        alpStk +  timeIdx      * DIMNUM * ndeNum;

		computeKernel(knlMat, ndeNowMat, knlOrder, knlWidth, ndeNum, DIMNUM);
		dgemm(&transA, &transK, &rowNum, &colNum, &innNum,
		      &oneVal, alpMat, &rowNum, knlMat, &innNum, &zroVal, fcnObj.vlcMat, &rowNum);

		*fcnVal += ddot(&dotDim, alpMat, &incNum, fcnObj.vlcMat, &incNum);

		xpby(ndeNxtMat, ndeNowMat, timeStp, fcnObj.vlcMat, DIMNUM * ndeNum);	
	}

	double *endNdeMat = fcnObj.ndeStk + (timeNum - 1) * DIMNUM * ndeNum;
	double  vfdVal;
	varifoldITF(&vfdVal, fcnObj.dis.dqVfdMat, endNdeMat, fcnObj);

	*fcnVal = timeStp * 0.5 * (*fcnVal) + fcnObj.dis.disWgt * vfdVal;

	scaleVector(fcnObj.pMat, -fcnObj.dis.disWgt, fcnObj.dis.dqVfdMat, DIMNUM * ndeNum);

	for ( int timeIdx = timeNum - 2; timeIdx >= 0; --timeIdx )
	{
		double *ndeMat = fcnObj.ndeStk + timeIdx * DIMNUM * ndeNum;
		double *knlMat = fcnObj.knlStk + timeIdx * ndeNum * ndeNum;
		double *alpMat =        alpStk + timeIdx * DIMNUM * ndeNum;
		double *grdMat =        grdStk + timeIdx * DIMNUM * ndeNum;
	
		xpby(fcnObj.ampMat, alpMat, -1.0, fcnObj.pMat, DIMNUM * ndeNum);
		dgemm(&transA, &transK, &rowNum, &colNum, &innNum,
		      &oneVal, fcnObj.ampMat, &rowNum, knlMat, &innNum, &zroVal, grdMat, &rowNum);

		xpby(fcnObj.ampMat, fcnObj.ampMat, -0.5, alpMat, DIMNUM * ndeNum);
		dqKernel(fcnObj.pDotMat, ndeMat, fcnObj.ampMat, alpMat,
		         knlOrder, knlWidth, ndeNum, DIMNUM);

		xpby(fcnObj.pMat, fcnObj.pMat, -timeStp, fcnObj.pDotMat, DIMNUM * ndeNum);
	}

	scaleVector(grdStk, timeStp, grdStk, DIMNUM * ndeNum * (timeNum - 1));

	return;
}
