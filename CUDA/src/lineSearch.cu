// Modified from HANSO 2.0
//
// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 05/20/2020

#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cublas_v2.h>
#include "struct.h"
#include "constants.h"

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

void objgrd(double *, double *, double *, fcndata &);

__global__ void tryPositionKernel(double *d_posTry, double *d_posNow, 
                                  double stpTry, double *d_dirNow, int varNum)
{
	int varIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( varIdx < varNum )
		d_posTry[varIdx] = d_posNow[varIdx] + stpTry * d_dirNow[varIdx];

	return;
}

void tryPosition(double *d_posTry, double *d_posNow, double stpTry, double *d_dirNow, int varNum)
{
	int blkNum = (varNum - 1) / BLKDIM + 1;
	tryPositionKernel <<<blkNum, BLKDIM>>> (d_posTry, d_posNow, stpTry, d_dirNow, varNum);

	return;
}

// modified from HANSO 2.0
int lineSearch(double *d_posNow, double *d_grdNow, double *d_dirNow, double *h_fcnNow,
               double wolfe1, double wolfe2, double tolVal,
               double *d_posTry, double *d_grdTry, double *h_fcnTry, 
               double &stpTry, int &objCnt, fcndata &fcnObj)
{
	int varNum = fcnObj.varNum;

	double relLft = 0.0;
	double relRgt = DBL_MAX;
	double relTry = 1.0;

	double h_slpNow;
	cublasDdot(fcnObj.blasHdl, varNum, d_grdNow, 1, d_dirNow, 1, &h_slpNow);
	if ( h_slpNow > -DBL_MIN )
	{
		printf("\n");
		printf("Not a descent direction.\n");
		printf("Quit LBFGS.\n");
		printf("\n");
		return 1;
	}
	
	double h_dirSqu;
	cublasDdot(fcnObj.blasHdl, varNum, d_dirNow, 1, d_dirNow, 1, &h_dirSqu);
	double h_dirLen = sqrt(h_dirSqu);

	// arbitrary heuristic limits
	int bisMax = max(50, log2(1e5 * h_dirLen));
	int epdMax = max(10, log2(1e5 / h_dirLen));

	int bisCnt = 0, epdCnt = 0;
	objCnt = 0;
	while ( bisCnt <= bisMax && epdCnt <= epdMax )
	{
		tryPosition(d_posTry, d_posNow, relTry, d_dirNow, varNum);
		objgrd(h_fcnTry, d_grdTry, d_posTry, fcnObj);
		++objCnt;

		double h_grdSqu;
		cublasDdot(fcnObj.blasHdl, varNum, d_grdTry, 1, d_grdTry, 1, &h_grdSqu);
		if ( sqrt(h_grdSqu) < tolVal )
		{
			stpTry = relTry * h_dirLen;
			return 0;
		}

		double h_slpTry;
		cublasDdot(fcnObj.blasHdl, varNum, d_grdTry, 1, d_dirNow, 1, &h_slpTry);

		if ( *h_fcnTry >= *h_fcnNow + wolfe1 * relTry * h_slpNow )
			relRgt = relTry;
		else if ( h_slpTry <= wolfe2 * h_slpNow )
			relLft = relTry;
		else
		{
			stpTry = relTry * h_dirLen;
			return 0;
		}

		if ( relRgt == DBL_MAX )
		{
			relTry *= 2;
			++epdCnt;
		}
		else
		{
			relTry = 0.5 * (relLft + relRgt);
			++bisCnt;
		}
	}

	if ( relRgt == DBL_MAX )
	{
		printf("\n");
		printf("Line search failed to bracket a point satisfying weak Wolfe conditions.\n");
		printf("Function may be unbounded below.\n");
		printf("Quit LBFGS.\n");
		printf("\n");
		return 2;
	}

	printf("\n");
	printf("Line search failed to satisfy weak Wolfe conditions, although\n");
	printf("a point satisfying conditions was bracketed in [%e, %e].\n", relLft, relRgt);
	printf("Quit LBFGS.\n");
	printf("\n");
	return 3;
}
