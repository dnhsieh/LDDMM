// Modified from HANSO 2.0
//
// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 05/20/2020

#include <cstdio>
#include <cfloat>
#include <cmath>
#include "blas.h"
#include "struct.h"

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

void objgrd(double *, double *, double *, fcndata &);

void tryPosition(double *posTry, double *posNow, double stpTry, double *dirNow, int varNum)
{
	for ( int varIdx = 0; varIdx < varNum; ++varIdx )
		posTry[varIdx] = posNow[varIdx] + stpTry * dirNow[varIdx];

	return;
}

int lineSearch(double *posNow, double *grdNow, double *dirNow, double *fcnNow, 
               double wolfe1, double wolfe2, double tolVal,
               double *posTry, double *grdTry, double *fcnTry,
               double &stpTry, int &objCnt, fcndata &fcnObj)
{
	int       varNum = fcnObj.varNum;
	ptrdiff_t dotDim = varNum, incNum = 1;

	double relLft = 0.0;
	double relRgt = DBL_MAX;
	double relTry = 1.0;

	double slpNow = ddot(&dotDim, grdNow, &incNum, dirNow, &incNum);
	if ( slpNow > -DBL_MIN )
	{
		printf("\n");
		printf("Not a descent direction.\n");
		printf("Quit LBFGS.\n");
		printf("\n");
		return 1;
	}
	
	double dirSqu = ddot(&dotDim, dirNow, &incNum, dirNow, &incNum);
	double dirLen = sqrt(dirSqu);

	// arbitrary heuristic limits
	int bisMax = max(50, log2(1e5 * dirLen));
	int epdMax = max(10, log2(1e5 / dirLen));

	int bisCnt = 0, epdCnt = 0;
	objCnt = 0;
	while ( bisCnt <= bisMax && epdCnt <= epdMax )
	{
		tryPosition(posTry, posNow, relTry, dirNow, varNum);
		objgrd(fcnTry, grdTry, posTry, fcnObj);
		++objCnt;

		double grdSqu = ddot(&dotDim, grdTry, &incNum, grdTry, &incNum);
		if ( sqrt(grdSqu) < tolVal )
		{
			stpTry = relTry * dirLen;
			return 0;
		}

		double slpTry = ddot(&dotDim, grdTry, &incNum, dirNow, &incNum);

		if ( *fcnTry >= *fcnNow + wolfe1 * relTry * slpNow )
			relRgt = relTry;
		else if ( slpTry <= wolfe2 * slpNow )
			relLft = relTry;
		else
		{
			stpTry = relTry * dirLen;
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
