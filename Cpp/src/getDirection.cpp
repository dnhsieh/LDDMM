// Algorithm 7.4 in Nocedal
//
// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 05/20/2020

#include <cstring>
#include "blas.h"

void getDirection(double *dirVec, double HIniVal, double *grdNow, double *sMat, double *yMat, 
                  int newIdx, int hisNum, int memNum, double *alpVec, int varNum)
{
	// s   = x_next - x_now = dspVec
	// y   = (grad f)_next - (grad f)_now = dgdVec
	// rho = 1 / (s^T y)

	ptrdiff_t dotDim = varNum, incNum = 1;
	
	memcpy(dirVec, grdNow, sizeof(double) * varNum);

	for ( int hisCnt = 0; hisCnt < hisNum; ++hisCnt )
	{
		int hisIdx = newIdx - hisCnt;	
		if ( hisIdx < 0 ) hisIdx += memNum;

		double *sVec   = sMat + hisIdx * varNum;
		double *yVec   = yMat + hisIdx * varNum;

		double  sTq    = ddot(&dotDim, sVec, &incNum, dirVec, &incNum);
		double  sTy    = ddot(&dotDim, sVec, &incNum,   yVec, &incNum);
		double  alpVal = sTq / sTy;

		for ( int varIdx = 0; varIdx < varNum; ++varIdx )
			dirVec[varIdx] -= alpVal * yVec[varIdx];

		alpVec[hisIdx] = alpVal;
	}

	for ( int varIdx = 0; varIdx < varNum; ++varIdx )
		dirVec[varIdx] *= HIniVal;

	int oldIdx = (hisNum < memNum ? 0 : newIdx + 1);
	if ( oldIdx == memNum ) oldIdx = 0;

	for ( int hisCnt = 0; hisCnt < hisNum; ++hisCnt )
	{
		int hisIdx = oldIdx + hisCnt;
		if ( hisIdx >= memNum ) hisIdx -= memNum;

		double  alpVal = alpVec[hisIdx];

		double *sVec   = sMat + hisIdx * varNum;
		double *yVec   = yMat + hisIdx * varNum;

		double  yTr    = ddot(&dotDim, yVec, &incNum, dirVec, &incNum);
		double  sTy    = ddot(&dotDim, sVec, &incNum,   yVec, &incNum);
		double  btaVal = yTr / sTy;

		for ( int varIdx = 0; varIdx < varNum; ++varIdx )
			dirVec[varIdx] += (alpVal - btaVal) * sVec[varIdx];
	}

	for ( int varIdx = 0; varIdx < varNum; ++varIdx )
		dirVec[varIdx] = -dirVec[varIdx];

	return;
}
