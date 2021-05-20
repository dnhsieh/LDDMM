#include <cstring>
#include <cublas_v2.h>
#include "struct.h"
#include "constants.h"

void computeKernel(double *, double *, int, double, int);
void varifoldITF(double *, double *, double *, fcndata &);
void dqKernel(double *, double *, double *, double *, int, double, int);

void xpby(double *, double *, double, double *, int);
void scaleVector(double *, double, double *, int);

void objgrd(double *h_fcnVal, double *d_grdStk, double *d_alpStk, fcndata &fcnObj)
{
	int    ndeNum  = fcnObj.prm.ndeNum;
	int    timeNum = fcnObj.prm.timeNum;
	double timeStp = fcnObj.prm.timeStp;

	int    knlOrder = fcnObj.prm.knlOrder;
	double knlWidth = fcnObj.prm.knlWidth;

	double dotDim = ndeNum * DIMNUM;
	double oneVal = 1.0;
	double zroVal = 0.0;

	cudaMemcpy(fcnObj.d_ndeStk, fcnObj.prm.d_iniNdeMat, sizeof(double) * ndeNum * DIMNUM, cudaMemcpyDeviceToDevice);
	cudaMemset(d_grdStk, 0, sizeof(double) * ndeNum * DIMNUM * (timeNum - 1));

	*h_fcnVal = 0.0;
	for ( int timeIdx = 0; timeIdx < timeNum - 1; ++timeIdx )
	{
		double *d_ndeNowMat = fcnObj.d_ndeStk +  timeIdx      * ndeNum * DIMNUM;
		double *d_ndeNxtMat = fcnObj.d_ndeStk + (timeIdx + 1) * ndeNum * DIMNUM;
		double *d_alpMat    =        d_alpStk +  timeIdx      * ndeNum * DIMNUM;

		computeKernel(fcnObj.d_knlMat, d_ndeNowMat, knlOrder, knlWidth, ndeNum);
		cublasDgemm(fcnObj.blasHdl, CUBLAS_OP_N, CUBLAS_OP_N, ndeNum, DIMNUM, ndeNum,
		            &oneVal, fcnObj.d_knlMat, ndeNum, d_alpMat, ndeNum, &zroVal, fcnObj.d_vlcMat, ndeNum);

		double h_fcnNow;
		cublasDdot(fcnObj.blasHdl, dotDim, d_alpMat, 1, fcnObj.d_vlcMat, 1, &h_fcnNow);
		*h_fcnVal += h_fcnNow;

		xpby(d_ndeNxtMat, d_ndeNowMat, timeStp, fcnObj.d_vlcMat, ndeNum * DIMNUM);	
	}

	double *d_endNdeMat = fcnObj.d_ndeStk + (timeNum - 1) * ndeNum * DIMNUM;
	double  h_vfdVal;
	varifoldITF(&h_vfdVal, fcnObj.dis.d_dqVfdMat, d_endNdeMat, fcnObj);

	*h_fcnVal = timeStp * 0.5 * (*h_fcnVal) + fcnObj.dis.disWgt * h_vfdVal;

	scaleVector(fcnObj.d_pMat, -fcnObj.dis.disWgt, fcnObj.dis.d_dqVfdMat, ndeNum * DIMNUM);

	for ( int timeIdx = timeNum - 2; timeIdx >= 0; --timeIdx )
	{
		double *d_ndeMat = fcnObj.d_ndeStk + timeIdx * ndeNum * DIMNUM;
		double *d_alpMat =        d_alpStk + timeIdx * ndeNum * DIMNUM;
		double *d_grdMat =        d_grdStk + timeIdx * ndeNum * DIMNUM;
	
		xpby(fcnObj.d_ampMat, d_alpMat, -1.0, fcnObj.d_pMat, ndeNum * DIMNUM);
		computeKernel(fcnObj.d_knlMat, d_ndeMat, knlOrder, knlWidth, ndeNum);
		cublasDgemm(fcnObj.blasHdl, CUBLAS_OP_N, CUBLAS_OP_N, ndeNum, DIMNUM, ndeNum,
		            &oneVal, fcnObj.d_knlMat, ndeNum, fcnObj.d_ampMat, ndeNum, &zroVal, d_grdMat, ndeNum);

		xpby(fcnObj.d_ampMat, fcnObj.d_ampMat, -0.5, d_alpMat, ndeNum * DIMNUM);
		dqKernel(fcnObj.d_pDotMat, d_ndeMat, fcnObj.d_ampMat, d_alpMat,
		         knlOrder, knlWidth, ndeNum);

		xpby(fcnObj.d_pMat, fcnObj.d_pMat, -timeStp, fcnObj.d_pDotMat, ndeNum * DIMNUM);
	}

	scaleVector(d_grdStk, timeStp, d_grdStk, ndeNum * DIMNUM * (timeNum - 1));

	return;
}
