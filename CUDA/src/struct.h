#ifndef STRUCT_H
#define STRUCT_H

#include <cublas_v2.h>

struct optdata
{
	int    varNum;

	int    memNum;
	int    itrMax;
	double tolVal;
	double wolfe1;
	double wolfe2;
	bool   vbsFlg;

	double *d_dspMat;   // varNum * memNum
	double *d_dgdMat;   // varNum * memNum
	double *d_dirVec;   // varNum
   double *d_posNxt;   // varNum
   double *d_grdNxt;   // varNum
   double *d_dspVec;   // varNum
   double *d_dgdVec;   // varNum
   double *h_recVec;   // memNum
};

struct parameters
{
	int     ndeNum;

	int     knlOrder;
	double  knlWidth;
	double  timeStp;
	int     timeNum;

	double *d_iniNdeMat;
};

struct discrepancy
{
	double disWgt;

	int dfmNdeNum;
	int dfmElmNum;

	int    *d_dfmElmVtxMat;
	int    *d_dfmElmIfoMat;

	double *d_dfmCenPosMat;    // DIMNUM * dfmElmNum
	double *d_dfmUniDirMat;    // DIMNUM * dfmElmNum
	double *d_dfmElmVolVec;    // dfmElmNum

	int tgtElmNum;

	double *d_tgtCenPosMat;
	double *d_tgtUniDirMat;
	double *d_tgtElmVolVec;

	char   cenKnlType;
	double cenKnlWidth;
	char   dirKnlType;
	double dirKnlWidth;

	double *d_vfdVec;          // dfmElmNum + tgtElmNum
	double *d_sumBufVec;       // SUMBLKDIM

	double *d_dqVfdMat;        // DIMNUM * ndeNum
	double *d_dcVfdMat;        // DIMNUM * dfmElmNum
	double *d_ddVfdMat;        // DIMNUM * dfmElmNum
};

struct fcndata
{
	struct parameters  prm;
	struct discrepancy dis;

	int varNum;

	double *d_ndeStk;          // DIMNUM * ndeNum * timeNum
	double *d_knlMat;          // ndeNum * ndeNum
	double *d_vlcMat;          // DIMNUM * ndeNum
	double *d_pMat;            // DIMNUM * ndeNum
	double *d_ampMat;          // DIMNUM * ndeNum
	double *d_pDotMat;         // DIMNUM * ndeNum

	cublasHandle_t blasHdl;
};

#endif
