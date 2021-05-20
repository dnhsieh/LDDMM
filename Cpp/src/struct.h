#ifndef STRUCT_H
#define STRUCT_H

struct optdata
{
	int    varNum;

	int    memNum;
	int    itrMax;
	double tolVal;
	double wolfe1;
	double wolfe2;
	bool   vbsFlg;

	double *dspMat;   // varNum * memNum
	double *dgdMat;   // varNum * memNum
	double *dirVec;   // varNum
   double *posNxt;   // varNum
   double *grdNxt;   // varNum
   double *dspVec;   // varNum
   double *dgdVec;   // varNum
   double *recVec;   // memNum
};

struct parameters
{
	int     ndeNum;

	int     knlOrder;
	double  knlWidth;
	double  timeStp;
	int     timeNum;

	double *iniNdeMat;
};

struct discrepancy
{
	double disWgt;

	int dfmNdeNum;
	int dfmElmNum;
	int dfmElmIfoNum;

	int    *dfmElmVtxMat;
	int    *dfmElmIfoMat;

	double *dfmCenPosMat;    // DIMNUM * dfmElmNum
	double *dfmUniDirMat;    // DIMNUM * dfmElmNum
	double *dfmElmVolVec;    // dfmElmNum

	int tgtElmNum;

	double *tgtCenPosMat;
	double *tgtUniDirMat;
	double *tgtElmVolVec;

	char   cenKnlType;
	double cenKnlWidth;
	char   dirKnlType;
	double dirKnlWidth;

	double *dqVfdMat;        // DIMNUM * ndeNum
	double *dcVfdMat;        // DIMNUM * dfmElmNum
	double *ddVfdMat;        // DIMNUM * dfmElmNum
};

struct fcndata
{
	struct parameters  prm;
	struct discrepancy dis;

	int varNum;

	double *ndeStk;          // DIMNUM * ndeNum *  timeNum
	double *knlStk;          // ndeNum * ndeNum * (timeNum - 1)
	double *vlcMat;          // DIMNUM * ndeNum
	double *pMat;            // DIMNUM * ndeNum
	double *ampMat;          // DIMNUM * ndeNum
	double *pDotMat;         // DIMNUM * ndeNum
};

#endif
