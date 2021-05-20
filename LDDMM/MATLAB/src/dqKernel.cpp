// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cstring>
#include <cmath>

void p0Fcn(double &p0Val, double xVal);
void p1Fcn(double &p1Val, double xVal);

double eucdist(double *v1Vec, double *v2Vec, int dimNum)
{
	double dstSqu = 0.0;
	for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
	{
		double difVal = v1Vec[dimIdx] - v2Vec[dimIdx];
		dstSqu += difVal * difVal;
	}

	return sqrt(dstSqu);
}

double dotProduct(double *v1Vec, double *v2Vec, int dimNum)
{
	double dotVal = 0.0;
	for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
		dotVal += v1Vec[dimIdx] * v2Vec[dimIdx];

	return dotVal;
}

void dqKernel(double *dqKMat, double *lmkMat, double *lftMat, double *rgtMat,
              int knlOrder, double knlWidth, int lmkNum, int dimNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	double knlWidthSqu = knlWidth * knlWidth;

	memset(dqKMat, 0, sizeof(double) * dimNum * lmkNum);

	if ( knlOrder == -1 )   // Gaussian
	{
		for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
		{
			double   *qiVec = lmkMat + lmkiIdx * dimNum;
			double   *liVec = lftMat + lmkiIdx * dimNum;
			double   *riVec = rgtMat + lmkiIdx * dimNum;
			double *dqiKVec = dqKMat + lmkiIdx * dimNum;

			for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
			{
				double *qjVec = lmkMat + lmkjIdx * dimNum;
				double *ljVec = lftMat + lmkjIdx * dimNum;
				double *rjVec = rgtMat + lmkjIdx * dimNum;

				double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
				double dqKVal = -2.0 * exp(-dijVal * dijVal) / knlWidthSqu;
				double  lrVal =  dotProduct(liVec, rjVec, dimNum)
				               + dotProduct(ljVec, riVec, dimNum);
				
				for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
					dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
			}
		}
	}
	else   // Matern
	{
		if ( dimNum == 2 )
		{
			switch ( knlOrder )
			{
				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double p1Val;
							p1Fcn(p1Val, dijVal);

							double dqKVal = -1.0 / (2.0 * knlWidthSqu) * p1Val;
							double  lrVal =  dotProduct(liVec, rjVec, dimNum)
							               + dotProduct(ljVec, riVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
						}
					}
					break;

				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double dqKVal = -1.0 / (8.0 * knlWidthSqu) * (p0Val + 2.0 * p1Val);
							double  lrVal =  dotProduct(liVec, rjVec, dimNum)
							               + dotProduct(ljVec, riVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double dqKVal = -1.0 / (48.0 * knlWidthSqu) * (4.0 * p0Val + (8.0 + dijSqu) * p1Val);
							double  lrVal =  dotProduct(liVec, rjVec, dimNum)
							               + dotProduct(ljVec, riVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double dqKVal = -1.0 / (384.0 * knlWidthSqu)
							               * ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
							double  lrVal =  dotProduct(liVec, rjVec, dimNum)
							               + dotProduct(ljVec, riVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
						}
					}
					break;
			}
		}
		else   // dimNum == 3
		{
			switch ( knlOrder )
			{
				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dqKVal = -exp(-dijVal) / knlWidthSqu;
							double  lrVal =  dotProduct(liVec, rjVec, dimNum)
							               + dotProduct(ljVec, riVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
						}
					}
					break;
			
				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dqKVal = -(1.0 + dijVal) * exp(-dijVal) / (3.0 * knlWidthSqu);
							double  lrVal =  dotProduct(liVec, rjVec, dimNum)
							               + dotProduct(ljVec, riVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
						}
					}
					break;
			
				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dqKVal = -(3.0 + dijVal * (3.0 + dijVal)) * exp(-dijVal) / (15.0 * knlWidthSqu);
							double  lrVal =  dotProduct(liVec, rjVec, dimNum)
							               + dotProduct(ljVec, riVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
						}
					}
					break;
			
				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dqKVal = -(15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) * exp(-dijVal)
							                / (105.0 * knlWidthSqu);
							double  lrVal =  dotProduct(liVec, rjVec, dimNum)
							               + dotProduct(ljVec, riVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
						}
					}
					break;
			}
		}
	}

	return;
}

void dqKernel(double *dqiKMat, double *dqjKMat, double *lmkiMat, double *lmkjMat, double *lftMat, double *rgtMat,
              int knlOrder, double knlWidth, int lmkiNum, int lmkjNum, int dimNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	double knlWidthSqu = knlWidth * knlWidth;

	memset(dqiKMat, 0, sizeof(double) * dimNum * lmkiNum);
	memset(dqjKMat, 0, sizeof(double) * dimNum * lmkjNum);

	if ( knlOrder == -1 )   // Gaussian
	{
		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			double   *qiVec = lmkiMat + lmkiIdx * dimNum;
			double   *liVec =  lftMat + lmkiIdx * dimNum;
			double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

			for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
			{
				double   *qjVec = lmkjMat + lmkjIdx * dimNum;
				double   *rjVec =  rgtMat + lmkjIdx * dimNum;
				double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

				double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
				double dqKVal = -2.0 * exp(-dijVal * dijVal) / knlWidthSqu;
				double  lrVal = dotProduct(liVec, rjVec, dimNum);
				
				for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
				{
					dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
					dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
				}
			}
		}
	}
	else   // Matern
	{
		if ( dimNum == 2 )
		{
			switch ( knlOrder )
			{
				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double   *qiVec = lmkiMat + lmkiIdx * dimNum;
						double   *liVec =  lftMat + lmkiIdx * dimNum;
						double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double   *qjVec = lmkjMat + lmkjIdx * dimNum;
							double   *rjVec =  rgtMat + lmkjIdx * dimNum;
							double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double p1Val;
							p1Fcn(p1Val, dijVal);

							double dqKVal = -1.0 / (2.0 * knlWidthSqu) * p1Val;
							double  lrVal = dotProduct(liVec, rjVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
								dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
							}
						}
					}
					break;

				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double   *qiVec = lmkiMat + lmkiIdx * dimNum;
						double   *liVec =  lftMat + lmkiIdx * dimNum;
						double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double   *qjVec = lmkjMat + lmkjIdx * dimNum;
							double   *rjVec =  rgtMat + lmkjIdx * dimNum;
							double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double dqKVal = -1.0 / (8.0 * knlWidthSqu) * (p0Val + 2.0 * p1Val);
							double  lrVal = dotProduct(liVec, rjVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
								dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
							}
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double   *qiVec = lmkiMat + lmkiIdx * dimNum;
						double   *liVec =  lftMat + lmkiIdx * dimNum;
						double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double   *qjVec = lmkjMat + lmkjIdx * dimNum;
							double   *rjVec =  rgtMat + lmkjIdx * dimNum;
							double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double dqKVal = -1.0 / (48.0 * knlWidthSqu) * (4.0 * p0Val + (8.0 + dijSqu) * p1Val);
							double  lrVal = dotProduct(liVec, rjVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
								dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
							}
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double   *qiVec = lmkiMat + lmkiIdx * dimNum;
						double   *liVec =  lftMat + lmkiIdx * dimNum;
						double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double   *qjVec = lmkjMat + lmkjIdx * dimNum;
							double   *rjVec =  rgtMat + lmkjIdx * dimNum;
							double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double dqKVal = -1.0 / (384.0 * knlWidthSqu)
							               * ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
							double  lrVal = dotProduct(liVec, rjVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
								dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
							}
						}
					}
					break;
			}
		}
		else   // dimNum == 3
		{
			switch ( knlOrder )
			{
				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double   *qiVec = lmkiMat + lmkiIdx * dimNum;
						double   *liVec =  lftMat + lmkiIdx * dimNum;
						double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double   *qjVec = lmkjMat + lmkjIdx * dimNum;
							double   *rjVec =  rgtMat + lmkjIdx * dimNum;
							double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dqKVal = -exp(-dijVal) / knlWidthSqu;
							double  lrVal = dotProduct(liVec, rjVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
								dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
							}
						}
					}
					break;
			
				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double   *qiVec = lmkiMat + lmkiIdx * dimNum;
						double   *liVec =  lftMat + lmkiIdx * dimNum;
						double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double   *qjVec = lmkjMat + lmkjIdx * dimNum;
							double   *rjVec =  rgtMat + lmkjIdx * dimNum;
							double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dqKVal = -(1.0 + dijVal) * exp(-dijVal) / (3.0 * knlWidthSqu);
							double  lrVal = dotProduct(liVec, rjVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
								dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
							}
						}
					}
					break;
			
				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double   *qiVec = lmkiMat + lmkiIdx * dimNum;
						double   *liVec =  lftMat + lmkiIdx * dimNum;
						double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double   *qjVec = lmkjMat + lmkjIdx * dimNum;
							double   *rjVec =  rgtMat + lmkjIdx * dimNum;
							double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dqKVal = -(3.0 + dijVal * (3.0 + dijVal)) * exp(-dijVal) / (15.0 * knlWidthSqu);
							double  lrVal = dotProduct(liVec, rjVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
								dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
							}
						}
					}
					break;
			
				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double   *qiVec = lmkiMat + lmkiIdx * dimNum;
						double   *liVec =  lftMat + lmkiIdx * dimNum;
						double *dqiKVec = dqiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double   *qjVec = lmkjMat + lmkjIdx * dimNum;
							double   *rjVec =  rgtMat + lmkjIdx * dimNum;
							double *dqjKVec = dqjKMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dqKVal = -(15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) * exp(-dijVal)
							                / (105.0 * knlWidthSqu);
							double  lrVal = dotProduct(liVec, rjVec, dimNum);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += lrVal * dqKVal * (qiVec[dimIdx] - qjVec[dimIdx]);
								dqjKVec[dimIdx] += lrVal * dqKVal * (qjVec[dimIdx] - qiVec[dimIdx]);
							}
						}
					}
					break;
			}
		}
	}

	return;
}
