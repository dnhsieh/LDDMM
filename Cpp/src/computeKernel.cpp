// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

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

void computeKernel(double *knlMat, double *lmkMat, int knlOrder, double knlWidth, int lmkNum, int dimNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	if ( knlOrder == -1 )   // Gaussian
	{
		for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
		{
			double *qiVec = lmkMat + lmkiIdx * dimNum;

			for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
			{
				double *qjVec = lmkMat + lmkjIdx * dimNum;

				double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
				double knlVal = exp(-dijVal * dijVal);
				knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
			}
		}
	}
	else   // Matern
	{
		if ( dimNum == 2 )
		{
			switch ( knlOrder )
			{
				case 0:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double p1Val;
							p1Fcn(p1Val, dijVal);

							double knlVal = p1Val;
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;

				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double knlVal = 0.5 * (p0Val + 2.0 * p1Val);
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;

				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double knlVal = (4.0 * p0Val + (8.0 + dijSqu) * p1Val) / 8.0;
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double knlVal = ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val) / 48.0;
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double knlVal = (12.0 * (16.0 + dijSqu) * p0Val + (384.0 + dijSqu * (72.0 + dijSqu)) * p1Val) / 384.0;
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;
			}
		}
		else   // dimNum == 3
		{
			switch ( knlOrder )
			{
				case 0:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = exp(-dijVal);
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;

				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (1.0 + dijVal) * exp(-dijVal);
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;

				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (3.0 + dijVal * (3.0 + dijVal)) / 3.0 * exp(-dijVal);
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / 15.0 * exp(-dijVal);
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (105.0 + dijVal * (105.0 + dijVal * (45.0 + dijVal * (10.0 + dijVal)))) / 105.0 * exp(-dijVal);
							knlMat[lmkiIdx * lmkNum + lmkjIdx] = knlVal;
						}
					}
					break;
			}
		}
	}

	return;
}

void computeKernel(double *knlMat, double *lmkiMat, double *lmkjMat,
                   int knlOrder, double knlWidth, int lmkiNum, int lmkjNum, int dimNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	if ( knlOrder == -1 )   // Gaussian
	{
		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			double *qjVec = lmkjMat + lmkjIdx * dimNum;

			for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
			{
				double *qiVec = lmkiMat + lmkiIdx * dimNum;

				double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
				double knlVal = exp(-dijVal * dijVal);
				knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
			}
		}
	}
	else   // Matern
	{
		if ( dimNum == 2 )
		{
			switch ( knlOrder )
			{
				case 0:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double p1Val;
							p1Fcn(p1Val, dijVal);

							double knlVal = p1Val;
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;

				case 1:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double knlVal = 0.5 * (p0Val + 2.0 * p1Val);
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;

				case 2:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double knlVal = (4.0 * p0Val + (8.0 + dijSqu) * p1Val) / 8.0;
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;

				case 3:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double knlVal = ((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val) / 48.0;
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;

				case 4:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double knlVal = (12.0 * (16.0 + dijSqu) * p0Val + (384.0 + dijSqu * (72.0 + dijSqu)) * p1Val) / 384.0;
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;
			}
		}
		else   // dimNum == 3
		{
			switch ( knlOrder )
			{
				case 0:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = exp(-dijVal);
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;

				case 1:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (1.0 + dijVal) * exp(-dijVal);
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;

				case 2:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (3.0 + dijVal * (3.0 + dijVal)) / 3.0 * exp(-dijVal);
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;

				case 3:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / 15.0 * exp(-dijVal);
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;

				case 4:
					for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
					{
						double *qjVec = lmkjMat + lmkjIdx * dimNum;
			
						for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
						{
							double *qiVec = lmkiMat + lmkiIdx * dimNum;
			
							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (105.0 + dijVal * (105.0 + dijVal * (45.0 + dijVal * (10.0 + dijVal)))) / 105.0 * exp(-dijVal);
							knlMat[lmkjIdx * lmkiNum + lmkiIdx] = knlVal;
						}
					}
					break;
			}
		}
	}

	return;
}
