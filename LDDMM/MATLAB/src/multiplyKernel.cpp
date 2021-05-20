// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cstring>
#include <cmath>

void p0Fcn(double &f0Val, double xVal);
void p1Fcn(double &f1Val, double xVal);

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

void multiplyKernel(double *vlcMat, double *lmkMat, double *alpMat,
                    int knlOrder, double knlWidth, int lmkNum, int dimNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	memset(vlcMat, 0, sizeof(double) * dimNum * lmkNum);

	if ( knlOrder == -1 )
	{
		for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
		{
			double *qiVec = lmkMat + lmkiIdx * dimNum;
			double *viVec = vlcMat + lmkiIdx * dimNum;

			for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
			{
				double *qjVec = lmkMat + lmkjIdx * dimNum;
				double *ajVec = alpMat + lmkjIdx * dimNum;

				double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
				double knlVal = exp(-dijVal * dijVal);

				for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
					viVec[dimIdx] += knlVal * ajVec[dimIdx];
			}
		}
	}
	else
	{
		if ( dimNum == 2 )
		{
			switch ( knlOrder )
			{
				case 0:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double f1Val;
							p1Fcn(f1Val, dijVal);

							double knlVal = f1Val;

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double f0Val, f1Val;
							p0Fcn(f0Val, dijVal);
							p1Fcn(f1Val, dijVal);

							double knlVal = 0.5 * (f0Val + 2.0 * f1Val);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double f0Val, f1Val;
							p0Fcn(f0Val, dijVal);
							p1Fcn(f1Val, dijVal);

							double knlVal = (4.0 * f0Val + (8.0 + dijSqu) * f1Val) / 8.0;

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double f0Val, f1Val;
							p0Fcn(f0Val, dijVal);
							p1Fcn(f1Val, dijVal);

							double knlVal = ((24.0 + dijSqu) * f0Val + 8.0 * (6.0 + dijSqu) * f1Val) / 48.0;

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double f0Val, f1Val;
							p0Fcn(f0Val, dijVal);
							p1Fcn(f1Val, dijVal);

							double knlVal = (12.0 * (16.0 + dijSqu) * f0Val + (384.0 + dijSqu * (72.0 + dijSqu)) * f1Val) / 384.0;

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;
			}
		}
		else
		{
			switch ( knlOrder )
			{
				case 0:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (1.0 + dijVal) * exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (3.0 + dijVal * (3.0 + dijVal)) / 3.0 * exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / 15.0 * exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double *qiVec = lmkMat + lmkiIdx * dimNum;
						double *viVec = vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ajVec = alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (105.0 + dijVal * (105.0 + dijVal * (45.0 + dijVal * (10.0 + dijVal)))) / 105.0 * exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;
			}
		}
	}

	return;
}

void multiplyKernel(double *vlcMat, double *lmkiMat, double *lmkjMat, double *alpMat,
                    int knlOrder, double knlWidth, int lmkiNum, int lmkjNum, int dimNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	memset(vlcMat, 0, sizeof(double) * dimNum * lmkiNum);

	if ( knlOrder == -1 )
	{
		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			double *qiVec = lmkiMat + lmkiIdx * dimNum;
			double *viVec =  vlcMat + lmkiIdx * dimNum;

			for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
			{
				double *qjVec = lmkjMat + lmkjIdx * dimNum;
				double *ajVec =  alpMat + lmkjIdx * dimNum;

				double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
				double knlVal = exp(-dijVal * dijVal);

				for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
					viVec[dimIdx] += knlVal * ajVec[dimIdx];
			}
		}
	}
	else
	{
		if ( dimNum == 2 )
		{
			switch ( knlOrder )
			{
				case 0:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double f1Val;
							p1Fcn(f1Val, dijVal);

							double knlVal = f1Val;

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double f0Val, f1Val;
							p0Fcn(f0Val, dijVal);
							p1Fcn(f1Val, dijVal);

							double knlVal = 0.5 * (f0Val + 2.0 * f1Val);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double f0Val, f1Val;
							p0Fcn(f0Val, dijVal);
							p1Fcn(f1Val, dijVal);

							double knlVal = (4.0 * f0Val + (8.0 + dijSqu) * f1Val) / 8.0;

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double f0Val, f1Val;
							p0Fcn(f0Val, dijVal);
							p1Fcn(f1Val, dijVal);

							double knlVal = ((24.0 + dijSqu) * f0Val + 8.0 * (6.0 + dijSqu) * f1Val) / 48.0;

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double f0Val, f1Val;
							p0Fcn(f0Val, dijVal);
							p1Fcn(f1Val, dijVal);

							double knlVal = (12.0 * (16.0 + dijSqu) * f0Val + (384.0 + dijSqu * (72.0 + dijSqu)) * f1Val) / 384.0;

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;
			}
		}
		else
		{
			switch ( knlOrder )
			{
				case 0:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 1:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (1.0 + dijVal) * exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (3.0 + dijVal * (3.0 + dijVal)) / 3.0 * exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) / 15.0 * exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double *qiVec = lmkiMat + lmkiIdx * dimNum;
						double *viVec =  vlcMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double *qjVec = lmkjMat + lmkjIdx * dimNum;
							double *ajVec =  alpMat + lmkjIdx * dimNum;

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double knlVal = (105.0 + dijVal * (105.0 + dijVal * (45.0 + dijVal * (10.0 + dijVal)))) / 105.0 * exp(-dijVal);

							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
								viVec[dimIdx] += knlVal * ajVec[dimIdx];
						}
					}
					break;
			}
		}
	}

	return;
}
