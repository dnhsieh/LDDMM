#include <cmath>
#include <cfloat>
#include "besselk.h"

double polyval(const int plyDeg, const double *plyVec, double xVal)
{
	double plyVal = plyVec[plyDeg];
	for ( int powIdx = plyDeg - 1; powIdx >= 0; --powIdx )
		plyVal = plyVal * xVal + plyVec[powIdx];

	return plyVal;
}

void p0Fcn(double &p0Val, double xVal)
{
	if ( xVal < DBL_EPSILON || xVal > xMax )
	{
		p0Val = 0.0;
		return;
	}

	if ( xVal <= 1.0 )
	{
		double xSqu = xVal * xVal;
		double logx = log(xVal);

		double P01Val = polyval(P01Deg, P01Vec, xSqu);	
		double Q01Val = polyval(Q01Deg, Q01Vec, xSqu);	
		double PQ01   = P01Val / Q01Val;

		double P02Val = polyval(P02Deg, P02Vec, xSqu);	
		double Q02Val = polyval(Q02Deg, Q02Vec, xSqu);	
		double PQ02   = P02Val / Q02Val;

		p0Val = xSqu * (PQ01 - logx * (xSqu * PQ02 + 1.0)); 

		return;
	}

	// 1 < x <= xMax

	double xInv = 1.0 / xVal;

	double P03Val = polyval(P03Deg, P03Vec, xInv);	
	double Q03Val = polyval(Q03Deg, Q03Vec, xInv);	
	double PQ03   = P03Val / Q03Val;

	p0Val = xVal * sqrt(xVal) * exp(-xVal) * PQ03;

	return;
}

void p1Fcn(double &p1Val, double xVal)
{
	if ( xVal < DBL_EPSILON )
	{
		p1Val = 1.0;
		return;
	}

	if ( xVal > xMax )
	{
		p1Val = 0.0;
		return;
	}

	if ( xVal <= 1.0 )
	{
		double xSqu = xVal * xVal;
		double logx = log(xVal);

		double P11Val = polyval(P11Deg, P11Vec, xSqu);	
		double Q11Val = polyval(Q11Deg, Q11Vec, xSqu);	
		double PQ11   = P11Val / Q11Val;

		double P12Val = polyval(P12Deg, P12Vec, xSqu);	
		double Q12Val = polyval(Q12Deg, Q12Vec, xSqu);	
		double PQ12   = P12Val / Q12Val;

		p1Val = PQ11 + xSqu * logx * PQ12;

		return;
	}

	// 1 < x <= xMax

	double xInv = 1.0 / xVal;

	double P13Val = polyval(P13Deg, P13Vec, xInv);	
	double Q13Val = polyval(Q13Deg, Q13Vec, xInv);	
	double PQ13   = P13Val / Q13Val;

	p1Val = sqrt(xVal) * exp(-xVal) * PQ13;

	return;
}
