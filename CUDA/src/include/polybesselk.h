#ifndef POLYBESSELK_H
#define POLYBESSELK_H

#include <cmath>
#include <cfloat>
#include "besselk.h"

inline __device__ double polyval(const int plyDeg, double *plyVec, double xVal)
{
	double plyVal = plyVec[plyDeg];
	for ( int powIdx = plyDeg - 1; powIdx >= 0; --powIdx )
		plyVal = plyVal * xVal + plyVec[powIdx];

	return plyVal;
}

inline __device__ void p0Fcn(double &p0Val, double xVal)
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

		double P01Val = polyval(P01Deg, c_P01Vec, xSqu);	
		double Q01Val = polyval(Q01Deg, c_Q01Vec, xSqu);	
		double PQ01   = P01Val / Q01Val;

		double P02Val = polyval(P02Deg, c_P02Vec, xSqu);	
		double Q02Val = polyval(Q02Deg, c_Q02Vec, xSqu);	
		double PQ02   = P02Val / Q02Val;

		p0Val = xSqu * (PQ01 - logx * (xSqu * PQ02 + 1.0)); 

		return;
	}

	// 1 < x <= xMax

	double xInv = 1.0 / xVal;

	double P03Val = polyval(P03Deg, c_P03Vec, xInv);	
	double Q03Val = polyval(Q03Deg, c_Q03Vec, xInv);	
	double PQ03   = P03Val / Q03Val;

	p0Val = xVal * sqrt(xVal) * exp(-xVal) * PQ03;

	return;
}

inline __device__ void p1Fcn(double &p1Val, double xVal)
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

		double P11Val = polyval(P11Deg, c_P11Vec, xSqu);	
		double Q11Val = polyval(Q11Deg, c_Q11Vec, xSqu);	
		double PQ11   = P11Val / Q11Val;

		double P12Val = polyval(P12Deg, c_P12Vec, xSqu);	
		double Q12Val = polyval(Q12Deg, c_Q12Vec, xSqu);	
		double PQ12   = P12Val / Q12Val;

		p1Val = PQ11 + xSqu * logx * PQ12;

		return;
	}

	// 1 < x <= xMax

	double xInv = 1.0 / xVal;

	double P13Val = polyval(P13Deg, c_P13Vec, xInv);	
	double Q13Val = polyval(Q13Deg, c_Q13Vec, xInv);	
	double PQ13   = P13Val / Q13Val;

	p1Val = sqrt(xVal) * exp(-xVal) * PQ13;

	return;
}

#endif
