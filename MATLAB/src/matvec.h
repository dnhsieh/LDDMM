// Date: 07/10/2020

#ifndef MATVEC_H
#define MATVEC_H

#include <cmath>
#include "constants.h"

#ifdef DIM2

inline void vectorAverage(double *avgVec, double *v1Vec, double *v2Vec)
{
	avgVec[0] = 0.5 * (v1Vec[0] + v2Vec[0]);
	avgVec[1] = 0.5 * (v1Vec[1] + v2Vec[1]);

	return;
}

inline void vectorSubtract(double *v12Vec, double *v1Vec, double *v2Vec)
{
	v12Vec[0] = v1Vec[0] - v2Vec[0];
	v12Vec[1] = v1Vec[1] - v2Vec[1];

	return;
}

inline double eucnorm(double *vec)
{
	return sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
}

inline double eucdistSqu(double *xVec, double *yVec)
{
	return  (xVec[0] - yVec[0]) * (xVec[0] - yVec[0])
	      + (xVec[1] - yVec[1]) * (xVec[1] - yVec[1]);
}

inline double dotProduct(double *xVec, double *yVec)
{
	return xVec[0] * yVec[0] + xVec[1] * yVec[1];
}

#elif DIM3

inline void vectorAverage(double *avgVec, double *v1Vec, double *v2Vec, double *v3Vec)
{
	avgVec[0] = (v1Vec[0] + v2Vec[0] + v3Vec[0]) / 3.0;
	avgVec[1] = (v1Vec[1] + v2Vec[1] + v3Vec[1]) / 3.0;
	avgVec[2] = (v1Vec[2] + v2Vec[2] + v3Vec[2]) / 3.0;

	return;
}

inline void vectorSubtract(double *v12Vec, double *v1Vec, double *v2Vec)
{
	v12Vec[0] = v1Vec[0] - v2Vec[0];
	v12Vec[1] = v1Vec[1] - v2Vec[1];
	v12Vec[2] = v1Vec[2] - v2Vec[2];

	return;
}

inline double eucnorm(double *vec)
{
	return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

inline double eucdistSqu(double *xVec, double *yVec)
{
	return  (xVec[0] - yVec[0]) * (xVec[0] - yVec[0])
	      + (xVec[1] - yVec[1]) * (xVec[1] - yVec[1])
	      + (xVec[2] - yVec[2]) * (xVec[2] - yVec[2]);
}

inline double dotProduct(double *xVec, double *yVec)
{
	return xVec[0] * yVec[0] + xVec[1] * yVec[1] + xVec[2] * yVec[2];
}

inline void crossProduct(double *xyVec, double *xVec, double *yVec)
{
	xyVec[0] =  xVec[1] * yVec[2] - xVec[2] * yVec[1];
	xyVec[1] = -xVec[0] * yVec[2] + xVec[2] * yVec[0];
	xyVec[2] =  xVec[0] * yVec[1] - xVec[1] * yVec[0];

	return;
}

#endif

#endif
