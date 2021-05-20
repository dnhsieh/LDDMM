// Date: 07/10/2020

#ifndef MATVEC_H
#define MATVEC_H

#include <cmath>
#include "constants.h"

#ifdef DIM2

typedef double2 vector;

struct matrix
{
	double2 x;
	double2 y;
};

inline __device__ void getVector(vector &vec, double *d_arr, int idx, int num)
{
	vec.x = d_arr[      idx];
	vec.y = d_arr[num + idx];

	return;
}

inline __device__ void setVector(double *d_arr, vector vec, int idx, int num)
{
	d_arr[      idx] = vec.x;
   d_arr[num + idx] = vec.y;

	return;
}

inline __device__ void getMatrix(matrix &mat, double *d_arr, int idx, int num)
{
	mat.x.x = d_arr[          idx];
	mat.x.y = d_arr[    num + idx];
	mat.y.x = d_arr[2 * num + idx];
	mat.y.y = d_arr[3 * num + idx];

	return;
}

inline __device__ void setMatrix(double *d_arr, matrix mat, int idx, int num)
{
	d_arr[          idx] = mat.x.x;
   d_arr[    num + idx] = mat.x.y;
   d_arr[2 * num + idx] = mat.y.x;
   d_arr[3 * num + idx] = mat.y.y;

	return;
}

inline __device__ void getElement(vector &v0Vec, vector &v1Vec, vector &v2Vec, double *d_arr, int idx, int num)
{
	v0Vec.x = d_arr[          idx];
	v0Vec.y = d_arr[    num + idx];

	v1Vec.x = d_arr[2 * num + idx];
	v1Vec.y = d_arr[3 * num + idx];

	v2Vec.x = d_arr[4 * num + idx];
	v2Vec.y = d_arr[5 * num + idx];

	return;
}

inline __device__ void setElement(double *d_arr, vector v0Vec, vector v1Vec, vector v2Vec, int idx, int num)
{
	d_arr[          idx] = v0Vec.x;
   d_arr[    num + idx] = v0Vec.y;
                                   
   d_arr[2 * num + idx] = v1Vec.x;
   d_arr[3 * num + idx] = v1Vec.y;
                                   
   d_arr[4 * num + idx] = v2Vec.x;
   d_arr[5 * num + idx] = v2Vec.y;

	return;
}

inline __device__ void getEdge(vector &v10Vec, vector &v20Vec, double *d_arr, int idx, int num)
{
	v10Vec.x = d_arr[          idx];
	v10Vec.y = d_arr[    num + idx];

	v20Vec.x = d_arr[2 * num + idx];
	v20Vec.y = d_arr[3 * num + idx];

	return;
}

inline __device__ void setEdge(double *d_arr, vector v10Vec, vector v20Vec, int idx, int num)
{
	d_arr[          idx] = v10Vec.x;
   d_arr[    num + idx] = v10Vec.y;
                                   
   d_arr[2 * num + idx] = v20Vec.x;
   d_arr[3 * num + idx] = v20Vec.y;
                                   
	return;
}

inline __device__ void getBoundary(vector &v0Vec, vector &v1Vec, double *d_arr, int idx, int num)
{
	v0Vec.x = d_arr[          idx];
	v0Vec.y = d_arr[    num + idx];

	v1Vec.x = d_arr[2 * num + idx];
	v1Vec.y = d_arr[3 * num + idx];

	return;
}

inline __device__ void setBoundary(double *d_arr, vector v0Vec, vector v1Vec, int idx, int num)
{
	d_arr[          idx] = v0Vec.x;
	d_arr[    num + idx] = v0Vec.y;
                          
	d_arr[2 * num + idx] = v1Vec.x;
	d_arr[3 * num + idx] = v1Vec.y;

	return;
}

inline __device__ void vectorSum(vector &v12Vec, vector v1Vec, vector v2Vec)
{
	v12Vec.x = v1Vec.x + v2Vec.x;
	v12Vec.y = v1Vec.y + v2Vec.y;

	return;
}

inline __device__ void vectorSubtract(vector &v12Vec, vector v1Vec, vector v2Vec)
{
	v12Vec.x = v1Vec.x - v2Vec.x;
	v12Vec.y = v1Vec.y - v2Vec.y;

	return;
}

inline __device__ void vectorAverage(vector &avgVec, vector v1Vec, vector v2Vec)
{
	avgVec.x = 0.5 * (v1Vec.x + v2Vec.x);
	avgVec.y = 0.5 * (v1Vec.y + v2Vec.y);

	return;
}

inline __device__ void vectorAverage(vector &avgVec, vector v1Vec, vector v2Vec, vector v3Vec)
{
	avgVec.x = (v1Vec.x + v2Vec.x + v3Vec.x) / 3.0;
	avgVec.y = (v1Vec.y + v2Vec.y + v3Vec.y) / 3.0;

	return;
}

inline __device__ double dotProduct(vector xVec, vector yVec)
{
	return xVec.x * yVec.x + xVec.y * yVec.y;
}

inline __device__ double eucnorm(vector vec)
{
	return sqrt(vec.x * vec.x + vec.y * vec.y);
}

inline __device__ double eucdist(vector xVec, vector yVec)
{
	return sqrt(  (xVec.x - yVec.x) * (xVec.x - yVec.x)
	            + (xVec.y - yVec.y) * (xVec.y - yVec.y) );
}

inline __device__ double eucnormSqu(vector vec)
{
	return vec.x * vec.x + vec.y * vec.y;
}

inline __device__ double eucdistSqu(vector xVec, vector yVec)
{
	return  (xVec.x - yVec.x) * (xVec.x - yVec.x)
	      + (xVec.y - yVec.y) * (xVec.y - yVec.y);
}

inline __device__ void matSum(matrix &ABMat, matrix AMat, matrix BMat)
{
	ABMat.x.x = AMat.x.x + BMat.x.x;
	ABMat.x.y = AMat.x.y + BMat.x.y;
	ABMat.y.x = AMat.y.x + BMat.y.x;
	ABMat.y.y = AMat.y.y + BMat.y.y;

	return;
}

inline __device__ void columns2Mat(matrix &mat, vector xVec, vector yVec)
{
	mat.x.x = xVec.x; mat.x.y = yVec.x;
	mat.y.x = xVec.y; mat.y.y = yVec.y;

	return;
}

inline __device__ double trace(matrix AMat)
{
	return AMat.x.x + AMat.y.y;
}

inline __device__ double matDotProduct(matrix AMat, matrix BMat)
{
	return  AMat.x.x * BMat.x.x + AMat.x.y * BMat.x.y
         + AMat.y.x * BMat.y.x + AMat.y.y * BMat.y.y;
}

inline __device__ void matVecMul(vector &AbVec, matrix AMat, vector bVec)
{
	AbVec.x = AMat.x.x * bVec.x + AMat.x.y * bVec.y;
	AbVec.y = AMat.y.x * bVec.x + AMat.y.y * bVec.y;

	return;
}

inline __device__ void matTVecMul(vector &AtbVec, matrix AMat, vector bVec)
{
	AtbVec.x = AMat.x.x * bVec.x + AMat.y.x * bVec.y;
	AtbVec.y = AMat.x.y * bVec.x + AMat.y.y * bVec.y;

	return;
}

inline __device__ double vecMatVecMul(vector aVec, matrix AMat, vector bVec)
{
	return  aVec.x * (AMat.x.x * bVec.x + AMat.x.y * bVec.y)
	      + aVec.y * (AMat.y.x * bVec.x + AMat.y.y * bVec.y);
}

inline __device__ double det(matrix mat)
{
	return mat.x.x * mat.y.y - mat.y.x * mat.x.y;
}

inline __device__ double det(vector xVec, vector yVec)
{
	return xVec.x * yVec.y - xVec.y * yVec.x;
}

inline __device__ void matInv(matrix &inv, matrix mat)
{
	double detVal = mat.x.x * mat.y.y - mat.y.x * mat.x.y;

	inv.x.x =  mat.y.y / detVal;
	inv.x.y = -mat.x.y / detVal;
	inv.y.x = -mat.y.x / detVal;
	inv.y.y =  mat.x.x / detVal;

	return;
}

inline __device__ void matInv(matrix &inv, vector xVec, vector yVec)
{
	double detVal = xVec.x * yVec.y - xVec.y * yVec.x;

	inv.x.x =  yVec.y / detVal;
	inv.x.y = -yVec.x / detVal;
	inv.y.x = -xVec.y / detVal;
	inv.y.y =  xVec.x / detVal;

	return;
}

inline __device__ void matMatMul(matrix &ABMat, matrix AMat, matrix BMat)
{
	ABMat.x.x = AMat.x.x * BMat.x.x + AMat.x.y * BMat.y.x;
	ABMat.x.y = AMat.x.x * BMat.x.y + AMat.x.y * BMat.y.y;

	ABMat.y.x = AMat.y.x * BMat.x.x + AMat.y.y * BMat.y.x;
	ABMat.y.y = AMat.y.x * BMat.x.y + AMat.y.y * BMat.y.y;

	return;
}

#elif DIM3

typedef double3 vector;

struct matrix
{
	double3 x;
	double3 y;
	double3 z;
};

inline __device__ void getVector(vector &vec, double *d_arr, int idx, int num)
{
	vec.x = d_arr[          idx];
	vec.y = d_arr[    num + idx];
	vec.z = d_arr[2 * num + idx];

	return;
}

inline __device__ void setVector(double *d_arr, vector vec, int idx, int num)
{
	d_arr[          idx] = vec.x;
   d_arr[    num + idx] = vec.y;
   d_arr[2 * num + idx] = vec.z;

	return;
}

inline __device__ void getMatrix(matrix &mat, double *d_arr, int idx, int num)
{
	mat.x.x = d_arr[          idx];
	mat.x.y = d_arr[    num + idx];
	mat.x.z = d_arr[2 * num + idx];

	mat.y.x = d_arr[3 * num + idx];
	mat.y.y = d_arr[4 * num + idx];
	mat.y.z = d_arr[5 * num + idx];

	mat.z.x = d_arr[6 * num + idx];
	mat.z.y = d_arr[7 * num + idx];
	mat.z.z = d_arr[8 * num + idx];

	return;
}

inline __device__ void setMatrix(double *d_arr, matrix mat, int idx, int num)
{
	d_arr[          idx] = mat.x.x;
   d_arr[    num + idx] = mat.x.y;
   d_arr[2 * num + idx] = mat.x.z;
                                  
   d_arr[3 * num + idx] = mat.y.x;
   d_arr[4 * num + idx] = mat.y.y;
   d_arr[5 * num + idx] = mat.y.z;
                                  
   d_arr[6 * num + idx] = mat.z.x;
   d_arr[7 * num + idx] = mat.z.y;
   d_arr[8 * num + idx] = mat.z.z;

	return;
}

inline __device__ void getElement(vector &v0Vec, vector &v1Vec, vector &v2Vec, vector &v3Vec,
                                  double *d_arr, int idx, int num)
{
	v0Vec.x = d_arr[           idx];
	v0Vec.y = d_arr[     num + idx];
	v0Vec.z = d_arr[ 2 * num + idx];

	v1Vec.x = d_arr[ 3 * num + idx];
	v1Vec.y = d_arr[ 4 * num + idx];
	v1Vec.z = d_arr[ 5 * num + idx];

	v2Vec.x = d_arr[ 6 * num + idx];
	v2Vec.y = d_arr[ 7 * num + idx];
	v2Vec.z = d_arr[ 8 * num + idx];

	v3Vec.x = d_arr[ 9 * num + idx];
	v3Vec.y = d_arr[10 * num + idx];
	v3Vec.z = d_arr[11 * num + idx];

	return;
}

inline __device__ void setElement(double *d_arr, vector v0Vec, vector v1Vec, vector v2Vec, vector v3Vec,
                                  int idx, int num)
{
	d_arr[           idx] = v0Vec.x;
   d_arr[     num + idx] = v0Vec.y;
   d_arr[ 2 * num + idx] = v0Vec.z;
                                   
   d_arr[ 3 * num + idx] = v1Vec.x;
   d_arr[ 4 * num + idx] = v1Vec.y;
   d_arr[ 5 * num + idx] = v1Vec.z;
                                   
   d_arr[ 6 * num + idx] = v2Vec.x;
   d_arr[ 7 * num + idx] = v2Vec.y;
   d_arr[ 8 * num + idx] = v2Vec.z;
                                   
   d_arr[ 9 * num + idx] = v3Vec.x;
   d_arr[10 * num + idx] = v3Vec.y;
   d_arr[11 * num + idx] = v3Vec.z;

	return;
}

inline __device__ void getEdge(vector &v10Vec, vector &v20Vec, vector &v30Vec,
                               double *d_arr, int idx, int num)
{
	v10Vec.x = d_arr[          idx];
	v10Vec.y = d_arr[    num + idx];
	v10Vec.z = d_arr[2 * num + idx];

	v20Vec.x = d_arr[3 * num + idx];
	v20Vec.y = d_arr[4 * num + idx];
	v20Vec.z = d_arr[5 * num + idx];

	v30Vec.x = d_arr[6 * num + idx];
	v30Vec.y = d_arr[7 * num + idx];
	v30Vec.z = d_arr[8 * num + idx];

	return;
}

inline __device__ void setEdge(double *d_arr, vector v10Vec, vector v20Vec, vector v30Vec,
                               int idx, int num)
{
	d_arr[          idx] = v10Vec.x;
   d_arr[    num + idx] = v10Vec.y;
   d_arr[2 * num + idx] = v10Vec.z;
                                  
   d_arr[3 * num + idx] = v20Vec.x;
   d_arr[4 * num + idx] = v20Vec.y;
   d_arr[5 * num + idx] = v20Vec.z;
                                  
   d_arr[6 * num + idx] = v30Vec.x;
   d_arr[7 * num + idx] = v30Vec.y;
   d_arr[8 * num + idx] = v30Vec.z;

	return;
}

inline __device__ void getBoundary(vector &v0Vec, vector &v1Vec, vector &v2Vec, double *d_arr, int idx, int num)
{
	v0Vec.x = d_arr[          idx];
	v0Vec.y = d_arr[    num + idx];
	v0Vec.z = d_arr[2 * num + idx];

	v1Vec.x = d_arr[3 * num + idx];
	v1Vec.y = d_arr[4 * num + idx];
	v1Vec.z = d_arr[5 * num + idx];

	v2Vec.x = d_arr[6 * num + idx];
	v2Vec.y = d_arr[7 * num + idx];
	v2Vec.z = d_arr[8 * num + idx];

	return;
}

inline __device__ void setBoundary(double *d_arr, vector v0Vec, vector v1Vec, vector v2Vec, int idx, int num)
{
	d_arr[          idx] = v0Vec.x;
	d_arr[    num + idx] = v0Vec.y;
	d_arr[2 * num + idx] = v0Vec.z;
                          
	d_arr[3 * num + idx] = v1Vec.x;
	d_arr[4 * num + idx] = v1Vec.y;
	d_arr[5 * num + idx] = v1Vec.z;
                          
	d_arr[6 * num + idx] = v2Vec.x;
	d_arr[7 * num + idx] = v2Vec.y;
	d_arr[8 * num + idx] = v2Vec.z;

	return;
}

inline __device__ void vectorSum(vector &v12Vec, vector v1Vec, vector v2Vec)
{
	v12Vec.x = v1Vec.x + v2Vec.x;
	v12Vec.y = v1Vec.y + v2Vec.y;
	v12Vec.z = v1Vec.z + v2Vec.z;

	return;
}

inline __device__ void vectorSubtract(vector &v12Vec, vector v1Vec, vector v2Vec)
{
	v12Vec.x = v1Vec.x - v2Vec.x;
	v12Vec.y = v1Vec.y - v2Vec.y;
	v12Vec.z = v1Vec.z - v2Vec.z;

	return;
}

inline __device__ void vectorAverage(vector &avgVec, vector v1Vec, vector v2Vec, vector v3Vec)
{
	avgVec.x = (v1Vec.x + v2Vec.x + v3Vec.x) / 3.0;
	avgVec.y = (v1Vec.y + v2Vec.y + v3Vec.y) / 3.0;
	avgVec.z = (v1Vec.z + v2Vec.z + v3Vec.z) / 3.0;

	return;
}

inline __device__ void vectorAverage(vector &avgVec, vector v1Vec, vector v2Vec, vector v3Vec, vector v4Vec)
{
	avgVec.x = 0.25 * (v1Vec.x + v2Vec.x + v3Vec.x + v4Vec.x);
	avgVec.y = 0.25 * (v1Vec.y + v2Vec.y + v3Vec.y + v4Vec.y);
	avgVec.z = 0.25 * (v1Vec.z + v2Vec.z + v3Vec.z + v4Vec.z);

	return;
}

inline __device__ double dotProduct(vector xVec, vector yVec)
{
	return xVec.x * yVec.x + xVec.y * yVec.y + xVec.z * yVec.z;
}

inline __device__ void crossProduct(vector &xyVec, vector xVec, vector yVec)
{
	xyVec.x =  xVec.y * yVec.z - xVec.z * yVec.y;
	xyVec.y = -xVec.x * yVec.z + xVec.z * yVec.x;
	xyVec.z =  xVec.x * yVec.y - xVec.y * yVec.x;

	return;
}

inline __device__ double eucnorm(vector vec)
{
	return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

inline __device__ double eucdist(vector xVec, vector yVec)
{
	return sqrt(  (xVec.x - yVec.x) * (xVec.x - yVec.x)
	            + (xVec.y - yVec.y) * (xVec.y - yVec.y)
	            + (xVec.z - yVec.z) * (xVec.z - yVec.z) );
}

inline __device__ double eucnormSqu(vector vec)
{
	return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

inline __device__ double eucdistSqu(vector xVec, vector yVec)
{
	return   (xVec.x - yVec.x) * (xVec.x - yVec.x)
	       + (xVec.y - yVec.y) * (xVec.y - yVec.y)
	       + (xVec.z - yVec.z) * (xVec.z - yVec.z);
}

inline __device__ void matSum(matrix &ABMat, matrix AMat, matrix BMat)
{
	ABMat.x.x = AMat.x.x + BMat.x.x;
	ABMat.x.y = AMat.x.y + BMat.x.y;
	ABMat.x.z = AMat.x.z + BMat.x.z;

	ABMat.y.x = AMat.y.x + BMat.y.x;
	ABMat.y.y = AMat.y.y + BMat.y.y;
	ABMat.y.z = AMat.y.z + BMat.y.z;
	
	ABMat.z.x = AMat.z.x + BMat.z.x;
	ABMat.z.y = AMat.z.y + BMat.z.y;
	ABMat.z.z = AMat.z.z + BMat.z.z;

	return;
}

inline __device__ void columns2Mat(matrix &mat, vector xVec, vector yVec, vector zVec)
{
	mat.x.x = xVec.x; mat.x.y = yVec.x; mat.x.z = zVec.x;
	mat.y.x = xVec.y; mat.y.y = yVec.y; mat.y.z = zVec.y;
	mat.z.x = xVec.z; mat.z.y = yVec.z; mat.z.z = zVec.z;

	return;
}

inline __device__ double trace(matrix AMat)
{
	return AMat.x.x + AMat.y.y + AMat.z.z;
}

inline __device__ double matDotProduct(matrix AMat, matrix BMat)
{
	return  AMat.x.x * BMat.x.x + AMat.x.y * BMat.x.y + AMat.x.z * BMat.x.z
	      + AMat.y.x * BMat.y.x + AMat.y.y * BMat.y.y + AMat.y.z * BMat.y.z
	      + AMat.z.x * BMat.z.x + AMat.z.y * BMat.z.y + AMat.z.z * BMat.z.z;
}

inline __device__ void matVecMul(vector &AbVec, matrix AMat, vector bVec)
{
	AbVec.x = AMat.x.x * bVec.x + AMat.x.y * bVec.y + AMat.x.z * bVec.z;
	AbVec.y = AMat.y.x * bVec.x + AMat.y.y * bVec.y + AMat.y.z * bVec.z;
	AbVec.z = AMat.z.x * bVec.x + AMat.z.y * bVec.y + AMat.z.z * bVec.z;

	return;
}

inline __device__ void matTVecMul(vector &AtbVec, matrix AMat, vector bVec)
{
	AtbVec.x = AMat.x.x * bVec.x + AMat.y.x * bVec.y + AMat.z.x * bVec.z;
	AtbVec.y = AMat.x.y * bVec.x + AMat.y.y * bVec.y + AMat.z.y * bVec.z;
	AtbVec.z = AMat.x.z * bVec.x + AMat.y.z * bVec.y + AMat.z.z * bVec.z;

	return;
}

inline __device__ double vecMatVecMul(vector aVec, matrix AMat, vector bVec)
{
	return  aVec.x * (AMat.x.x * bVec.x + AMat.x.y * bVec.y + AMat.x.z * bVec.z)
	      + aVec.y * (AMat.y.x * bVec.x + AMat.y.y * bVec.y + AMat.y.z * bVec.z)
	      + aVec.z * (AMat.z.x * bVec.x + AMat.z.y * bVec.y + AMat.z.z * bVec.z);
}

inline __device__ double det(matrix mat)
{
	return  mat.x.x * mat.y.y * mat.z.z
	      + mat.y.x * mat.z.y * mat.x.z
	      + mat.z.x * mat.x.y * mat.y.z
	      - mat.x.x * mat.z.y * mat.y.z
	      - mat.y.x * mat.x.y * mat.z.z
	      - mat.z.x * mat.y.y * mat.x.z;
}

inline __device__ double det(vector xVec, vector yVec, vector zVec)
{
	return  xVec.x * yVec.y * zVec.z
	      + xVec.y * yVec.z * zVec.x
	      + xVec.z * yVec.x * zVec.y
	      - xVec.x * yVec.z * zVec.y
	      - xVec.y * yVec.x * zVec.z
	      - xVec.z * yVec.y * zVec.x;
}

inline __device__ void matInv(matrix &inv, matrix mat)
{
	double detVal =   mat.x.x * mat.y.y * mat.z.z
	                + mat.y.x * mat.z.y * mat.x.z
	                + mat.z.x * mat.x.y * mat.y.z
	                - mat.x.x * mat.z.y * mat.y.z
	                - mat.y.x * mat.x.y * mat.z.z
	                - mat.z.x * mat.y.y * mat.x.z;

	inv.x.x =  (mat.y.y * mat.z.z - mat.z.y * mat.y.z) / detVal;
	inv.x.y = -(mat.x.y * mat.z.z - mat.z.y * mat.x.z) / detVal;
	inv.x.z =  (mat.x.y * mat.y.z - mat.y.y * mat.x.z) / detVal;

	inv.y.x = -(mat.y.x * mat.z.z - mat.z.x * mat.y.z) / detVal;
	inv.y.y =  (mat.x.x * mat.z.z - mat.z.x * mat.x.z) / detVal;
	inv.y.z = -(mat.x.x * mat.y.z - mat.y.x * mat.x.z) / detVal;
	
	inv.z.x =  (mat.y.x * mat.z.y - mat.z.x * mat.y.y) / detVal;
	inv.z.y = -(mat.x.x * mat.z.y - mat.z.x * mat.x.y) / detVal;
	inv.z.z =  (mat.x.x * mat.y.y - mat.y.x * mat.x.y) / detVal;

	return;
}

inline __device__ void matInv(matrix &inv, vector xVec, vector yVec, vector zVec)
{
	double detVal =   xVec.x * yVec.y * zVec.z
	                + xVec.y * yVec.z * zVec.x
	                + xVec.z * yVec.x * zVec.y
	                - xVec.x * yVec.z * zVec.y
	                - xVec.y * yVec.x * zVec.z
	                - xVec.z * yVec.y * zVec.x;

	inv.x.x =  (yVec.y * zVec.z - yVec.z * zVec.y) / detVal;
	inv.x.y = -(yVec.x * zVec.z - yVec.z * zVec.x) / detVal;
	inv.x.z =  (yVec.x * zVec.y - yVec.y * zVec.x) / detVal;

	inv.y.x = -(xVec.y * zVec.z - xVec.z * zVec.y) / detVal;
	inv.y.y =  (xVec.x * zVec.z - xVec.z * zVec.x) / detVal;
	inv.y.z = -(xVec.x * zVec.y - xVec.y * zVec.x) / detVal;

	inv.z.x =  (xVec.y * yVec.z - xVec.z * yVec.y) / detVal;
	inv.z.y = -(xVec.x * yVec.z - xVec.z * yVec.x) / detVal;
	inv.z.z =  (xVec.x * yVec.y - xVec.y * yVec.x) / detVal;

	return;
}

inline __device__ void matMatMul(matrix &ABMat, matrix AMat, matrix BMat)
{
	ABMat.x.x = AMat.x.x * BMat.x.x + AMat.x.y * BMat.y.x + AMat.x.z * BMat.z.x;
	ABMat.x.y = AMat.x.x * BMat.x.y + AMat.x.y * BMat.y.y + AMat.x.z * BMat.z.y;
	ABMat.x.z = AMat.x.x * BMat.x.z + AMat.x.y * BMat.y.z + AMat.x.z * BMat.z.z;

	ABMat.y.x = AMat.y.x * BMat.x.x + AMat.y.y * BMat.y.x + AMat.y.z * BMat.z.x;
	ABMat.y.y = AMat.y.x * BMat.x.y + AMat.y.y * BMat.y.y + AMat.y.z * BMat.z.y;
	ABMat.y.z = AMat.y.x * BMat.x.z + AMat.y.y * BMat.y.z + AMat.y.z * BMat.z.z;

	ABMat.z.x = AMat.z.x * BMat.x.x + AMat.z.y * BMat.y.x + AMat.z.z * BMat.z.x;
	ABMat.z.y = AMat.z.x * BMat.x.y + AMat.z.y * BMat.y.y + AMat.z.z * BMat.z.y;
	ABMat.z.z = AMat.z.x * BMat.x.z + AMat.z.y * BMat.y.z + AMat.z.z * BMat.z.z;

	return;
}

#endif

#endif
