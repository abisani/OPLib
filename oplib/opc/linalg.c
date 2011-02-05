// linalg.cpp --- Part of the project OPLib 1.0, a high performance pricing library
// based on operator methods, higher level BLAS and multicore architectures 

// Author:     2009 Claudio Albanese
// Maintainer: Claudio Albanese <claudio@albanese.co.uk>
// Created:    April-July 2009
// Version:    1.0.0
// Credits:    The CUDA code for SGEMM4, SGEMV4 and SSQMM were inspired by 
//             Vasily Volkov's implementation of SGEMM
//			   We use several variations of the multi-threaded Mersenne Twister algorithm of 
//			   period 2203 due to Makoto Matsumoto.
//             The Monte Carlo routine in SMC includes code by Victor Podlozhnyuk 
//             included in the CUDA SDK.
//             CPU-side BLAS and random number generators link to primitives in the
//			   Intel Math Kernel Libraries. 

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
// Boston, MA 02111-1307, USA.



#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"

#ifdef LINUX
#define __declspec(x)
#endif
 

__declspec( dllexport )  void  opc_sgemv(int m, int n, float alpha, 
													float *a, int lda, float *x, int incx,
													float beta, float *y, int incy) {

		SGEMV("n", &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);

}


__declspec( dllexport )  void  opc_dgemv( int m, int n, double alpha, double *a, int lda, double *x, int incx,
            double beta, double *y, int incy) {

		DGEMV("n", &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);

}



__declspec( dllexport )  int  opc_dgesv(int n, int nrhs, double* ap, int ma, int na, 
												   int lda, int * ipivp, double *bp, int ldb) {

	int info;
	DGESV(&n, &nrhs, ap, &lda, ipivp, bp, &ldb, &info);

	return info;
}


__declspec( dllexport ) int  opc_sgesv(int n, int nrhs, float* ap, int ma, int na, 
												  int lda, int * ipivp, float *bp, int ldb, int *info) {

SGESV(&n, &nrhs, ap, &lda, ipivp, bp, &ldb, info);

return *info;
}


__declspec( dllexport ) void  opc_sgemm(float *a, float *b, float *c, int lda, 
												   int ldb, int ldc, int m, int n, int k) {

float alpha = 1;
float beta = 0;

SGEMM("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

}



__declspec( dllexport ) void  opc_dgemm(double* a, double* b, double* c, 
			    int lda, int ldb, int ldc, int m, int n, int k) {

double alpha = 1;
double beta = 0;

DGEMM("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

}


