// S1DGrid.cs --- Part of the project OPLib 1.0, a high performance pricing library
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


using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace OPModel.Types
{
    /** 
     **/
    public class S1DGrid : SGrid
    {    

        public S1DGrid(
           CDevice device,
           int nx,
           DateTime today,
           DateTime[] t_i,
           TimeSpan dt0,
           double xval0,
           double[] xpivot_p,
           double[] xgridspacing_p)
            : base(device, nx, 1, today, t_i, dt0, xval0, xpivot_p, xgridspacing_p, null, -1)
        {
        }

        public S1DGrid(S1DGrid grid, bool deep_copy)
            : base(grid, deep_copy)
        {
        }

        override public void mkgrid(ref double[] xval_y, ref double[] rval_y, ref int x0, int r0, ref int y0, double xval0,
            double[] xpivot_p, double[] xgridspacing_p, double[] rval_r)
        {
            xval_y = new double[d];
            rval_y = new double[d];
            for (int y = 0; y < d; y++) rval_y[y] = 0;
            if (d != nx) throw new System.Exception();
            xgrid(ref xval_y, ref x0, xpivot_p, xgridspacing_p, xval0);
            y0 = x0;
        }

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private int opc_dgesv(int n, int nrhs, double* a, int ma,
                                           int na, int lda, int* ipiv, double* b, int ldb);

        override public double[] pregen()
        {

            double[] mat = new double[2 * 2];
            double[] invm = new double[2 * 2 * d];
            for (int x = 0; x < d; x++)
            {
                for (int z = 0; z < 2; z++)
                {
                    invm[4 * x + z + 2 * z] = 1;
                }
            }
            int ld = 2;
            int[] ipiv = new int[20];
            for (int x0 = 0; x0 < d; x0++)
            {

                if (x0 > 0)
                {
                    mat[0 + ld * 0] = host_d_xval(x0 - 1) - host_d_xval(x0);
                }
                else
                {
                    mat[0 + ld * 0] = 1;
                }
                if (x0 < d - 1)
                {
                    mat[0 + ld * 1] = host_d_xval(x0 + 1) - host_d_xval(x0);
                }
                else
                {
                    mat[0 + ld * 1] = 1;
                }
                if (x0 > 0 && x0 < d)
                {
                    mat[1 + ld * 0] = Math.Pow(mat[0 + ld * 0], 2);
                    mat[1 + ld * 1] = Math.Pow(mat[0 + ld * 1], 2);
                }
                else
                {
                    if (x0 == 0)
                    {
                        mat[1 + ld * 0] = 0;
                        mat[1 + ld * 1] = 1;                     
                    }
                    if (x0 == d - 1)
                    {
                        mat[1 + ld * 0] = 1;
                        mat[1 + ld * 1] = 0;                       
                    }
                }

                unsafe
                {
                    fixed (double* ap = &mat[0])
                    {
                        fixed (int* ipivp = &ipiv[0])
                        {
                            fixed (double* bp = &invm[0])
                            {
                                int status;
                                status = opc_dgesv(2, 2, ap, 2, 2, 2, ipivp, bp + 4 * x0, 2);
                                if (status != 0) throw new System.Exception();
                            }
                        }
                    }
                }
            }
            return invm;
        }

    }
}
