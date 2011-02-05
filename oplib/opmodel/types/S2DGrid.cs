// S2DGrid.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
    public class S2DGrid : SGrid
    {      

        public S2DGrid(
            CDevice device,
            int nx,
            int nr,
            DateTime today,
            DateTime[] t_i,
            TimeSpan dt0,
            double xval0,
            double[] xpivot_p,
            double[] xgridspacing_p,
            double[] rval_r,
            int r0)
            :
            base(device, nx, nr, today, t_i, dt0, xval0, xpivot_p, xgridspacing_p, rval_r, r0)
        {
        }

        public S2DGrid(S2DGrid grid, bool deep_copy)
            : base(grid, deep_copy)
        {
        }

        override public void mkgrid(ref double[] xval_y, ref double[] rval_y, ref int x0, int r0, ref int y0, double xval0,
            double[] xpivot_p, double[] xgridspacing_p, double[] rval_r)
        {
            xval_y = new double[d];
            rval_y = new double[d];
            for (int r = 0; r < nr; r++)
            {
                for (int x = 0; x < nx; x++)
                {
                    rval_y[x + nx * r] = rval_r[r];
                }
            }

            if (d != nx * nr) throw new System.Exception();
            xgrid(ref xval_y, ref x0, xpivot_p, xgridspacing_p, xval0);
            for (int r = 1; r < nr; r++)
            {
                for (int x = 0; x < nx; x++)
                {
                    xval_y[x + nx * r] = xval_y[x];
                }
            }
            y0 = x0 + nx * r0;
        }


        public double iterate(double[] xval_y, double[] xpivot_p, double[] xgridspacing_p, double scale)
        {
            xval_y[0] = xpivot_p[0];
            double nextp = xpivot_p[1];
            int p; p = 0;
            int np = xpivot_p.Length;
            for (int x = 1; x < nx; x++)
            {
                xval_y[x] = xval_y[x - 1] + scale * xgridspacing_p[p];
                if (xval_y[x] > nextp)
                {
                    if (p < np - 2)
                    {
                        p += 1;
                        nextp = xpivot_p[1];
                    }
                    if (p == np - 2)
                    {
                        p += 1;
                    }
                }
            }
            return xval_y[d - 1];
        }


        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private int opc_dgesv(int n, int nrhs, double* a, int ma, int na, int lda, int* ipiv, double* b, int ldb);


        override public double[] pregen()
        {
            double[] mat = new double[4 * 4];
            double[] invm = new double[4 * 4 * d];
            for (int y = 0; y < d; y++)
            {
                for (int i = 0; i < 4; i++)
                {
                    invm[16 * y + i + 4 * i] = 1;
                }
            }

            int ld = 4;
            int[] ipiv = new int[4];
            int a0;
            for (int x0 = 0; x0 < nx; x0++)
            {
                for (a0 = 0; a0 < nr; a0++)
                {
                    int y0 = x0 + nx * a0;
                    if (x0 > 0)
                    {
                        mat[0 + ld * 0] = host_d_xval(x0 - 1 + nx * a0) - host_d_xval(x0 + nx * a0);
                    }
                    else
                    {
                        mat[0 + ld * 0] = 0;
                    }
                    if (x0 < nx - 1)
                    {
                        mat[0 + ld * 1] = host_d_xval(x0 + 1 + nx * a0) - host_d_xval(x0 + nx * a0);
                    }
                    else
                    {
                        mat[0 + ld * 1] = 0;
                    }
                    mat[0 + ld * 2] = 0;
                    mat[0 + ld * 3] = 0;                 

                    if (x0 > 0 && x0 < nx - 1)
                    {
                        mat[1 + ld * 0] = Math.Pow(mat[0 + ld * 0], 2);
                        mat[1 + ld * 1] = Math.Pow(mat[0 + ld * 1], 2);
                    }
                    else
                    {
                        if (x0 == 0)
                        {
                            mat[1 + ld * 0] = 1;
                            mat[1 + ld * 1] = 0;                           
                        }
                        if (x0 == nx - 1)
                        {
                            mat[1 + ld * 0] = 0;
                            mat[1 + ld * 1] = 1;                           
                        }
                    }
                    mat[0 + ld * 2] = 0;
                    mat[0 + ld * 3] = 0;
                   
                    mat[2 + ld * 0] = 0;
                    mat[2 + ld * 1] = 0;
                    if (a0 > 0)
                    {
                        mat[2 + ld * 2] = host_d_rval(x0 + nx * (a0 - 1)) - host_d_rval(x0 + nx * a0);
                    }
                    else
                    {
                        mat[2 + ld * 2] = 0;
                    }
                    if (a0 < nr - 1)
                    {
                        mat[2 + ld * 3] = host_d_rval(x0 + nx * (a0 + 1)) - host_d_rval(x0 + nx * a0);
                    }
                    else
                    {
                        mat[2 + ld * 3] = 0;
                    }
                   
                    if (a0 > 0 && a0 < nr - 1)
                    {
                        mat[3 + ld * 0] = 0;
                        mat[3 + ld * 1] = 0;
                        mat[3 + ld * 2] = Math.Pow(mat[2 + ld * 2], 2);
                        mat[3 + ld * 3] = Math.Pow(mat[2 + ld * 3], 2);
                    }
                    else
                    {
                        if (a0 == 0)
                        {
                            mat[3 + ld * 0] = 0;
                            mat[3 + ld * 1] = 0;
                            mat[3 + ld * 2] = 1;
                            mat[3 + ld * 3] = 0;                           
                        }
                        if (a0 == nr - 1)
                        {
                            mat[3 + ld * 0] = 0;
                            mat[3 + ld * 1] = 0;
                            mat[3 + ld * 2] = 0;
                            mat[3 + ld * 3] = 1;                           
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
                                    status = opc_dgesv(4, 4, ap, 4, 4, 4, ipivp, bp + 16 * y0, 4);
                                    if (status != 0) throw new System.Exception();
                                }
                            }
                        }
                    }
                }
            }
            return invm;
        }

    }
}
