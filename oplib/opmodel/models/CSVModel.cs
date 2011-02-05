// CSVModel.cs --- Part of the project OPLib 1.0, a high performance pricing library
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

namespace OPModel
{
    /** A Monte Carlo pricing model which assumes stochastic volatility.
     **/
    public class CSVModel : CModel
    {
        private double[,] _SVol_iy;
        private double[,] _VolVol_iy;
        private double[,] _VolDrift_iy;
        private double[,] _Jumpsz_plus_iy;
        private double[,] _Jumpsz_minus_iy;
        public const int ncurves = 8;
        public const int ntmcols = 4;
        private double[,] _taumatrix_ccol;
        private double[,] _rhomatrix_ic;


        public CSVModel(Types.S2DGrid grid, string name, double[] df_i)
            : base(grid, name)
        {
            if (grid.fpu == EFloatingPointUnit.device)
            {
                device_set_discount_curve(df_i);
                return;
            }
            if (grid.fpu == EFloatingPointUnit.host && grid.fpp == EFloatingPointPrecision.bit64)
            {
                grid.host_d_set_discount_curve(df_i);
                return;
            }
            if (grid.fpu == EFloatingPointUnit.host && grid.fpp == EFloatingPointPrecision.bit32)
            {
                grid.host_s_set_discount_curve(df_i);
                return;
            }
            throw new System.Exception();
        }



        private double SVol(int i, int y)
        {
            return _SVol_iy[i, y];
        }

        private double VolVol(int i, int y)
        {
            return _VolVol_iy[i, y];
        }

        private double VolDrift(int i, int y)
        {
            return _VolDrift_iy[i, y];
        }

        private double Jumpsz_minus(int i, int y)
        {
            return _Jumpsz_minus_iy[i, y];
        }

        private double Jumpsz_plus(int i, int y)
        {
            return _Jumpsz_plus_iy[i, y];
        }


        override public void mkgen(double[,] taumatrix_ccol, double[,] rhomatrix_ic)
        {

            if (taumatrix_ccol.GetLength(0) != 8) throw new System.Exception();
            if (taumatrix_ccol.GetLength(1) != 4) throw new System.Exception();
            grid.Set_ncurves(taumatrix_ccol.GetLength(0));
            grid.Set_ntmcols(taumatrix_ccol.GetLength(1));

            CArray.copy(ref _taumatrix_ccol, taumatrix_ccol);
            CArray.copy(ref _rhomatrix_ic, rhomatrix_ic);
            CArray.alloc(ref _SVol_iy, grid.ni, grid.d);
            CArray.alloc(ref _VolVol_iy, grid.ni, grid.d);
            CArray.alloc(ref _VolDrift_iy, grid.ni, grid.d);
            CArray.alloc(ref _Jumpsz_minus_iy, grid.ni, grid.d);
            CArray.alloc(ref _Jumpsz_plus_iy, grid.ni, grid.d);

            for (int i = 0; i < grid.ni; i++)
            {
                double t1, xi;
                t1 = (grid.t_i[i] - grid.today).TotalDays / 365.25;
                double vol = 0, lowbeta = 0, highbeta = 0, volvol = 0, volmrr = 0,
                                 volmrl = 0, jumpsz_minus = 0, jumpsz_plus = 0;
                for (int c = 0; c < grid.ncurves; c++)
                {
                    double A, B;
                    if (taumatrix_ccol[c, 2] == 0)
                    {
                        xi = taumatrix_ccol[c, 0];
                    }
                    else
                    {
                        A = taumatrix_ccol[c, 0];

                        B = taumatrix_ccol[c, 1] - A;
                        xi = A + B * (1 + taumatrix_ccol[c, 3] * t1 / taumatrix_ccol[c, 2]) * Math.Exp(-t1 / taumatrix_ccol[c, 2]);
                    }
                    if (rhomatrix_ic != null)
                    {
                        xi += rhomatrix_ic[i, c];
                    }
                    if (Double.IsNaN(xi) || Double.IsInfinity(xi)) throw new System.Exception();

                    switch (c)
                    {
                        case 0: vol = xi; break;
                        case 1: lowbeta = xi; break;
                        case 2: highbeta = xi; break;
                        case 3: volvol = xi; break;
                        case 4: volmrr = xi; break;
                        case 5: volmrl = xi; break;
                        case 6: jumpsz_minus = xi; break;
                        case 7: jumpsz_plus = xi; break;
                    }
                }

                double xval_max = Math.Sqrt(grid.host_d_xval(grid.nx - 1)), xval_min = Math.Sqrt(grid.host_d_xval(0));

                for (int y = 0; y < grid.d; y++)
                {
                    double beta = (highbeta * (Math.Sqrt(grid.host_d_xval_y[y]) - xval_min)
                                                            + lowbeta * (xval_max - Math.Sqrt(grid.host_d_xval_y[y])))
                                                            / (xval_max - xval_min);
                    _SVol_iy[i, y] = grid.host_d_rval(y) * vol * Math.Pow(grid.host_d_xval(y) / grid.host_d_xval(grid.y0), beta);
                    _VolVol_iy[i, y] = volvol;
                    _VolDrift_iy[i, y] = volmrr * (volmrl - grid.host_d_rval(y));
                    _Jumpsz_minus_iy[i, y] = jumpsz_minus;
                    _Jumpsz_plus_iy[i, y] = jumpsz_plus;
                }
            }


            if (grid.fpu == EFloatingPointUnit.device)
            {
                device_sgen();
            }
            if (grid.fpu == EFloatingPointUnit.host)
            {
                host_d_gen();
            }
        }




        override public void host_d_gen()
        {
            int ni = grid.ni;
            int d = grid.d;
            double[] invm = grid.host_d_invm;
            CArray.calloc(ref _host_d_gen_yy_i, ni * d * d);
            double[] rhs = new double[4];
            double xi;
            int nx = grid.nx;
            int nr = grid.nr;

            for (int i = 0; i < ni; i++)
            {
                for (int y1 = 0; y1 < d; y1++)
                {
                    for (int y2 = 0; y2 < d; y2++)
                    {
                        this._host_d_gen_yy_i[i * d * d + (y1 + d * y2)] = 0;
                    }
                }

                for (int x0 = 0; x0 < nx; x0++)
                {
                    for (int r0 = 0; r0 < nr; r0++)
                    {
                        int y0 = x0 + nx * r0;

                        rhs[0] = 0;
                        if (x0 > 0 && x0 < nx - 1)
                        {
                            rhs[1] = Math.Pow(SVol(i, y0), 2);
                        }
                        else
                        {
                            rhs[1] = 0;
                        }
                        rhs[2] = VolDrift(i, y0);
                        if (r0 > 0 && r0 < nr - 1)
                        {
                            rhs[3] = Math.Pow(VolVol(i, y0), 2);
                        }
                        else
                        {
                            rhs[3] = 0;
                        }
                        int x1, r1, y1;
                        x1 = x0 - 1;
                        r1 = r0;
                        y1 = x1 + nx * r1;
                        if (x0 > 0)
                        {
                            xi = 0;
                            for (int i1 = 0; i1 < 4; i1++) xi += invm[16 * y0 + 0 + 4 * i1] * rhs[i1];
                            this._host_d_gen_yy_i[i * d * d + (y0 + d * y1)] = xi;
                        }

                        x1 = x0 + 1;
                        r1 = r0;
                        y1 = x1 + nx * r1;
                        if (x0 < nx - 1)
                        {
                            xi = 0;
                            for (int i1 = 0; i1 < 4; i1++) xi += invm[16 * y0 + 1 + 4 * i1] * rhs[i1];
                            this._host_d_gen_yy_i[i * d * d + (y0 + d * y1)] = xi;
                        }

                        x1 = x0;
                        r1 = r0 - 1;
                        y1 = x1 + nx * r1;
                        if (r0 > 0)
                        {
                            xi = 0;
                            for (int i1 = 0; i1 < 4; i1++) xi += invm[16 * y0 + 2 + 4 * i1] * rhs[i1];
                            this._host_d_gen_yy_i[i * d * d + (y0 + d * y1)] = xi;
                        }

                        x1 = x0;
                        r1 = r0 + 1;
                        y1 = x1 + nx * r1;
                        if (r0 < nr - 1)
                        {
                            xi = 0;
                            for (int i1 = 0; i1 < 4; i1++) xi += invm[16 * y0 + 3 + 4 * i1] * rhs[i1];
                            this._host_d_gen_yy_i[i * d * d + (y0 + d * y1)] = xi;
                        }
                    }
                }

                for (int x0 = 0; x0 < grid.nx; x0++)
                {
                    for (int r0 = 0; r0 < grid.nr; r0++)
                    {
                        int y0 = x0 + grid.nx * r0;
                        for (int r1 = r0 + 1; r1 < grid.nr; r1++)
                        {
                            double prob = 0;
                            double sum0 = 0;
                            for (int x1 = 0; x1 < grid.nx; x1++)
                            {
                                int y1 = x1 + grid.nx * r1;
                                prob += _host_d_gen_yy_i[i * d * d + (y0 + d * y1)];
                            }
                            if (prob > 0)
                            {
                                for (int x1 = 0; x1 < x0 - 1; x1++)
                                {
                                    int y1 = x1 + grid.nx * r1;
                                    if (Jumpsz_minus(i, y0) > 0)
                                    {
                                        double jumpsz = Jumpsz_minus(i, y0) * grid.host_d_xval(y0);
                                        sum0 += Math.Exp((grid.host_d_xval(y1) - grid.host_d_xval(y0)) / jumpsz);
                                    }
                                }
                                for (int x1 = x0 + 1; x1 < grid.nx; x1++)
                                {
                                    int y1 = x1 + grid.nx * r1;
                                    if (Jumpsz_plus(i, y0) > 0)
                                    {
                                        double jumpsz = Jumpsz_plus(i, y0) * grid.host_d_xval(y0);
                                        sum0 += Math.Exp(-(grid.host_d_xval(y1) - grid.host_d_xval(y0)) / jumpsz);
                                    }
                                }
                                if (sum0 > 0)
                                {
                                    double ratio = prob / sum0;
                                    for (int x1 = 0; x1 < x0 - 1; x1++)
                                    {
                                        int y1 = x1 + grid.nx * r1;
                                        _host_d_gen_yy_i[i * d * d + (y0 + d * y1)] = ratio * Math.Exp((grid.host_d_xval(y1) - grid.host_d_xval(y0)) / Jumpsz_minus(i, y0));
                                        if (Double.IsNaN(_host_d_gen_yy_i[i * d * d + y0 + d * y1])) throw new System.Exception();
                                    }
                                    for (int x1 = x0 + 1; x1 < grid.nx; x1++)
                                    {
                                        int y1 = x1 + grid.nx * r1;
                                        _host_d_gen_yy_i[i * d * d + (y0 + d * y1)] = ratio * Math.Exp(-(grid.host_d_xval(y1) - grid.host_d_xval(y0)) / Jumpsz_plus(i, y0));
                                        if (Double.IsNaN(_host_d_gen_yy_i[i * d * d + y0 + d * y1])) throw new System.Exception();
                                    }
                                }
                            }
                        }
                    }
                }


                for (int x0 = 0; x0 < grid.nx; x0++)
                {
                    for (int r0 = 0; r0 < grid.nr; r0++)
                    {
                        double sum0 = 0;
                        double drift = 0;
                        int y0 = x0 + grid.nx * r0;
                        for (int y1 = 0; y1 < d; y1++)
                        {
                            if (y0 != y1)
                            {
                                if (this._host_d_gen_yy_i[i * d * d + (y0 + d * y1)] < 0)
                                {
                                    this._host_d_gen_yy_i[i * d * d + (y0 + d * y1)] = 0;
                                }
                                else
                                {
                                    drift += _host_d_gen_yy_i[i * d * d + (y0 + d * y1)] * (grid.host_d_xval(y1) - grid.host_d_xval(y0));
                                    sum0 += _host_d_gen_yy_i[i * d * d + (y0 + d * y1)];
                                }
                            }
                        }
                        double drift0 = grid.host_d_ir_yi[y0 + grid.d * i] * grid.host_d_xval(y0);
                        if (drift > drift0)
                        {
                            if (x0 > 0)
                            {
                                int y1 = x0 - 1 + grid.nx * r0;
                                double ratio = (drift - drift0) / (grid.host_d_xval(y1) - grid.host_d_xval(y0));
                                _host_d_gen_yy_i[i * d * d + (y0 + d * y1)] += ratio;
                                sum0 += ratio;
                            }
                        }
                        else
                        {
                            if (x0 < grid.nx - 1)
                            {
                                int y1 = x0 + 1 + grid.nx * r0;
                                double ratio = (drift0 - drift) / (grid.host_d_xval(y1) - grid.host_d_xval(y0));
                                _host_d_gen_yy_i[i * d * d + (y0 + d * y1)] += ratio;
                                sum0 += ratio;
                            }
                        }
                        this._host_d_gen_yy_i[i * d * d + (y0 + d * y0)] = -sum0;
                    }
                }

            }
        }



        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sgsvg(uint gpu_gen_yy_i, int ni, int nx, int nr, uint gpu_invmat, uint gpu_pars);

        override protected void device_sgen()
        {

            int ni = grid.ni;
            int d = grid.d;

            CArray.alloc(ref this._device_gen_yy_i, ni * d * d, EType.float_t, EMemorySpace.device, this, "_device_gen_yy_i");

            unsafe
            {
                _device_pars = new CArray(d + 6 * ni * d, EType.float_t, EMemorySpace.device, this, "_device_pars");
                float[] spar = new float[d + 6 * ni * d];
                for (int y0 = 0; y0 < d; y0++)
                {
                    spar[y0] = (float)grid.host_d_xval(y0);
                    for (int i = 0; i < ni; i++)
                    {
                        spar[d + 0 * ni * d + i * d + y0] = (float)(grid.host_d_ir_yi[d * i] * grid.host_d_xval(y0));
                        spar[d + 1 * ni * d + i * d + y0] = (float)SVol(i, y0);
                        spar[d + 2 * ni * d + i * d + y0] = (float)VolDrift(i, y0);
                        spar[d + 3 * ni * d + i * d + y0] = (float)VolVol(i, y0);
                        spar[d + 4 * ni * d + i * d + y0] = (float)Jumpsz_minus(i, y0);
                        spar[d + 5 * ni * d + i * d + y0] = (float)Jumpsz_plus(i, y0);
                    }
                }
                CArray.copy(ref _device_pars, spar);
            }

            opcuda_ssetall(_device_gen_yy_i.ptr, grid.d * grid.d * grid.ni, 0, 1);
            float[] sgen = new float[_device_gen_yy_i.length];
            CArray.copy(ref sgen, _device_gen_yy_i);
            opcuda_sgsvg(_device_gen_yy_i.ptr, ni, grid.nx, grid.nr, _device_invm.ptr, _device_pars.ptr);
            CArray.copy(ref sgen, _device_gen_yy_i);
        }

    }
}
