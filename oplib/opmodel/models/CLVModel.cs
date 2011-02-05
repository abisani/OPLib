// CLVModel.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
    /** A Monte Carlo pricing model which assumes local volatility - i.e.
     * the volatility of the asset price is a function of time and the asset price only.
     **/
    public class CLVModel : CModel
    {
        public double[][] SDrift_i_y;
        public double[][] SVol_i_y;

        public CLVModel(Types.SGrid grid, string name)
            : base(grid, name)
        {
        }


        public override void mkgen(double[,] taumatrix_ccol, double[,] rhomatrix_ic)
        {
            throw new NotImplementedException();
        }


        public void mkgen(double[][] SDrift_i_y, double[][] SVol_i_y)
        {
            this.SDrift_i_y = new double[grid.ni][];
            this.SVol_i_y = new double[grid.ni][];

            for (int i = 0; i < grid.ni; i++)
            {
                this.SDrift_i_y[i] = new double[grid.d];
                this.SVol_i_y[i] = new double[grid.d];
                SDrift_i_y[i].CopyTo(this.SDrift_i_y[i], 0);
                SVol_i_y[i].CopyTo(this.SVol_i_y[i], 0);
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
            double[] invm = grid.host_d_invm;
            CArray.calloc(ref _host_d_gen_yy_i, grid.ni * grid.d * grid.d);

            for (int i = 0; i < grid.ni; i++)
            {
                double xi;
                double[] rhs = new double[2];
                double sum;

                for (int x0 = 0; x0 < grid.d; x0++)
                {
                    if (x0 > 0 && x0 < grid.d - 1)
                    {
                        rhs[0] = this.SDrift_i_y[i][x0];
                    }
                    else
                    {
                        rhs[0] = 0;
                    }
                    if (x0 > 0 && x0 < grid.d - 1)
                    {
                        rhs[1] = Math.Pow(this.SVol_i_y[i][x0], 2);
                    }
                    else
                    {
                        rhs[1] = 0;
                    }
                    int x1;
                    sum = 0;
                    x1 = x0 - 1;
                    if (x0 > 0)
                    {
                        xi = 0;
                        for (int i1 = 0; i1 < 2; i1++) xi += invm[4 * x0 + 0 + 2 * i1] * rhs[i1];
                        this._host_d_gen_yy_i[i * grid.d * grid.d + (x0 + grid.d * x1)] = xi;
                        sum += xi;
                    }
                    x1 = x0 + 1;
                    if (x0 < grid.d - 1)
                    {
                        xi = 0;
                        for (int i1 = 0; i1 < 2; i1++) xi += invm[4 * x0 + 1 + 2 * i1] * rhs[i1];
                        this._host_d_gen_yy_i[i * grid.d * grid.d + (x0 + grid.d * x1)] = xi;
                        sum += xi;
                    }
                    this._host_d_gen_yy_i[i * grid.d * grid.d + (x0 + grid.d * x0)] = -sum;
                }
            }
        }




        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sglvg(uint gpu_gen_yy_i,
                     int ni, int d, uint gpu_invmat, uint gpu_pars);

        override protected void device_sgen()
        {
            int d = grid.d;
            int ni = grid.ni;
            CArray.alloc(ref this._device_gen_yy_i, ni * d * d, EType.float_t, EMemorySpace.device, this, "_device_gen_yy_i");
            
            unsafe
            {

                CArray.alloc(ref _device_pars, 2 * ni * d, EType.float_t, EMemorySpace.device, this, "_device_pars");
                float[] spar = new float[2 * ni * d];
                for (int i = 0; i < ni; i++)
                {
                    for (int y0 = 0; y0 < d; y0++)
                    {
                        spar[0 * ni * d + i * d + y0] = (float)this.SDrift_i_y[i][y0];
                        spar[1 * ni * d + i * d + y0] = (float)SVol_i_y[i][y0];
                    }
                }
                CArray.copy(ref _device_pars, spar);
            }
            opcuda_sglvg(_device_gen_yy_i.ptr, ni, d, _device_invm.ptr, _device_pars.ptr);
        }
    }
}

