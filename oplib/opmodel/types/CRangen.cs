// CRangen.cs --- Part of the project OPLib 1.0, a high performance pricing library
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

namespace OPModel.Types
{
    static public class CRangen
    {

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private int opc_rangen_getstatus();

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_rangen_init_seed(uint* seed_th, int nth);

        static private int _nth;

        static public int nth
        {
            get
            {
                return _nth;
            }
        }

        static public void stinit()
        {
            mtinit(1);
        }

        static public void stinit(int seed)
        {
            mtinit(1, seed);
        }


        static public void mtinit(int nth)
        {
            Random rg = new Random();
            mtinit(nth, rg);
        }

        static public void mtinit(int nth, int seed)
        {
            Random rg = new Random(seed);
            mtinit(nth, rg);
        }


        static public void mtinit(int nth, Random rg)
        {
            _nth = nth;
            uint[] seed_th = new uint[nth];
            double[] rand = new double[1000];

            for (int th = 0; th < nth; th++)
            {
                seed_th[th] = (uint)rg.Next(1, int.MaxValue);
            }

            unsafe
            {
                fixed (uint* seed_thp = &seed_th[0])
                {
                    opc_rangen_init_seed(seed_thp, nth);
                }

            }

            for (int th = 0; th < nth; th++)
            {
                int status = opc_rangen_getstatus();
                if (status > 0) throw new System.Exception();
                genunif(rand, 0, 1, th);
            }
        }

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_rangen_finalize();
        static public void finalize()
        {
            opc_rangen_finalize();
        }

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_genunif(double* r, int n, double a, double b, int th);

        static public void genunif(double[] r, double a, double b)
        {
            if (nth != 0) throw new System.Exception();
            genunif(r, a, b, 0);
        }
        
        
        static public void genunif(double[] r, double a, double b, int th)
        {
            int n = r.GetLength(0);
            unsafe
            {
                if (n == 0) throw new System.Exception();
                fixed (double* rp = &r[0])
                {
                    int status = opc_rangen_getstatus();
                    if (status != 0) throw new System.Exception("gennor: error status = " + status);
                    opc_genunif(rp, n, a, b, th);
                    if (status != 0) throw new System.Exception("gennor: error status = " + status);
                }
            }
        }


        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_gennor(double* r, int n, double mean, double sigma, int th);

        static public void gennor(double[] r, double mean, double sigma)
        {
            if (nth != 0) throw new System.Exception();
            gennor(r, mean, sigma, 0);
        }


        static public void gennor(double[] r, double mean, double sigma, int th)
        {
            int n = r.GetLength(0);
            unsafe
            {
                fixed (double* rp = &r[0])
                {
                    opc_gennor(rp, n, mean, sigma, th);
                    int status = opc_rangen_getstatus();
                    if (status != 0) throw new System.Exception("gennor: error status = " + status);
                }
            }
        }

        static public void gennor(double[,] r, double mean, double sigma)
        {
            if (nth != 0) throw new System.Exception();
            gennor(r, mean, sigma, 0);
        }

        static public void gennor(double[,] r, double mean, double sigma, int th)
        {
            int n = r.Length;
            unsafe
            {
                fixed (double* rp = &r[0, 0])
                {
                    opc_gennor(rp, n, mean, sigma, th);
                }
            }
        }
    }
}
