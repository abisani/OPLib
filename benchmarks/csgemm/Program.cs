// CPayoff.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
using OPModel;
using OPModel.Types;


namespace CSGEMM
{
    class Program
    {

        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private void opc_sgemm(float* a, float* b, float* c, int lda, int ldb, int ldc, int m, int n, int k);


        static void Main(string[] args)
        {

            Console.WriteLine("running sgemm on the cpu");

            Random rg = new Random();
            int d = 96 * 6;
            int fsz = 3 * d * d;
            float[] host_float_buf = new float[fsz];

            for (int a = 0; a < host_float_buf.Length; a++)
                host_float_buf[a] = (float)(0.01 * rg.Next(-1000, 1000));

            CStopWatch sw = new CStopWatch();
            sw.Reset();

            int niter = 100;

            unsafe
            {
                fixed (float* ap = &host_float_buf[0])
                {
                    fixed (float* bp = &host_float_buf[d * d])
                    {
                        fixed (float* cp = &host_float_buf[2 * d * d])
                        {
                            for (int iter = 0; iter < niter; iter++)
                            {
                                opc_sgemm(ap, bp, cp, d, d, d, d, d, d);
                            }
                        }
                    }
                }
            }

            double time1 = sw.Peek();

            double nflops = 2d * niter * (double)d * (double)d * (double)d;
            double gigaflops_per_second = nflops / (1000000000d * time1);

            Console.Write(Environment.NewLine + "performance: " + String.Format("{0:0.0}", gigaflops_per_second) + " GF/sec");
            Console.Read();


        }
    }
}
