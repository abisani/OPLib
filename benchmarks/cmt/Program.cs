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


namespace CMT
{
    class Program
    {
        [System.Runtime.InteropServices.DllImport("opc")]
        static extern unsafe private int opc_mt_benchmark(int nscen_per_batch, float* unif_s, int th); 

        static void Main(string[] args)
        {
            Console.WriteLine("running the Mersenne Twister benchmark on the CPU");

            int nscen_per_batch = 4096 * 25;
            int nbatches = 2000;

            int nth = Environment.ProcessorCount;
            OPModel.Types.CRangen.mtinit(nth);

            float[] host_unif_scen_th = new float[nscen_per_batch * nth];

            OPModel.Types.CJobQueue Queue = new OPModel.Types.CJobQueue();
            object[] p_b = new object[nbatches];
            for (int b = 0; b < nbatches; b++)
            {
                host_d_mc_mt_func_input input = new host_d_mc_mt_func_input();
                input.nth = nth;
                input.batch = b;
                input.nscen_per_batch = nscen_per_batch;
                input.host_unif_scen_th = host_unif_scen_th;
                p_b[b] = input;
            }

            CStopWatch sw = new CStopWatch();
            sw.Reset();
            Queue.Exec(host_d_mc_mt_func, null, p_b, nth);
            double time = sw.Peek();

            double nevals = (double)nbatches * (double)nscen_per_batch;
            double milion_evals_per_second = nevals / (1000000 * time);

            Console.WriteLine("performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");
            host_unif_scen_th = null;
            System.GC.Collect();

            Console.Read();

        }

        static public Object host_d_mc_mt_func(Object p, int th)
        {
            host_d_mc_mt_func_input input = (host_d_mc_mt_func_input)p;
            unsafe
            {
                fixed (float* unif_scenp = &input.host_unif_scen_th[input.nscen_per_batch * th])
                {
                    opc_mt_benchmark(input.nscen_per_batch, unif_scenp, th);
                }
            }
            return null;
        }

        public struct host_d_mc_mt_func_input
        {
            public int nth;
            public float[] host_unif_scen_th;
            public int batch;
            public int nscen_per_batch;
        }

        public struct host_d_mc_mt_func_return
        {
        }

    }


}
