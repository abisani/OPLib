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
using System.IO;


namespace GMT
{
    class Program
    {

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_device_get_count();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_shutdown();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_cublas_init();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_set_device(uint device_number);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc_load_mt_gpu(byte* MT_stream, long sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc_nrng();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc_status_sz();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mc_setseed(IntPtr host_seedptr, uint mtptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mt_benchmark(uint mtptr, uint unif_ptr, int nscen);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_memcpy_d2h(uint dptr, IntPtr hptr, uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_thread_synchronize();


        static void Main(string[] args)
        {
            int ndev = opcuda_device_get_count();

            for (uint dev = 0; dev < ndev; dev++) run_benchmark_gpu_mt(dev);

            Console.Read();
        }

        static public void run_benchmark_gpu_mt(uint dev)
        {

            Console.WriteLine("running the Mersenne Twister benchmark on device " + dev);

            int nscen_per_batch = 4096 * 250;
            int nbatches = 200;

            int status = opcuda_cublas_init();
            if (status != 0) throw new ExecutionEngineException();
            opcuda_set_device(dev);

            FileStream stream;
            try
            {
                stream = new FileStream("MersenneTwister.dat", FileMode.Open, FileAccess.Read);
            }
            catch
            {
                Console.WriteLine("device_mc_init() failed on device " + dev + ". Aborting");
                return;
            }

            byte[] MT = new byte[stream.Length];
            stream.Read(MT, 0, (int)stream.Length);

            unsafe
            {
                fixed (byte* MTp = &MT[0])
                {
                    status = opcuda_mc_load_mt_gpu(MTp, stream.Length);
                    if (status != 0) throw new System.Exception();
                }
            }


            Random rand = new Random();
            int nrng = opcuda_mc_nrng();

            CArray host_seed_rg = new CArray(nrng, EType.int_t, EMemorySpace.host, null, "host_seed_rg");
            unsafe
            {
                int* seed_rg = (int*)host_seed_rg.hptr;
                for (int rg = 0; rg < nrng; rg++)
                {
                    seed_rg[rg] = (int)(rand.NextDouble() * int.MaxValue);
                }
            }

            CArray device_rgstatus = new CArray(opcuda_mc_status_sz(), EType.int_t, EMemorySpace.device, null, "mcbuf._device_rgstatus");
            unsafe
            {
                opcuda_mc_setseed(host_seed_rg.hptr, device_rgstatus.ptr);
            }

            CArray device_unif_s = new CArray(nscen_per_batch, EType.float_t, EMemorySpace.device, null, "device_unif_s");
            CArray host_unif_s = new CArray(nscen_per_batch, EType.float_t, EMemorySpace.host, null, "host_unif_s");

            CStopWatch sw = new CStopWatch();
            sw.Reset();
            unsafe
            {
                for (int b = 0; b < nbatches; b++)
                {
                    opcuda_mt_benchmark(device_rgstatus.ptr, device_unif_s.ptr, nscen_per_batch);
                    opcuda_memcpy_d2h(device_unif_s.ptr, host_unif_s.hptr, (uint)(sizeof(short) * host_unif_s.length));
                }
            }
            opcuda_thread_synchronize();
            double time = sw.Peek();

            double nevals = (double)nbatches * (double)nscen_per_batch;
            double milion_evals_per_second = nevals / (1000000 * time);
            status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();
            Console.WriteLine("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");


            sw.Reset();
            unsafe
            {
                for (int b = 0; b < nbatches; b++)
                {
                    opcuda_mt_benchmark(device_rgstatus.ptr, device_unif_s.ptr, nscen_per_batch);
                }
            }
            opcuda_thread_synchronize();
            time = sw.Peek();

            nevals = (double)nbatches * (double)nscen_per_batch;
            milion_evals_per_second = nevals / (1000000 * time);
            status = opcuda_shutdown();
            if (status != 0) throw new ExecutionEngineException();
            Console.WriteLine("mc performance: " + String.Format("{0:0.0}", milion_evals_per_second) + " milion eval/sec");

        }

    }
}
