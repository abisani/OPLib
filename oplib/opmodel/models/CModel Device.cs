// CModelDevice.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
using OPModel;
using System.IO;

namespace OPModel
{

    abstract public partial class CModel
    {

        protected CArray _device_gen_yy_i;						
        protected CArray _device_pars;							
        private CArray _device_y0;
        protected CArray _device_invm;


        [System.Runtime.InteropServices.DllImport("opcuda")] 
        static extern unsafe private int opcuda_cublas_init();

        [System.Runtime.InteropServices.DllImport("opcuda")] 
        static extern unsafe private int opcuda_shutdown();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_thread_synchronize();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sdot2(uint aptr, uint bptr, uint cptr, int m, int n);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sek(int d, uint genPtr, float dt, uint tpkerPtr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_sck(int d, int ni, uint tpker_ptr, uint ctpker_ptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe protected void opcuda_ssetall(uint xPtr, int n, float c, int incx);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe protected int opcuda_scopy1(uint destination, uint source, uint n);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe protected void opcuda_sgemm(int m, int n, int k, float alpha, uint APtr, int lda, uint BPtr, int ldb, float beta, uint CPtr, int ldc);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe protected void opcuda_scopy(int n, uint xptr, int incx, uint yptr, int incy);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc1f(uint mtptr, int y0, uint y_sk, int nscen, int nk, int d,
                                                        uint ker_yym, uint m_k, uint yhostptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc_load_mt_gpu(byte* MT_stream, long sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mc_setseed(IntPtr host_seedptr, uint mtptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc_nrng();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_mc_status_sz();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mem_free_host(IntPtr hptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mem_free_device(uint ptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_get_status();



        protected void device_set_discount_curve(double[] df_i){

            grid.host_d_set_discount_curve(df_i);

            if (df_i != null)
            {
                if (grid.ni != df_i.Length) throw new System.Exception();
                float[] fdf_i = new float[grid.ni];
                for (int i = 0; i < grid.ni; i++)
                {
                    fdf_i[i] = (float)df_i[i];
                }

                grid._device_df_i = new CArray(grid.ni, EType.float_t, EMemorySpace.device, this, "grid.device_df_i");
                CArray.copy(ref grid._device_df_i, fdf_i);
            }


            float[] fir_yi = new float[grid.ni * grid.d];
            for (int i = 0; i < grid.ni; i++)
            {
                for (int y = 0; y < grid.d; y++)
                {
                    fir_yi[y + grid.d * i] = (float)grid.host_d_ir_yi[y + grid.d * i];
                }
            }

            grid._device_ir_yi = new CArray(grid.ni, EType.float_t, EMemorySpace.device, this, "grid.device_df_i");
            CArray.copy(ref grid._device_ir_yi, fir_yi);

        }


        public static void gpu_init()
        {
            opcuda_cublas_init();
        }

        public static void gpu_shutdown()
        {
            opcuda_shutdown();
        }


        public void device_thread_synchronize()
        {
            opcuda_thread_synchronize();
        }

        public void device_sdot2(CArray a, CArray b, ref CArray c, int m, int n)
        {
            if (a.length != m * n) throw new System.Exception();
            if (b.length != m * n) throw new System.Exception();
            if (c.length != m) throw new System.Exception();
            opcuda_sdot2(a.ptr, b.ptr, c.ptr, m, n);
            _gpu_nflops += (double)m * n;
        }


        public void device_grid_init()
        {
            int status = opcuda_cublas_init();
            if (status != 0) throw new ExecutionEngineException();

            this._device_invm = new CArray(grid.host_d_invm.Length * sizeof(float), 
                                                EType.float_t, EMemorySpace.device, this, "_device_invm");
            float[] sinvm = new float[grid.host_d_invm.Length];
            for (int j = 0; j < grid.host_d_invm.Length; j++)
            {
                sinvm[j] = (float)grid.host_d_invm[j];
            }
            CArray.copy(ref _device_invm, sinvm);
        }


        abstract protected void device_sgen();

        private void device_mc_sek()
        {
            int d = grid.d;

            for (int j = 0; j < _mcplan.nj; j++)
            {
                int i = mcplan.i_j[j];
                opcuda_sek(d, (uint)(_device_gen_yy_i.ptr + i * d * d * sizeof(float)), (float)mcplan.host_d_dt_j[j], (uint)(mcbuf._device_xtpk_yy_j.ptr + j * d * d * sizeof(float)));
                _gpu_nflops += 4f * d * d;
            }
        }




        private void device_mc_sck()
        {
            int d = grid.d;

            unsafe
            {
                opcuda_sck(d, mcplan.nm, mcbuf._device_tpk_yy_m.ptr, mcbuf._device_ctpk_yy_m.ptr);
            }
        }



        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_ssqmm(int d, int ni, uint A_i, uint B_i, uint C_i);

        void device_mc_sfex()
        {
            int d = grid.d;          
            device_mc_sek();
            for (int r = 0; r < mcplan.nr; r++)
            {
                opcuda_ssqmm(d, mcplan.njtodo_r[r], (uint)(mcplan.device_A_idxr.ptr + r * mcplan.nj * sizeof(uint)),
                              (uint)(mcplan.device_A_idxr.ptr + r * mcplan.nj * sizeof(uint)),
                              (uint)(mcplan.device_C_idxr.ptr + r * mcplan.nj * sizeof(uint)));
                _gpu_nflops += 2f * d * d * d * mcplan.njtodo_r[r];
            }           
            device_mc_scopy();
        }



        void device_mc_scopy()
        {
            int d = grid.d;
            for (int j = 0; j < mcplan.nj; j++)
            {
                if (mcplan.niter_j[j] % 2 == 1)
                {
                    opcuda_scopy1((uint)(
                        mcbuf._device_xtpk_yy_j.ptr + j * d * d * sizeof(float)),
                        (uint)(mcbuf._device_xtpkbuf_yy_idx.ptr + j * d * d * sizeof(float)),
                        (uint)(d * d));

                }
            }
        }





        void device_mc_sminit()
        {
            int d = grid.d;
            for (int m = 0; m < mcplan.nm; m++)
            {
                int j;
                j = mcplan.jfactor_ms[m][0];
                {
                    opcuda_scopy1((uint)(mcbuf._device_tpk_yy_m.ptr + m * d * d * sizeof(float)),
                    (uint)(mcbuf._device_xtpk_yy_j.ptr + j * d * d * sizeof(float)),
                    (uint)(d * d));
                }
            }
        }



        void device_mc_smk()
        {
            int d = grid.d;         
            for (int s = 0; s < mcplan.ns; s++)
            {
                opcuda_ssqmm(d, mcplan.nmtodo_s[s], (uint)(mcplan.device_A_idxs.ptr + s * mcplan.nm * sizeof(uint)),
                       (uint)(mcplan.device_B_idxs.ptr + s * mcplan.nm * sizeof(uint)),
                       (uint)(mcplan.device_C_idxs.ptr + s * mcplan.nm * sizeof(uint)));
                _gpu_nflops += 2f * d * d * d * mcplan.nmtodo_s[s];
            }
        }

        void device_mc_smcopy()
        {
            int d = grid.d;           
            for (int m = 0; m < mcplan.nm; m++)
            {
                if (mcplan.jfactor_ms[m].Length % 2 == 0)
                {
                    int status;
                    status = opcuda_scopy1((uint)(mcbuf._device_tpk_yy_m.ptr + m * d * d * sizeof(float)),
                             (uint)(mcbuf._device_xtpkbuf_yy_idx.ptr + m * d * d * sizeof(float)),
                             (uint)(d * d));
                    if (status != 0) throw new System.Exception();
                }
            }
        }


        public void device_setall(ref CArray buf, double c)
        {
            opcuda_ssetall(buf.ptr, buf.length, (float)c, 1);
        }

        public int device_mc_init()
        {

            FileStream stream = new FileStream("MersenneTwister.dat", FileMode.Open, FileAccess.Read);

            byte[] MT = new byte[stream.Length];
            stream.Read(MT, 0, (int) stream.Length);

            unsafe
            {
                fixed (byte* MTp = &MT[0])
                {
                    int status = opcuda_mc_load_mt_gpu(MTp, stream.Length);
                    if(status !=0) throw new System.Exception();
                }
            }

            Random rand = new Random();
            int nrng = opcuda_mc_nrng();

            CArray host_seed_rg = new CArray(nrng, EType.int_t, EMemorySpace.host, this, "host_seed_rg");
            unsafe
            {
                int* seed_rg = (int*)host_seed_rg.hptr;
                for (int rg = 0; rg < nrng; rg++)
                {
                    seed_rg[rg] = (int)(rand.NextDouble() * int.MaxValue);
                }
            }

            mcbuf._device_rgstatus = new CArray(opcuda_mc_status_sz(), EType.int_t, EMemorySpace.device, this, "mcbuf._device_rgstatus");

            unsafe
            {
                opcuda_mc_setseed(host_seed_rg.hptr, mcbuf._device_rgstatus.ptr);
            }
            return 0;
        }


        public void device_mc_run1f(double[] payoff_a, CMCEvaluator evaluator)
        {         
            if (mcplan._nscen_per_batch % 4096 != 0) throw new System.Exception();
            CArray device_y_sk = new CArray(mcplan.nk * mcplan._nscen_per_batch, EType.short_t, EMemorySpace.device, this, "_device_volatile_buf");

            for (int b = 0; b < mcplan._nbatches; b++)
            {
                unsafe
                {
                    fixed (short* yhost_sk = &mcbuf.host_y_sk_th[0])
                    {   
                        int status = opcuda_mc1f(mcbuf._device_rgstatus.ptr, grid.y0,
                                                   device_y_sk.ptr,
                                                   mcplan._nscen_per_batch, mcplan.nk, grid.d,
                                                   mcbuf._device_ctpk_yy_m.ptr, mcbuf._device_m_k.ptr,
                                                   (uint)yhost_sk);

                        if (status > 0) throw new System.Exception();
                        device_thread_synchronize();

                        evaluator.eval(yhost_sk, 0, b);
                    }
                }
            }
        }





    }
}
