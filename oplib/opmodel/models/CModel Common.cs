// CModelCommon.cs --- Part of the project OPLib 1.0, a high performance pricing library
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

namespace OPModel
{
    /** The base class for monte carlo pricing models.
     * */
    abstract public partial class CModel
    {

        private Types.SGrid _grid;
        private SMCPlan _mcplan;
        private SMCBuffers _mcbuf;
        protected double _gpu_nflops;
        protected double _cpu_nflops;

        public void reset_flop_counter()
        {
            _cpu_nflops = 0;
            _gpu_nflops = 0;
        }

        public double peek_gpu_flops()
        {
            return _gpu_nflops;
        }


        public SMCBuffers mcbuf
        {
            get
            {
                return _mcbuf;
            }
        }

        public Types.SGrid grid
        {
            get
            {
                return _grid;
            }
        }

        public SMCPlan mcplan
        {
            get
            {
                return _mcplan;
            }
        }

        public double gpu_nflops
        {
            get
            {
                return _gpu_nflops;
            }
        }

        public double cpu_nflops
        {
            get
            {
                return _cpu_nflops;
            }
        }

       
        abstract public void mkgen(double[,] taumatrix_ccol, double[,] rhomatrix_ic);

        public CModel(Types.SGrid grid, string name)
        {
            _grid = grid;

            if (grid.fpu == EFloatingPointUnit.device)
            {
                device_grid_init(); 
            }
        }

        public void set_discount_curve(double[] df_i)
        {
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
            if (grid.fpu == EFloatingPointUnit.device && grid.fpp == EFloatingPointPrecision.bit32)
            {
                device_set_discount_curve(df_i);
                return;
            }
            throw new System.Exception();
        }



        public void make_mc_plan(int nscen_per_batch, int nbatches, DateTime[] t_k)
        {

            _mcbuf = new SMCBuffers();
            _mcplan = new SMCPlan();
            _mcplan._nscen_per_batch = nscen_per_batch;
            _mcplan._nbatches = nbatches;
            _mcplan._nscen = _mcplan._nscen_per_batch * _mcplan._nbatches;
           
            if (grid.fpu == EFloatingPointUnit.device)
            {
                _device_y0 = new CArray(2, EType.short_t, EMemorySpace.device, this, "_device_y0");
                short[] host_y0 = new short[1];
                host_y0[0] = (short)grid.y0;
                CArray.copy(ref _device_y0, host_y0);
            }

            make_mc_plan(t_k);
        }






        private void make_mc_plan(DateTime[] t_k)
        {
            int ni = grid.ni;
            int d = grid.d;
            DateTime[] t_i = grid.t_i;
            double dt0 = grid.dt0.Days / 365.25;

            if (t_k[0] <= grid.today) throw new System.Exception();
            for (int k = 1; k < mcplan.nk; k++)
            {
                if (t_k[k] <= t_k[k - 1]) throw new System.Exception();
            }

            if (t_i[0] <= grid.today) throw new System.Exception();
            int i;
            for (i = 1; i < t_i.Length; i++)
            {
                if (t_i[i] <= t_i[i - 1]) throw new System.Exception();
            }

            _mcplan.t_k = new DateTime[t_k.Length];
            t_k.CopyTo(mcplan.t_k, 0);
            _mcplan.nk = mcplan.t_k.Length;

            CArray.calloc(ref _mcplan.t_q, ni + mcplan.nk);
            t_k.CopyTo(mcplan.t_q, 0);
            t_i.CopyTo(mcplan.t_q, mcplan.nk);
            Array.Sort(mcplan.t_q);

            int q0 = 0;
            for (int q = 1; q < ni + mcplan.nk; q++)
            {
                if (mcplan.t_q[q] > mcplan.t_q[q0] && mcplan.t_q[q] <= t_k[mcplan.nk - 1])
                {
                    mcplan.t_q[++q0] = mcplan.t_q[q];
                }
            }
            _mcplan.nq = q0 + 1;
            Array.Resize(ref _mcplan.t_q, mcplan.nq);

            CArray.calloc(ref _mcplan.i_q, mcplan.nq);
            i = 0;
            for (int q = 0; q < mcplan.nq; q++)
            {
                if (mcplan.t_q[q] <= t_i[i]) mcplan.i_q[q] = i;
                else mcplan.i_q[q] = ++i;
            }

            bool[] todo_q = new bool[mcplan.nq];
            for (int q1 = 0; q1 < mcplan.nq; q1++)
            {
                todo_q[q1] = true;
            }

            for (int q2 = 1; q2 < mcplan.nq; q2++)
            {
                if ((mcplan.t_q[q2] - mcplan.t_q[q2 - 1] == (mcplan.t_q[0] - grid.today)) && (mcplan.i_q[0] == mcplan.i_q[q2]))
                    todo_q[q2] = false;
            }
            for (int q1 = 1; q1 < mcplan.nq; q1++)
            {
                for (int q2 = q1 + 1; q2 < mcplan.nq; q2++)
                {
                    if ((mcplan.t_q[q2] - mcplan.t_q[q2 - 1] == mcplan.t_q[q1] - mcplan.t_q[q1 - 1]) && (mcplan.i_q[q1] == mcplan.i_q[q2]))
                        todo_q[q2] = false;
                }
            }

            CArray.calloc(ref _mcplan.j_q, mcplan.nq);
            _mcplan.nj = 0;
            for (int q1 = 0; q1 < mcplan.nq; q1++)
            {
                if (todo_q[q1])
                {
                    mcplan.j_q[q1] = mcplan.nj;
                    _mcplan.nj += 1;
                    int q2;
                    for (q2 = q1 + 1; q2 < mcplan.nq; q2++)
                    {
                        if (mcplan.i_q[q1] == mcplan.i_q[q2])
                        {
                            if (q1 == 0)
                            {
                                if (mcplan.t_q[q2] - mcplan.t_q[q2 - 1] == mcplan.t_q[q1] - grid.today)
                                    mcplan.j_q[q2] = mcplan.j_q[q1];
                            }
                            else
                            {
                                if (mcplan.t_q[q2] - mcplan.t_q[q2 - 1] == mcplan.t_q[q1] - mcplan.t_q[q1 - 1])
                                    mcplan.j_q[q2] = mcplan.j_q[q1];
                            }
                        }
                    }
                }
            }

            CArray.calloc(ref _mcplan.t0_j, mcplan.nj);
            CArray.calloc(ref _mcplan.t1_j, mcplan.nj);
            CArray.calloc(ref _mcplan.i_j, mcplan.nj);
            for (int q = 0; q < mcplan.nq; q++)
            {
                if (todo_q[q])
                {
                    int j = mcplan.j_q[q];
                    if (q == 0) mcplan.t0_j[0] = grid.today;
                    else
                    {
                        mcplan.t0_j[j] = mcplan.t_q[q - 1];
                    }
                    mcplan.t1_j[j] = mcplan.t_q[q];
                    mcplan.i_j[j] = mcplan.i_q[q];
                }
            }

            CArray.calloc(ref _mcplan.host_d_dt_j, mcplan.nj);
            CArray.calloc(ref _mcplan.niter_j, mcplan.nj);
           
            if (grid.fpu == EFloatingPointUnit.device)
            {
                host_d_gen();
            }
            if (grid.fpu == EFloatingPointUnit.host)
            {
                if (_host_d_gen_yy_i == null) throw new System.Exception();
            }
            
            for (int j = 0; j < mcplan.nj; j++)
            {
                i = mcplan.i_j[j];
                double maxdiag = 0.0;
                for (int y1 = 0; y1 < d; y1++)
                {
                    if (maxdiag < Math.Abs(_host_d_gen_yy_i[i * d * d + y1 + d * y1]))
                        maxdiag = Math.Abs(_host_d_gen_yy_i[i * d * d + y1 + d * y1]);
                }
                double dt; dt = 0.5 / maxdiag;
                if (dt > dt0) dt = dt0;

                double DeltaT;
                DeltaT = (mcplan.t1_j[j] - mcplan.t0_j[j]).Days / 365.25;
                if (DeltaT == 0) throw new System.Exception();

                mcplan.niter_j[j] = (int)Math.Ceiling(Math.Log(DeltaT / dt) / Math.Log(2.0));
                mcplan.host_d_dt_j[j] = DeltaT * Math.Pow(2.0, -(mcplan.niter_j[j]));
            }

            _mcplan.nr = 0;
            for (int j = 0; j < mcplan.nj; j++)
            {
                if (mcplan.niter_j[j] > mcplan.nr) _mcplan.nr = mcplan.niter_j[j];
            }

            CArray.calloc(ref _mcplan.njtodo_r, mcplan.nr);
            CArray.calloc(ref _mcplan.tocopy_j, mcplan.nj);

            for (int j = 0; j < mcplan.nj; j++)
            {
                if (mcplan.niter_j[j] % 2 == 1) mcplan.tocopy_j[j] = 1;
            }

            bool[] todo_k = new bool[mcplan.nk];
            for (int k1 = 0; k1 < mcplan.nk; k1++)
            {
                todo_k[k1] = true;
            }

            for (int k1 = 0; k1 < mcplan.nk; k1++)
            {
                int dt1;
                int i1, ii1;
                for (i1 = 0; i1 < ni; i1++) if (t_i[i1] >= mcplan.t_k[k1]) break;
                if (k1 == 0)
                {
                    dt1 = (mcplan.t_k[k1] - grid.today).Days;
                    ii1 = 0;
                }
                else
                {
                    dt1 = (mcplan.t_k[k1] - mcplan.t_k[k1 - 1]).Days;
                    for (ii1 = 0; ii1 < ni; ii1++)
                        if (t_i[ii1] >= mcplan.t_k[k1 - 1]) break;
                }

                if (i1 == ii1)
                {
                    for (int k2 = k1 + 1; k2 < mcplan.nk; k2++)
                    {
                        int dt2, i2, ii2;
                        for (i2 = 0; i2 < ni; i2++)
                            if (t_i[i2] >= mcplan.t_k[k2]) break;
                        if (k2 == 0)
                        {
                            dt2 = (mcplan.t_k[k2] - grid.today).Days;
                            ii2 = 0;
                        }
                        else
                        {
                            dt2 = (mcplan.t_k[k2] - mcplan.t_k[k2 - 1]).Days;
                            for (ii2 = 0; ii2 < ni; ii2++) if (t_i[ii2] >= mcplan.t_k[k2 - 1]) break;
                        }
                        if ((i2 == ii2) && (i1 == i2) && (dt1 == dt2)) todo_k[k2] = false;
                    }
                }
            }


            CArray.calloc(ref _mcplan.m_k, mcplan.nk);
            _mcplan.nm = 0;
            for (int k = 0; k < mcplan.nk; k++)
            {
                if (todo_k[k])
                {
                    mcplan.m_k[k] = mcplan.nm;
                    _mcplan.nm += 1;
                }
            }

            for (int k1 = 0; k1 < mcplan.nk; k1++)
            {
                if (todo_k[k1])
                {
                    int dt1, i1, ii1;
                    for (i1 = 0; i1 < ni; i1++) if (t_i[i1] >= mcplan.t_k[k1]) break;
                    if (k1 == 0)
                    {
                        dt1 = (mcplan.t_k[k1] - grid.today).Days;
                        ii1 = 0;
                    }
                    else
                    {
                        dt1 = (mcplan.t_k[k1] - mcplan.t_k[k1 - 1]).Days;
                        for (ii1 = 0; ii1 < ni; ii1++)
                            if (t_i[ii1] >= mcplan.t_k[k1 - 1]) break;
                    }

                    if (i1 == ii1)
                    {
                        for (int k2 = k1 + 1; k2 < mcplan.nk; k2++)
                        {
                            int dt2, i2, ii2;
                            for (i2 = 0; i2 < ni; i2++)
                                if (t_i[i2] >= mcplan.t_k[k2]) break;
                            if (k2 == 0)
                            {
                                dt2 = (mcplan.t_k[k2] - grid.today).Days;
                                ii2 = 0;
                            }
                            else
                            {
                                dt2 = (mcplan.t_k[k2] - mcplan.t_k[k2 - 1]).Days;
                                for (ii2 = 0; ii2 < ni; ii2++) if (t_i[ii2] >= mcplan.t_k[k2 - 1]) break;
                            }
                            if ((i2 == ii2) && (i1 == i2) && (dt1 == dt2))
                            {
                                mcplan.m_k[k2] = mcplan.m_k[k1];
                            }
                        }
                    }
                }
            }


            if (grid.fpu == EFloatingPointUnit.device)
            {
                _mcplan.device_m_k = new CArray(mcplan.nk, EType.int_t, EMemorySpace.device, this, "_mcplan.device_m_k");
                CArray.copy(ref _mcplan.device_m_k, mcplan.m_k);
            }

            _mcplan.jfactor_ms = new int[mcplan.nm][];

            for (int k = 0; k < mcplan.nk; k++)
            {
                int q;
                int ns, s;
                ns = 0;
                int t0, t1;
                t0 = 0; if (k > 0) t0 = (t_k[k - 1] - grid.today).Days;
                t1 = (t_k[k] - grid.today).Days;

                for (q = 0; q < mcplan.nq; q++)
                {
                    int tq0, tq1;
                    tq0 = 0; if (q > 0) tq0 = (mcplan.t_q[q - 1] - grid.today).Days;
                    tq1 = (mcplan.t_q[q] - grid.today).Days;
                    if (tq0 >= t0 && tq1 <= t1) ns += 1;
                }

                int m = mcplan.m_k[k];
                bool done;
                if (mcplan.jfactor_ms[m] == null)
                {
                    done = false;
                    mcplan.jfactor_ms[m] = new int[ns];
                }
                else done = true;

                s = 0;
                for (q = 0; q < mcplan.nq; q++)
                {
                    int tq0, tq1;
                    tq0 = 0; if (q > 0) tq0 = (mcplan.t_q[q - 1] - grid.today).Days;
                    tq1 = (mcplan.t_q[q] - grid.today).Days;
                    if (tq0 >= t0 && tq1 <= t1)
                    {
                        if (!done)
                        {
                            mcplan.jfactor_ms[m][s] = mcplan.j_q[q];
                        }
                        else
                        {
                            if (mcplan.jfactor_ms[m][s] != mcplan.j_q[q]) throw new System.Exception("internal error");
                        }
                        s += 1;
                    }
                }
            }

            _mcplan.niter_m = new int[mcplan.nm];
            _mcplan.ns = 0;
            for (int m = 0; m < mcplan.nm; m++)
            {
                if (mcplan.jfactor_ms[m].Length - 1 > mcplan.ns) _mcplan.ns = mcplan.jfactor_ms[m].Length - 1;
                mcplan.niter_m[m] = mcplan.jfactor_ms[m].Length;
            }



            int maxnjnm = Math.Max(mcplan.nj, mcplan.nm);
            if (grid.fpu == EFloatingPointUnit.device)
            {
                mcbuf._device_xtpk_yy_j = new CArray(d * d * mcplan.nj, EType.float_t, EMemorySpace.device, this, "mcbuf._device_xtpk_yy_j");
                mcbuf._device_xtpkbuf_yy_idx = new CArray(d * d * maxnjnm, EType.float_t, EMemorySpace.device, this, "mcbuf._device_xtpkbuf_yy_idx");
            }
            if (grid.fpu == EFloatingPointUnit.host)
            {
                mcbuf.host_d_xtpk_yy_j = new double[d * d * mcplan.nj];
                mcbuf.host_d_xtpkbuf_yy_idx = new double[d * d * maxnjnm];
            }

            if (grid.fpu == EFloatingPointUnit.device)
            {
                mcbuf._device_tpk_yy_m = new CArray(d * d * mcplan.nm, EType.float_t, EMemorySpace.device, this, "mcbuf._device_tpk_yy_m");
                mcbuf._device_ctpk_yy_m = new CArray(d * d * mcplan.nm, EType.float_t, EMemorySpace.device, this, "mcbuf._device_tpk_yy_m");
            }         

            if (grid.fpu == EFloatingPointUnit.device)
            {
                uint[] A_idxr = new uint[mcplan.nr * mcplan.nj];
                uint[] C_idxr = new uint[mcplan.nr * mcplan.nj];
                _mcplan.device_A_idxr = new CArray(mcplan.nj * mcplan.nj, EType.uint_t, EMemorySpace.device, this, "_mcplan.device_A_idxr");
                _mcplan.device_C_idxr = new CArray(mcplan.nj * mcplan.nj, EType.uint_t, EMemorySpace.device, this, "_mcplan.device_C_idxr");
                for (int r = 0; r < mcplan.nr; r++)
                {
                    int idx;

                    idx = 0;
                    for (int j = 0; j < mcplan.nj; j++)
                    {
                        if (r < mcplan.niter_j[j])
                        {
                            mcplan.njtodo_r[r] += 1;
                            A_idxr[idx + r * mcplan.nj] = (uint)(mcbuf._device_xtpk_yy_j.ptr + j * d * d * mcbuf._device_xtpk_yy_j.Size_of_one);
                            C_idxr[idx + r * mcplan.nj] = (uint)(mcbuf._device_xtpkbuf_yy_idx.ptr + j * d * d * mcbuf._device_xtpkbuf_yy_idx.Size_of_one);
                            idx += 1;
                        }
                    }

                    if (++r == mcplan.nr) break;
                    idx = 0;
                    for (int j = 0; j < mcplan.nj; j++)
                    {
                        if (r < mcplan.niter_j[j])
                        {
                            mcplan.njtodo_r[r] += 1;
                            A_idxr[idx + r * mcplan.nj] = (uint)(mcbuf._device_xtpkbuf_yy_idx.ptr + j * d * d * mcbuf._device_xtpk_yy_j.Size_of_one);
                            C_idxr[idx + r * mcplan.nj] = (uint)(mcbuf._device_xtpk_yy_j.ptr + j * d * d * mcbuf._device_xtpkbuf_yy_idx.Size_of_one);
                            idx += 1;
                        }
                    }

                }
                CArray.copy(ref _mcplan.device_A_idxr, A_idxr);
                CArray.copy(ref _mcplan.device_C_idxr, C_idxr);
                uint[] A_idxs = new uint[mcplan.ns * mcplan.nm];
                uint[] B_idxs = new uint[mcplan.ns * mcplan.nm];
                uint[] C_idxs = new uint[mcplan.ns * mcplan.nm];
                _mcplan.device_A_idxs = new CArray(mcplan.nm * mcplan.nm, EType.uint_t, EMemorySpace.device, this, "_mcplan.device_A_idxs");
                _mcplan.device_B_idxs = new CArray(mcplan.nm * mcplan.nm, EType.uint_t, EMemorySpace.device, this, "_mcplan.device_B_idxs");
                _mcplan.device_C_idxs = new CArray(mcplan.nm * mcplan.nm, EType.uint_t, EMemorySpace.device, this, "_mcplan.device_C_idxs");
                _mcplan.nmtodo_s = new int[mcplan.ns];

                for (int s = 0; s < mcplan.ns; s++)
                {
                    int idx;
                    idx = 0;
                    for (int m = 0; m < mcplan.nm; m++)
                    {
                        int j;
                        j = mcplan.jfactor_ms[m][s];

                        if (s % 2 == 0)
                        {
                            if (s < mcplan.niter_m[m])
                            {
                                mcplan.nmtodo_s[s] += 1;
                                A_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_tpk_yy_m.ptr + m * d * d * mcbuf._device_tpk_yy_m.Size_of_one);
                                B_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_xtpk_yy_j.ptr + j * d * d * mcbuf._device_xtpk_yy_j.Size_of_one);
                                C_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_xtpkbuf_yy_idx.ptr + m * d * d * mcbuf._device_xtpkbuf_yy_idx.Size_of_one);
                                idx += 1;
                            }
                        }
                        else
                        {
                            if (s < mcplan.niter_m[m])
                            {
                                mcplan.nmtodo_s[s] += 1;
                                A_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_xtpkbuf_yy_idx.ptr + m * d * d * mcbuf._device_xtpkbuf_yy_idx.Size_of_one);
                                B_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_xtpk_yy_j.ptr + j * d * d * mcbuf._device_xtpk_yy_j.Size_of_one);
                                C_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_tpk_yy_m.ptr + m * d * d * mcbuf._device_tpk_yy_m.Size_of_one);
                                idx += 1;
                            }
                        }
                    }

                    if (++s == mcplan.ns) break;
                    idx = 0;
                    for (int m = 0; m < mcplan.nm; m++)
                    {
                        int j;
                        j = mcplan.jfactor_ms[m][s];

                        if (s % 2 == 0)
                        {
                            if (s < mcplan.niter_m[m])
                            {
                                mcplan.nmtodo_s[s] += 1;
                                A_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_xtpkbuf_yy_idx.ptr + m * d * d * mcbuf._device_xtpkbuf_yy_idx.Size_of_one);
                                B_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_xtpk_yy_j.ptr + j * d * d * mcbuf._device_xtpk_yy_j.Size_of_one);
                                C_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_tpk_yy_m.ptr + m * d * d * mcbuf._device_tpk_yy_m.Size_of_one);
                                idx += 1;
                            }
                        }
                        else
                        {
                            if (s < mcplan.niter_m[m])
                            {
                                mcplan.nmtodo_s[s] += 1;
                                A_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_tpk_yy_m.ptr + m * d * d * mcbuf._device_tpk_yy_m.Size_of_one);
                                B_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_xtpk_yy_j.ptr + j * d * d * mcbuf._device_xtpk_yy_j.Size_of_one);
                                C_idxs[idx + s * mcplan.nm] = (uint)(mcbuf._device_xtpkbuf_yy_idx.ptr + m * d * d * mcbuf._device_xtpkbuf_yy_idx.Size_of_one);
                                idx += 1;
                            }
                        }
                    }

                }

                CArray.copy(ref _mcplan.device_A_idxs, A_idxs);
                CArray.copy(ref _mcplan.device_B_idxs, B_idxs);
                CArray.copy(ref _mcplan.device_C_idxs, C_idxs);
            }


            if (grid.fpu == EFloatingPointUnit.device)
            {
                int nth = 1;
                mcplan.nth = nth;
                mcbuf._device_m_k = new CArray(mcplan.nk, EType.int_t, EMemorySpace.device, this, "mcbuf._device_m_k");
                CArray.copy(ref mcbuf._device_m_k, mcplan.m_k);
                mcbuf.host_y_sk_th = new short[mcplan.nk * mcplan._nscen_per_batch];
            }


            if (grid.fpu == EFloatingPointUnit.host)
            {
                int nth = System.Environment.ProcessorCount;
                mcplan.nth = nth;
                mcbuf.host_unif_scen_th = new uint[mcplan._nscen_per_batch * nth];
                mcbuf.host_y_sk_th = new short[mcplan.nk * mcplan._nscen_per_batch * nth];
            }

        }






        public void exe_mc_plan()
        {
            if (grid.fpu == EFloatingPointUnit.host)
            {                
                host_d_mc_fex();
                host_d_mc_minit();
                for (int s = 0; s < mcplan.ns; s++) host_d_mc_mk(s);
                host_d_mc_mcopy();
                host_d_mc_hash();
            }
            else
            {
                device_mc_sfex();
                device_mc_sminit();
                device_mc_smk();
                device_mc_smcopy();
                device_mc_sck();
            }
        }

      
       public void host_memfree(IntPtr hptr)
        {
            if ((long)hptr != 0)
            {
                opcuda_mem_free_host(hptr);
                int status = opcuda_get_status();
                //if (status != 0) throw new System.Exception();
            }
        }

       public void device_memfree(uint ptr)
       {
           if (ptr != 0)
           {
               opcuda_mem_free_device(ptr);
               int status = opcuda_get_status();
               //if (status != 0) throw new System.Exception();
           }
       }     


    }
}
