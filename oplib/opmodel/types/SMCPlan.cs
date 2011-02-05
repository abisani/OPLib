// SMCPlan.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
    public class SMCPlan
    {
        public int _nscen;
        public int _nscen_per_batch;
        public int _nbatches;
        public int nk; 
        public int nq;
        public int nj;
        public int nm;
        public DateTime[] t_k;
        public int[] j_k;
        public DateTime[] t0_j;
        public DateTime[] t1_j;
        public double[] host_d_dt_j;
        public float[] host_s_dt_j;
        public int[] i_j;
        public int[] i_m;
        public bool[] todo_mq;
        public int[][] jfactor_ms;
        public DateTime[] t_q;
        public int[] i_q;
        public int[] j_q;
        public int[] m_k;
        public int nr;
        public int[] niter_j;
        public int[] njtodo_r;
        public CArray device_A_idxr;
        public CArray device_C_idxr;
        public int[] tocopy_j;
        public int ns;
        public int[] niter_m;
        public int[] nmtodo_s;
        public CArray device_A_idxs;
        public CArray device_B_idxs;
        public CArray device_C_idxs;
        public CArray device_m_k;
        public int[] tocopy_m;
        public int nth;

        public SMCPlan()
        {
        }

    };
}
