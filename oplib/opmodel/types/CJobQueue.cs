// CJobQueue.cs --- Part of the project OPLib 1.0, a high performance pricing library
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
using System.Diagnostics;


namespace OPModel.Types
{

    public class SJobQueue
    {
        public Object[] p_j;
        public Object[] ret_j;
        public int nj;
        public int[] th_j;
        public bool[] submitted_j;
        public bool[] completed_j;
        public bool[] outputted_j;
    }


    public class CJobQueue
    {
        public delegate Object func(Object p, int th);
        public delegate void output(Object p, int th);
        private func _f;
        private output _output;
        private SJobQueue _job;
        private int _nth;
        private bool _done;

        public Object[] Exec(func f, output output, Object[] p_j, int nth)
        {
            _f = f;
            _output = output;
            _nth = nth;
            _job = new SJobQueue();
            _job.p_j = p_j;
            _job.nj = p_j.Length;
            _job.submitted_j = new bool[_job.nj];
            _job.completed_j = new bool[_job.nj];
            _job.outputted_j = new bool[_job.nj];
            _job.th_j = new int[_job.nj];
            _job.ret_j = new Object[_job.nj];
            for (int j = 0; j < _job.nj; j++) _job.submitted_j[j] = false;
            for (int j = 0; j < _job.nj; j++) _job.completed_j[j] = false;
            for (int j = 0; j < _job.nj; j++) _job.outputted_j[j] = false;

            System.Threading.Thread[] thread_th = new System.Threading.Thread[nth];

            for (int th = 0; th < nth; th++)
            {
                thread_th[th] = new System.Threading.Thread(Iterate);
                thread_th[th].Name = "CJobQueue Worker Thread " + th;
                thread_th[th].Start();
                System.Threading.Thread.Sleep(10);
            }

            Output();
            for (int th = 0; th < nth; th++)
            {
                thread_th[th].Join();
            }
            return _job.ret_j;
        }



        void Output()
        {
            if (_output == null) return;
            bool flag_exit;
            bool[] todo_j = new bool[_job.nj];

            while (true)
            {

                for (int j = 0; j < _job.nj; j++)
                {
                    todo_j[j] = false;
                }

                flag_exit = true;

                lock (_job)
                {
                    for (int j = 0; j < _job.nj; j++)
                    {
                        if (_job.completed_j[j])
                        {
                            if (!_job.outputted_j[j])
                            {
                                todo_j[j] = true;
                            }
                        }
                        else
                        {
                            flag_exit = false;
                        }
                    }
                    System.Threading.Monitor.Pulse(_job);
                    System.Threading.Thread.Sleep(0);
                }

                for (int j = 0; j < _job.nj; j++)
                {
                    if (todo_j[j])
                    {
                        _output(_job.ret_j[j], _job.th_j[j]);
                        _job.outputted_j[j] = true;
                    }
                }

                if (flag_exit)
                {
                    break;
                }
                else
                {
                    System.Threading.Thread.Sleep(100);
                }
            } //end while
        }


        public CJobQueue()
        {
        }



        public bool done
        {
            get { return _done; }
        }

        public int nth
        {
            get { return _nth; }
        }


        void Iterate()
        {

            int id = System.Convert.ToInt32(System.Threading.Thread.CurrentThread.Name.Substring(24));
            int j;

            while (true)
            {
                if (done) break;

                //get next job
                lock (_job)
                {
                    for (j = 0; (j < _job.nj); j++)
                        if (!_job.submitted_j[j]) break;
                    if (j == _job.nj)
                    {
                        _done = true;
                        return;
                    }
                    _job.submitted_j[j] = true;
                    _job.th_j[j] = id;
                    System.Threading.Monitor.Pulse(_job);
                    System.Threading.Thread.Sleep(0);
                }

                Object ret = _f(_job.p_j[j], id);

                //process result

                lock (_job)
                {
                    _job.completed_j[j] = true;
                    _job.ret_j[j] = ret;
                    System.Threading.Monitor.Pulse(_job);
                    System.Threading.Thread.Sleep(0);
                }

            }
        }


    }
}