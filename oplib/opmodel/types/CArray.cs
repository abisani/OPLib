// CArray.cs --- Part of the project OPLib 1.0, a high performance pricing library
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


namespace OPModel
{
    public class CArray
    {
        private CModel _model;
        private IntPtr _hptr;
        private uint _ptr;
        private string _name;
        private int _length;
        private int _sz;
        private EType _type;
        private EMemorySpace _side;
        private bool _suballocated;
        private int _offset;
        private int _number_of_internal_buffers;
        private int[] _internal_buffer_pointers;
        private string[] _internal_buffer_names;

        private CArray(CModel model)
        {
            _model = model;
            _suballocated = false;
            _number_of_internal_buffers = 0;
            _internal_buffer_pointers = null;
        }

        public CArray(CArray buf, uint offset_in_bytes, int sz, CModel model, string name)
        {
            if (buf == null) throw new System.Exception();
            if (sz > buf._sz - offset_in_bytes) throw new System.Exception();
            this._ptr = buf._ptr + offset_in_bytes;
            this._length = 0;
            this._sz = sz;
            this._side = buf._side;
            _model = model;
            _suballocated = false;
            _number_of_internal_buffers = 0;
            _internal_buffer_pointers = null;
            _name = name;
        }


        public CArray(int n, EType type, EMemorySpace side, CModel model, string name)
        {
            alloc(n, type, side, model);
            _model = model;
            _suballocated = false;
            _number_of_internal_buffers = 0;
            _internal_buffer_pointers = null;
            _name = name;
        }



        CModel model
        {
            get
            {
                return _model;
            }
        }


        static private int sizeof_type(EType type)
        {
            if (type == EType.byte_t) return 1;
            if (type == EType.short_t) return 2;
            if (type == EType.float_t) return 4;
            if (type == EType.int_t) return 4;
            if (type == EType.uint_t) return 4;
            if (type == EType.double_t) return 8;
            throw new System.Exception();
        }


        public void change_type(EType type)
        {
            this._type = type;
            this._length = 0;
        }


        public int Size_of_one
        {
            get
            {
                if (_type == EType.byte_t) return 1;
                if (_type == EType.short_t) return 2;
                if (_type == EType.float_t) return 4;
                if (_type == EType.int_t) return 4;
                if (_type == EType.uint_t) return 4;
                if (_type == EType.double_t) return 8;
                throw new System.Exception();
            }
        }

        public EMemorySpace side
        {
            get
            {
                return _side;
            }
        }

        public EType type
        {
            get
            {
                return _type;
            }
        }

        public uint ptr
        {
            get
            {
                return _ptr;
            }
        }

        public IntPtr hptr
        {
            get
            {
                return _hptr;
            }
        }

        public int length
        {
            get
            {
                return _length;
            }
        }


        public int sz
        {
            get
            {
                return _sz;
            }
        }


        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_memcpy_h2d(uint dptr, IntPtr hptr, uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_memcpy_d2h(uint dptr, IntPtr hptr, uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_get_status();

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private uint opcuda_mem_alloc(uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_ssetone(uint xPtr, int n, float c, int i);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private IntPtr opcuda_mem_alloc_host(uint sz);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mem_free_device(uint dptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private void opcuda_mem_free_host(IntPtr hptr);

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe private int opcuda_memcpy_d2d(uint dptr2, uint dptr1, uint sz);


        static public void setall(ref double[] buf, double c)
        {
            for (int i = 0; i < buf.Length; i++)
            {
                buf[i] = c;
            }
        }


        static public void setall(ref CArray buf, double c)
        {
            if (buf.side == EMemorySpace.device)
            {
                buf.model.device_setall(ref buf, (float)c);
            }
            else
            {
                unsafe
                {
                    float* bufp = (float*)buf.ptr;
                    float cf = (float)c;
                    for (int i = 0; i < buf.length; i++)
                    {
                        bufp[i] = cf;
                    }
                }
            }
        }



        static public void setone(ref CArray buf, double c, int i)
        {
            if (buf.side == EMemorySpace.device)
            {
                opcuda_ssetone(buf.ptr, buf.length, (float)c, 1);
            }
            else
            {
                unsafe
                {
                    float* bufp = (float*)buf.ptr;
                    float cf = (float)c;
                    bufp[i] = cf;
                }
            }
        }


        static public int round(int n, EType type)
        {
            return (int)(128 * Math.Ceiling((double)n * sizeof_type(type) / 128)) / sizeof_type(type);
        }

        static public void suballoc(ref CArray target, CArray buf, int n, EType type, EMemorySpace side, string name)
        {
            if (target != null) throw new System.Exception();
            if (buf._internal_buffer_pointers == null)
            {
                buf._internal_buffer_pointers = new int[1000];
                for (int i = 0; i < 1000; i++)
                {
                    buf._internal_buffer_pointers[i] = -1;
                }
            }

            if (buf._internal_buffer_names == null)
            {
                buf._internal_buffer_names = new string[1000];
                for (int i = 0; i < 1000; i++)
                {
                    buf._internal_buffer_names[i] = null;
                }
            }

            if (buf == null) throw new System.Exception();
            target = new CArray(buf.model);
            target._sz = round(n, type) * sizeof_type(type);
            if (buf.sz < buf._offset + target._sz) throw new System.Exception();
            target._suballocated = true;
            target._ptr = (uint)(buf._ptr + buf._offset);
            target._name = name;
            buf._internal_buffer_pointers[buf._number_of_internal_buffers] = (int)target._ptr;
            buf._internal_buffer_names[buf._number_of_internal_buffers] = target._name;
            buf._number_of_internal_buffers += 1;
            target._side = side;
            target._type = type;
            target._length = n;
            target._name = name;
            buf._offset += target._sz;
            if (target.ptr - (target.ptr / 128) * 128 > 0) throw new System.Exception();

        }


        static public void subfree(ref CArray target, CArray buf)
        {
            if (target == null) throw new System.Exception();
            if (buf == null) throw new System.Exception();
            if (target.ptr + target.sz != buf.ptr + buf._offset) throw new System.Exception();
            if (target.ptr != buf._internal_buffer_pointers[buf._number_of_internal_buffers - 1])
                throw new System.Exception();
            buf._offset -= target._sz;
            buf._internal_buffer_pointers[--buf._number_of_internal_buffers] = -1;
            buf._internal_buffer_names[buf._number_of_internal_buffers] = null;
            target = null;
        }


        static public void alloc(ref CArray buf, int n, EType type, EMemorySpace side, CModel model, string name)
        {

            if (buf == null)
            {
                buf = new CArray(n, type, side, model, name);
                return;
            }
            if (buf._suballocated) throw new System.Exception();

            if (buf.sz < n * sizeof_type(type))
            {
                if (model != buf._model) throw new System.Exception();
                buf = new CArray(n, type, side, model, name);
                return;
            }
            if (model != buf._model) throw new System.Exception();
        }


        static public void alloc(ref double[,] a, int m, int n)
        {
            if (m == 0) return;
            if (n == 0) return;
            if (a == null)
            {
                a = new double[m, n];
                return;
            }
            if (a.GetLength(0) != m || a.GetLength(1) != n)
            {
                a = new double[m, n];
            }
        }

        static public void alloc(ref int[, ,] a, int m, int n, int k)
        {
            if (m == 0) return;
            if (n == 0) return;
            if (k == 0) return;
            if (a == null)
            {
                a = new int[m, n, k];
                return;
            }
            if (a.GetLength(0) != m || a.GetLength(1) != n || a.GetLength(2) != k)
            {
                a = new int[m, n, k];
            }
        }

        static public void alloc(ref int[,] a, int m, int n)
        {
            if (m == 0) return;
            if (n == 0) return;
            if (a == null)
            {
                a = new int[m, n];
                return;
            }
            if (a.GetLength(0) != m || a.GetLength(1) != n)
            {
                a = new int[m, n];
            }
        }

        static public void alloc(ref double[] a, int n)
        {
            if (n == 0) return;
            if (a == null)
            {
                a = new double[n];
                return;
            }
            if (a.Length != n)
            {
                a = new double[n];
            }
        }


        static public void alloc(ref float[] a, int n)
        {
            if (n == 0) return;
            if (a == null)
            {
                a = new float[n];
                return;
            }
            if (a.Length != n)
            {
                a = new float[n];
            }
        }

        static public void alloc(ref int[] a, int n)
        {
            if (n == 0) return;
            if (a == null)
            {
                a = new int[n];
                return;
            }
            if (a.Length != n)
            {
                a = new int[n];
            }
        }

        static public void calloc(ref double[] a, int n)
        {
            if (n == 0) return;
            if (a == null)
            {
                a = new double[n];
                return;
            }
            if (a.Length != n)
            {
                a = new double[n];
                return;
            }
            if (a.Length == n)
            {
                for (int i = 0; i < n; i++)
                {
                    a[i] = 0;
                }
            }
        }


        static public void calloc(ref float[] a, int n)
        {
            if (n == 0) return;
            if (a == null)
            {
                a = new float[n];
                return;
            }
            if (a.Length != n)
            {
                a = new float[n];
                return;
            }
            if (a.Length == n)
            {
                for (int i = 0; i < n; i++)
                {
                    a[i] = 0;
                }
            }
        }


        static public void calloc(ref double[,] a, int m, int n)
        {
            if (n == 0 && m == 0) return;
            if (a == null)
            {
                a = new double[m, n];
                return;
            }
            if (a.GetLength(0) != m || a.GetLength(1) != n)
            {
                a = new double[m, n];
                return;
            }
            if (a.GetLength(0) == m && a.GetLength(1) == n)
            {
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        a[i, j] = 0;
                    }
                }
            }
        }

        static public void calloc(ref DateTime[] a, int n)
        {
            if (n == 0) return;
            if (a == null)
            {
                a = new DateTime[n];
                return;
            }
            if (a.Length != n)
            {
                a = new DateTime[n];
                return;
            }
            DateTime t0 = new DateTime();
            if (a.Length == n)
            {
                for (int i = 0; i < n; i++)
                {
                    a[i] = t0;
                }
            }
        }

        static public void calloc(ref int[] a, int n)
        {
            if (n == 0) return;
            if (a == null)
            {
                a = new int[n];
                return;
            }
            if (a.Length != n)
            {
                a = new int[n];
            }
            if (a.Length == n)
            {
                for (int i = 0; i < n; i++)
                {
                    a[i] = 0;
                }
            }
        }




        private void alloc(int n, EType type, EMemorySpace side, CModel model)
        {
            if (n == 0) return;
            if (_model == null)
            {
                _model = model;
            }
            else
            {
                if (_model != model) throw new System.Exception();
            }

            this._type = type;
            this._side = side;
            if (_sz == 0)
            {
                if (side == EMemorySpace.device)
                {
                    if (_suballocated) throw new System.Exception();
                    _ptr = opcuda_mem_alloc((uint)(Size_of_one * n));
                }
                else
                {
                    if (_suballocated) throw new System.Exception();
                    _hptr = opcuda_mem_alloc_host((uint)(Size_of_one * n));
                }

                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();
                _length = n;
                _sz = Size_of_one * n;
                return;
            }
            if (_sz < Size_of_one * n)
            {
                if (side == EMemorySpace.device)
                {
                    if (_suballocated) throw new System.Exception();
                    opcuda_mem_free_device(ptr);
                    _ptr = opcuda_mem_alloc((uint)(Size_of_one * n));
                }
                else
                {
                    if (_suballocated) throw new System.Exception();
                    opcuda_mem_free_host(hptr);
                    _hptr = opcuda_mem_alloc_host((uint)(Size_of_one * n));
                }

                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();
                _sz = Size_of_one * n;
                _length = n;
                return;
            }
            _length = n;
        }


        static public void copy(ref double[,] destination, double[,] source)
        {
            if (source == null)
            {
                destination = null;
                return;
            }
            alloc(ref destination, source.GetLength(0), source.GetLength(1));
            for (int i = 0; i < source.GetLength(0); i++)
            {
                for (int j = 0; j < source.GetLength(1); j++)
                {
                    destination[i, j] = source[i, j];
                }
            }
        }

        static public void copy(ref int[,] destination, int[,] source)
        {
            if (source == null)
            {
                destination = null;
                return;
            }
            alloc(ref destination, source.GetLength(0), source.GetLength(1));
            for (int i = 0; i < source.GetLength(0); i++)
            {
                for (int j = 0; j < source.GetLength(1); j++)
                {
                    destination[i, j] = source[i, j];
                }
            }
        }

        static public void copy(ref int[, ,] destination, int[, ,] source)
        {
            if (source == null)
            {
                destination = null;
                return;
            }
            alloc(ref destination, source.GetLength(0), source.GetLength(1), source.GetLength(2));
            for (int i = 0; i < source.GetLength(0); i++)
            {
                for (int j = 0; j < source.GetLength(1); j++)
                {
                    for (int k = 0; k < source.GetLength(2); k++)
                    {
                        destination[i, j, k] = source[i, j, k];
                    }
                }
            }
        }

        static public void copy(ref int[] destination, int[] source)
        {
            if (source == null)
            {
                destination = null;
                return;
            }
            alloc(ref destination, source.GetLength(0));
            for (int i = 0; i < source.GetLength(0); i++)
            {
                destination[i] = source[i];
            }
        }



        static public void copy(ref double[] destination, double[] source)
        {
            if (source == null)
            {
                destination = null;
                return;
            }
            alloc(ref destination, source.GetLength(0));
            for (int i = 0; i < source.GetLength(0); i++)
            {
                destination[i] = source[i];
            }
        }

        [System.Runtime.InteropServices.DllImport("opcuda")]
        static extern unsafe protected int opcuda_scopy1(uint destination, uint source, uint n);


        static public void copy(ref CArray destination, CArray source)
        {

            if (source._model != destination._model) throw new System.Exception();

            destination._type = source.type;
            destination.alloc(source.length, source.type, destination.side, source._model);

            if (source.side == EMemorySpace.device && destination.side == EMemorySpace.host)
            {
                opcuda_memcpy_d2h(source.ptr, destination.hptr, (uint)(source.Size_of_one * source.length));
            }
            if (source.side == EMemorySpace.device && destination.side == EMemorySpace.device)
            {
                if (source.Size_of_one == 4)
                {
                    opcuda_scopy1(destination.ptr, source.ptr, (uint)(source.length));
                }
                else
                {
                    opcuda_memcpy_d2d(destination.ptr, source.ptr, (uint)(source.Size_of_one * source.length));
                }
            }
            if (source.side == EMemorySpace.host && destination.side == EMemorySpace.device)
            {
                opcuda_memcpy_h2d(destination.ptr, source.hptr, (uint)(source.Size_of_one * source.length));
            }
            if (source.side == EMemorySpace.host && destination.side == EMemorySpace.host)
            {
                throw new System.Exception();
            }

            int status = opcuda_get_status();
            if (status != 0) throw new System.Exception();
        }


        static public void copy(ref float[] destination, CArray source)
        {

            if (source.type == EType.float_t)
            {
                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();
                if (destination.Length < source.length) destination = new float[source.length];
                unsafe
                {
                    fixed (float* destinationp = &destination[0])
                    {
                        opcuda_memcpy_d2h(source.ptr, (IntPtr)destinationp, (uint)(source.Size_of_one * source.length));
                    }
                }
                status = opcuda_get_status();
                if (status != 0) throw new System.Exception();
                return;
            }
            throw new System.Exception();
        }


        static public double mismatch(CArray a, double[] b, ref double maxerror, ref int imax)
        {
            if (a.type == EType.float_t)
            {
                imax = -1;
                float[] a1 = new float[a.length];
                copy(ref a1, a);
                double error;
                maxerror = 0;
                for (int i = 0; i < b.Length; i++)
                {
                    error = Math.Abs(a1[i] - b[i]);
                    if (error > maxerror)
                    {
                        maxerror = error;
                        imax = i;
                    }
                }
                return maxerror;
            }
            throw new System.Exception();
        }





        static public double mismatch(CArray a, float[] b, ref double maxerror, ref int imax)
        {
            if (a.type == EType.float_t)
            {
                imax = -1;
                float[] a1 = new float[a.length];
                copy(ref a1, a);
                double error;
                maxerror = 0;
                for (int i = 0; i < b.Length; i++)
                {
                    error = Math.Abs(a1[i] - b[i]);
                    if (error > maxerror)
                    {
                        maxerror = error;
                        imax = i;
                    }
                }
                return maxerror;
            }
            throw new System.Exception();
        }


        static public double mismatch(CArray a, int[] b, ref double maxerror, ref int imax)
        {
            if (a.type == EType.int_t)
            {
                imax = -1;
                int[] a1 = new int[a.length];
                copy(ref a1, a);
                double error;
                maxerror = 0;
                for (int i = 0; i < b.Length; i++)
                {
                    error = Math.Abs(a1[i] - b[i]);
                    if (error > maxerror)
                    {
                        maxerror = error;
                        imax = i;
                    }
                }
                return maxerror;
            }
            throw new System.Exception();
        }

        static public void copy(ref uint[] destination, CArray source)
        {
            if (source.length == 0) return;

            if (source.type == EType.uint_t)
            {
                if (destination.Length < source.length) destination = new uint[source.length];
                unsafe
                {
                    fixed (uint* destinationp = &destination[0])
                    {
                        opcuda_memcpy_d2h(source.ptr, (IntPtr)destinationp, (uint)(source.Size_of_one * source.length));
                    }
                }
                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();
                return;
            }
            throw new System.Exception();
        }


        static public void copy(ref int[] destination, CArray source)
        {
            if (source.type == EType.int_t)
            {
                if (destination.Length < source.length) destination = new int[source.length];

                unsafe
                {
                    fixed (int* destinationp = &destination[0])
                    {
                        opcuda_memcpy_d2h(source.ptr, (IntPtr)destinationp, (uint)(source.Size_of_one * source.length));
                    }
                }
                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();
                return;
            }
            throw new System.Exception();
        }



        static public void copy(ref CArray destination, float[] source)
        {
            if (destination.side == EMemorySpace.device)
            {
                if (destination.type == EType.float_t)
                {
                    if (destination.length < source.Length) destination.alloc(source.Length, EType.float_t, EMemorySpace.device, destination._model);
                    destination._length = source.Length;
                    if (source.Length == 0) return;
                    unsafe
                    {
                        fixed (float* sourcep = &source[0])
                        {
                            opcuda_memcpy_h2d(destination.ptr, (IntPtr)sourcep, (uint)(source.Length * sizeof(float)));
                        }
                    }
                    int status = opcuda_get_status();
                    if (status != 0) throw new System.Exception();

                    return;
                }
            }
            throw new System.Exception();
        }

        static public void copy(ref CArray destination, int[] source)
        {
            if (destination.side == EMemorySpace.device)
            {
                if (destination.type == EType.int_t)
                {
                    if (destination.length < source.Length) destination.alloc(source.Length, EType.int_t, EMemorySpace.device, destination._model);
                    destination._length = source.Length;
                    if (source.Length == 0) return;
                    unsafe
                    {
                        fixed (int* sourcep = &source[0])
                        {
                            opcuda_memcpy_h2d(destination.ptr, (IntPtr)sourcep, (uint)(source.Length * sizeof(int)));
                        }
                    }
                    int status = opcuda_get_status();
                    if (status != 0) throw new System.Exception();
                    return;
                }
            }
            throw new System.Exception();
        }


        static public void copy(ref CArray destination, uint[] source)
        {
            if (source == null)
            {
                destination._length = 0;
                destination._type = EType.uint_t;
                return;
            }

            if (source.Length == 0)
            {
                destination._length = 0;
                destination._type = EType.uint_t;
                return;
            }

            if (destination.type == EType.uint_t)
            {
                if (destination.length < source.Length) destination.alloc(source.Length, EType.uint_t, EMemorySpace.device, destination._model);
                destination._length = source.Length;

                unsafe
                {
                    fixed (uint* sourcep = &source[0])
                    {
                        opcuda_memcpy_h2d(destination.ptr, (IntPtr)sourcep, (uint)(source.Length * sizeof(uint)));
                    }
                }

                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();

                return;
            }
            throw new System.Exception();
        }




        static public void copy(ref double[] destination, CArray source, CArray buf)
        {
            if (source.side == EMemorySpace.device)
            {
                if (EMemorySpace.device != source.side) throw new System.Exception();
                if (EMemorySpace.host != buf.side) throw new System.Exception();
            }

            if (source == null)
            {
                destination = null;
                return;
            }

            if (source.length == 0)
            {
                destination = new double[0];
                return;
            }

            if (source.type == EType.double_t)
            {
                alloc(ref destination, source.length);
                unsafe
                {
                    fixed (double* destinationp = &destination[0])
                    {
                        opcuda_memcpy_h2d(source.ptr, (IntPtr)destinationp, (uint)(source.length * CArray.sizeof_type(source.type)));
                    }
                }
                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();
                return;
            }

            if (source.type == EType.float_t)
            {
                if (buf == null) throw new System.Exception();
                if (buf.type != EType.float_t) throw new System.Exception();
                if (buf.length < source.length) throw new System.Exception();

                alloc(ref destination, source.length);

                unsafe
                {
                    opcuda_memcpy_d2h(source.ptr, buf.hptr, (uint)(source.length * CArray.sizeof_type(source.type)));
                    float* bufp = (float*)buf.ptr;
                    for (int a = 0; a < source.length; a++)
                    {
                        destination[a] = (double)bufp[a];
                    }
                }
                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();
                return;
            }

            throw new System.Exception();
        }



        static public void copy(ref CArray destination, short[] source)
        {

            if (destination.type == EType.short_t)
            {
                if (destination.length < source.Length) destination.alloc(source.Length, EType.short_t, EMemorySpace.device, destination._model);
                destination._length = source.Length;

                unsafe
                {
                    fixed (short* sourcep = &source[0])
                    {
                        opcuda_memcpy_h2d(destination.ptr, (IntPtr)sourcep, (uint)(source.Length * sizeof(short)));
                    }
                }

                int status = opcuda_get_status();
                if (status != 0) throw new System.Exception();

                return;
            }

            throw new System.Exception();
        }

        ~CArray()
        {
            if (_suballocated) return;
            if (_model != null)
            {
                if (side == EMemorySpace.device) _model.device_memfree(ptr);
                if (side == EMemorySpace.host) _model.host_memfree(hptr);
            }
        }

    }


}








