/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )
   Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )
   Copyright (c) 2020  Dario Mambro ( dario.mambro@gmail.com )

   Based on original fortran 77 code from FFTPACKv4 from NETLIB
   (http://www.netlib.org/fftpack), authored by Dr Paul Swarztrauber
   of NCAR, in 1985.

   As confirmed by the NCAR fftpack software curators, the following
   FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
   released under the same terms.

   FFTPACK license:

   http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

   Copyright (c) 2004 the University Corporation for Atmospheric
   Research ("UCAR"). All rights reserved. Developed by NCAR's
   Computational and Information Systems Laboratory, UCAR,
   www.cisl.ucar.edu.

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of NCAR's Computational and Information Systems
   Laboratory, the University Corporation for Atmospheric Research,
   nor the names of its sponsors or contributors may be used to
   endorse or promote products derived from this Software without
   specific prior written permission.

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
   SOFTWARE.


   PFFFT : a Pretty Fast FFT.

   This file is largerly based on the original FFTPACK implementation, modified in
   order to take advantage of SIMD instructions of modern CPUs.
*/

/* Intentionally included twice, with PFFFT_CPLX_ORDERED set to 0 and 1.
   Share the butterfly arithmetic while specializing only the input/output
   layout. Keeping the unordered body intact avoids compiler scheduling
   regressions in a loop that was already vectorized. */

PFFFT_CPLX_SCOPE void PFFFT_CPLX_FINALIZE(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  int k, dk = Ncvec/SIMD_SZ; /* number of 4x4 matrix blocks */
  v4sf r0, i0, r1, i1, r2, i2, r3, i3;
  v4sf sr0, dr0, sr1, dr1, si0, di0, si1, di1;
  assert(in != out);
  for (k=0; k < dk; ++k) {
    r0 = in[8*k+0]; i0 = in[8*k+1];
    r1 = in[8*k+2]; i1 = in[8*k+3];
    r2 = in[8*k+4]; i2 = in[8*k+5];
    r3 = in[8*k+6]; i3 = in[8*k+7];
    VTRANSPOSE4(r0,r1,r2,r3);
    VTRANSPOSE4(i0,i1,i2,i3);
    VCPLXMUL(r1,i1,e[k*6+0],e[k*6+1]);
    VCPLXMUL(r2,i2,e[k*6+2],e[k*6+3]);
    VCPLXMUL(r3,i3,e[k*6+4],e[k*6+5]);

    sr0 = VADD(r0,r2); dr0 = VSUB(r0, r2);
    sr1 = VADD(r1,r3); dr1 = VSUB(r1, r3);
    si0 = VADD(i0,i2); di0 = VSUB(i0, i2);
    si1 = VADD(i1,i3); di1 = VSUB(i1, i3);

    /*
      transformation for each column is:

      [1   1   1   1   0   0   0   0]   [r0]
      [1   0  -1   0   0  -1   0   1]   [r1]
      [1  -1   1  -1   0   0   0   0]   [r2]
      [1   0  -1   0   0   1   0  -1]   [r3]
      [0   0   0   0   1   1   1   1] * [i0]
      [0   1   0  -1   1   0  -1   0]   [i1]
      [0   0   0   0   1  -1   1  -1]   [i2]
      [0  -1   0   1   1   0  -1   0]   [i3]
    */

    r0 = VADD(sr0, sr1); i0 = VADD(si0, si1);
    r1 = VADD(dr0, di1); i1 = VSUB(di0, dr1);
    r2 = VSUB(sr0, sr1); i2 = VSUB(si0, si1);
    r3 = VSUB(dr0, di1); i3 = VADD(di0, dr1);

#if PFFFT_CPLX_ORDERED
    INTERLEAVE2(r0, i0, out[2*k], out[2*k+1]);
    INTERLEAVE2(r1, i1, out[2*(dk+k)], out[2*(dk+k)+1]);
    INTERLEAVE2(r2, i2, out[2*(2*dk+k)], out[2*(2*dk+k)+1]);
    INTERLEAVE2(r3, i3, out[2*(3*dk+k)], out[2*(3*dk+k)+1]);
#else
    *out++ = r0; *out++ = i0; *out++ = r1; *out++ = i1;
    *out++ = r2; *out++ = i2; *out++ = r3; *out++ = i3;
#endif
  }
}

PFFFT_CPLX_SCOPE void PFFFT_CPLX_PREPROCESS(int Ncvec, const v4sf *in, v4sf *out, const v4sf *e) {
  int k, dk = Ncvec/SIMD_SZ; /* number of 4x4 matrix blocks */
  v4sf r0, i0, r1, i1, r2, i2, r3, i3;
  v4sf sr0, dr0, sr1, dr1, si0, di0, si1, di1;
  assert(in != out);
  for (k=0; k < dk; ++k) {
#if PFFFT_CPLX_ORDERED
    UNINTERLEAVE2(in[2*k], in[2*k+1], r0, i0);
    UNINTERLEAVE2(in[2*(dk+k)], in[2*(dk+k)+1], r1, i1);
    UNINTERLEAVE2(in[2*(2*dk+k)], in[2*(2*dk+k)+1], r2, i2);
    UNINTERLEAVE2(in[2*(3*dk+k)], in[2*(3*dk+k)+1], r3, i3);
#else
    r0 = in[8*k+0]; i0 = in[8*k+1];
    r1 = in[8*k+2]; i1 = in[8*k+3];
    r2 = in[8*k+4]; i2 = in[8*k+5];
    r3 = in[8*k+6]; i3 = in[8*k+7];
#endif

    sr0 = VADD(r0,r2); dr0 = VSUB(r0, r2);
    sr1 = VADD(r1,r3); dr1 = VSUB(r1, r3);
    si0 = VADD(i0,i2); di0 = VSUB(i0, i2);
    si1 = VADD(i1,i3); di1 = VSUB(i1, i3);

    r0 = VADD(sr0, sr1); i0 = VADD(si0, si1);
    r1 = VSUB(dr0, di1); i1 = VADD(di0, dr1);
    r2 = VSUB(sr0, sr1); i2 = VSUB(si0, si1);
    r3 = VADD(dr0, di1); i3 = VSUB(di0, dr1);

    VCPLXMULCONJ(r1,i1,e[k*6+0],e[k*6+1]);
    VCPLXMULCONJ(r2,i2,e[k*6+2],e[k*6+3]);
    VCPLXMULCONJ(r3,i3,e[k*6+4],e[k*6+5]);

    VTRANSPOSE4(r0,r1,r2,r3);
    VTRANSPOSE4(i0,i1,i2,i3);

    *out++ = r0; *out++ = i0; *out++ = r1; *out++ = i1;
    *out++ = r2; *out++ = i2; *out++ = r3; *out++ = i3;
  }
}
