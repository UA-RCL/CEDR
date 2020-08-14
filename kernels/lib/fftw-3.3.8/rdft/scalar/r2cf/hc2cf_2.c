/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/* This file was automatically generated --- DO NOT EDIT */
/* Generated on Thu May 24 08:06:55 EDT 2018 */

#include "rdft/codelet-rdft.h"

#if defined(ARCH_PREFERS_FMA) || defined(ISA_EXTENSION_PREFERS_FMA)

/* Generated by: ../../../genfft/gen_hc2c.native -fma -compact -variables 4 -pipeline-latency 4 -n 2 -dit -name hc2cf_2 -include rdft/scalar/hc2cf.h */

/*
 * This function contains 6 FP additions, 4 FP multiplications,
 * (or, 4 additions, 2 multiplications, 2 fused multiply/add),
 * 11 stack variables, 0 constants, and 8 memory accesses
 */
#include "rdft/scalar/hc2cf.h"

static void hc2cf_2(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me, INT ms)
{
     {
	  INT m;
	  for (m = mb, W = W + ((mb - 1) * 2); m < me; m = m + 1, Rp = Rp + ms, Ip = Ip + ms, Rm = Rm - ms, Im = Im - ms, W = W + 2, MAKE_VOLATILE_STRIDE(8, rs)) {
	       E T1, Ta, T3, T6, T4, T8, T2, T7, T9, T5;
	       T1 = Rp[0];
	       Ta = Rm[0];
	       T3 = Ip[0];
	       T6 = Im[0];
	       T2 = W[0];
	       T4 = T2 * T3;
	       T8 = T2 * T6;
	       T5 = W[1];
	       T7 = FMA(T5, T6, T4);
	       T9 = FNMS(T5, T3, T8);
	       Rm[0] = T1 - T7;
	       Im[0] = T9 - Ta;
	       Rp[0] = T1 + T7;
	       Ip[0] = T9 + Ta;
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 1, 2},
     {TW_NEXT, 1, 0}
};

static const hc2c_desc desc = { 2, "hc2cf_2", twinstr, &GENUS, {4, 2, 2, 0} };

void X(codelet_hc2cf_2) (planner *p) {
     X(khc2c_register) (p, hc2cf_2, &desc, HC2C_VIA_RDFT);
}
#else

/* Generated by: ../../../genfft/gen_hc2c.native -compact -variables 4 -pipeline-latency 4 -n 2 -dit -name hc2cf_2 -include rdft/scalar/hc2cf.h */

/*
 * This function contains 6 FP additions, 4 FP multiplications,
 * (or, 4 additions, 2 multiplications, 2 fused multiply/add),
 * 9 stack variables, 0 constants, and 8 memory accesses
 */
#include "rdft/scalar/hc2cf.h"

static void hc2cf_2(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me, INT ms)
{
     {
	  INT m;
	  for (m = mb, W = W + ((mb - 1) * 2); m < me; m = m + 1, Rp = Rp + ms, Ip = Ip + ms, Rm = Rm - ms, Im = Im - ms, W = W + 2, MAKE_VOLATILE_STRIDE(8, rs)) {
	       E T1, T8, T6, T7;
	       T1 = Rp[0];
	       T8 = Rm[0];
	       {
		    E T3, T5, T2, T4;
		    T3 = Ip[0];
		    T5 = Im[0];
		    T2 = W[0];
		    T4 = W[1];
		    T6 = FMA(T2, T3, T4 * T5);
		    T7 = FNMS(T4, T3, T2 * T5);
	       }
	       Rm[0] = T1 - T6;
	       Im[0] = T7 - T8;
	       Rp[0] = T1 + T6;
	       Ip[0] = T7 + T8;
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 1, 2},
     {TW_NEXT, 1, 0}
};

static const hc2c_desc desc = { 2, "hc2cf_2", twinstr, &GENUS, {4, 2, 2, 0} };

void X(codelet_hc2cf_2) (planner *p) {
     X(khc2c_register) (p, hc2cf_2, &desc, HC2C_VIA_RDFT);
}
#endif
