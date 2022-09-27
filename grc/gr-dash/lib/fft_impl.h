/* -*- c++ -*- */
/*
 * Copyright 2022 gr-dash author.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_DASH_FFT_IMPL_H
#define INCLUDED_DASH_FFT_IMPL_H

#include <dash/fft.h>

namespace gr {
  namespace dash {

    class fft_impl : public fft
    {
     private:
	const size_t d_vlen, d_direction;

     public:
      fft_impl(bool direction, size_t vlen);
      ~fft_impl();

      // Where all the action really happens
      int work(
              int noutput_items,
              gr_vector_const_void_star &input_items,
              gr_vector_void_star &output_items
      );
    };

  } // namespace dash
} // namespace gr

#endif /* INCLUDED_DASH_FFT_IMPL_H */

