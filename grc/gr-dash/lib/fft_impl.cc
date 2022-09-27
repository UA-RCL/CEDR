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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "fft_impl.h"

namespace gr {
  namespace dash {

    fft::sptr
    fft::make(bool direction, size_t vlen)
    {
      return gnuradio::get_initial_sptr
        (new fft_impl(direction, vlen));
    }


    /*
     * The private constructor
     */
    fft_impl::fft_impl(bool direction, size_t vlen)
      : gr::sync_block("fft",
              gr::io_signature::make(1, 1, sizeof(gr_complex)*vlen),
              gr::io_signature::make(1, 1, sizeof(gr_complex)*vlen)),
	d_direction(direction),
	d_vlen(vlen)
    {}

    /*
     * Our virtual destructor.
     */
    fft_impl::~fft_impl()
    {
    }

    int
    fft_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
	auto in = reinterpret_cast<const gr_complex*>(input_items[0]);
	auto out = reinterpret_cast<gr_complex*>(output_items[0]);

	int count = 0;
//  	printf("new work called with noutput_items: %d\n", noutput_items);

	while (count++ < noutput_items) {
	      double *temp_in = new double[2*d_vlen];
	      double *temp_out = new double[2*d_vlen];
	      for (size_t i = 0; i < (size_t)d_vlen; i++){
		const gr_complex x = in[i]; // gr_complex is a type signature for std::complex<float>
		temp_in[2*i] = (double) x.real();
		temp_in[2*i+1] = (double) x.imag();
	      }

	      /* Call out to the DASH_FFT Macro that will hook into CEDR
	       * complex input and output,  input[2*i+0] = real, input[2*i+1] = imaginary
	       * - d_vlen is size of the complex transform, true is forward, false is inverse */
	      DASH_FFT(temp_in, temp_out, d_vlen, d_direction);

	      // Copy the temporary output buffer into the GR output buffer - and also cast back to float from double
	      for (size_t i = 0; i < (size_t)d_vlen; i++)
	      {
		gr_complex o = gr_complex(temp_out[2*i],temp_out[2*i+1]);
		*out++ = (gr_complex) o;
	      }


	      // Free the dynamically allocated temp buffers
	      delete[] temp_in;
	      delete[] temp_out;
	      // Move the pointer forward by FFT length
	      in += d_vlen;
	      // The out pointer has already been moved forward appropriately by the output copy loop
	}
      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace dash */
} /* namespace gr */

