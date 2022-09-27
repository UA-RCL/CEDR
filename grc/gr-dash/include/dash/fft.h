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

#ifndef INCLUDED_DASH_FFT_H
#define INCLUDED_DASH_FFT_H

#include <dash/api.h>
#include <gnuradio/sync_block.h>
#include "dash.h"

namespace gr {
  namespace dash {

    /*!
     * \brief <+description of block+>
     * \ingroup dash
     *
     */
    class DASH_API fft : virtual public gr::sync_block
    {
     public:
      typedef boost::shared_ptr<fft> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of dash::fft.
       *
       * To avoid accidental use of raw pointers, dash::fft's
       * constructor is in a private implementation
       * class. dash::fft::make is the public interface for
       * creating new instances.
       */
      static sptr make(bool direction, size_t vlen);
    };

  } // namespace dash
} // namespace gr

#endif /* INCLUDED_DASH_FFT_H */

