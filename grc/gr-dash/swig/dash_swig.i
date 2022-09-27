/* -*- c++ -*- */

#define DASH_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "dash_swig_doc.i"

%{
#include "dash/fft.h"
%}

%include "dash/fft.h"
GR_SWIG_BLOCK_MAGIC2(dash, fft);
