#!/bin/bash
c++ -I../../libdash -I/root/.grc_gnuradio -I../gr-dash/include -fPIC horn.cpp -o horn -lboost_system -lgnuradio-blocks -lgnuradio-runtime -lgnuradio-pmt -llog4cpp -lgnuradio-dash ../../build/libdash/libdash.a -g 
