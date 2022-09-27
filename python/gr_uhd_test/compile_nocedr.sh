#!/bin/bash
PYTHONLIBVER=python$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')$(python3-config --abiflags)
c++ $(python3-config --includes) -I../../libdash -fPIC gr_py_test.c -o gr_py_test.so ../../build/libdash/libdash.a  $(python3-config --ldflags) -l$PYTHONLIBVER -g
