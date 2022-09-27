#!/bin/bash
PYTHONLIBVER=python$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')$(python3-config --abiflags)
c++ $(python3-config --includes) -I../libdash -fPIC main.c -o pymain.so ../build/libdash/libdash.a  $(python3-config --ldflags) -l$PYTHONLIBVER -g
