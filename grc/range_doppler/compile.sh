#!/bin/bash
if [[ $# -ne 1 ]]
then
    echo "Please use file name as an input"
    exit 1
fi
echo "Running Python to C conversion..."
cython3 "$1.py" --embed
echo "Compiling shared object..."
PYTHONLIBVER=python$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')$(python3-config --abiflags)
c++ $(python3-config --includes) -I../libdash -fPIC "$1.c" -o "$1.so" $(python3-config --ldflags) -l$PYTHONLIBVER -shared
echo "done"
