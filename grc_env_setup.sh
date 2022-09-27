#!/bin/bash

mkdir build
cd build
#cmake -DLIBDASH_MODULES="GPU" ../
cmake ../
make -j
sudo make install
cd ..

git clone https://github.com/WISCA/gr-cedr.git
cd gr-cedr
mkdir build
cd build
cmake ../
make -j
sudo make install
cd ..

