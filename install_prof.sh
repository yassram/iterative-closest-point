#! /bin/sh

rm -rf build
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS=-pg ..
make
cd ..
