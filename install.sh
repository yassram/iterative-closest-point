#! /bin/sh

rm -rf build
mkdir build
cd build
conan install ..
cmake ..
make
cd ..
