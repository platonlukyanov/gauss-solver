#!/bin/bash

rm a.out &>/dev/null

CCOPT="--std=c++17 $(pkgconf --cflags eigen3 openblas) -march=native -I../externals/lazycsv/include -D EIGEN_USE_BLAS -D EIGEN_USE_LAPACKE"
LDOPT="$(pkgconf --libs eigen3 openblas)"

g++ $CCOPT  ttt.cc $LDOPT
