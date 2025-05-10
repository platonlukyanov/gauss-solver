#!/bin/bash

rm a.out tests &>/dev/null

CCOPT="--std=c++17 $(pkgconf --cflags eigen3 openblas) -march=native -I../externals/lazycsv/include -D EIGEN_USE_BLAS -D EIGEN_USE_LAPACKE"
LDOPT="$(pkgconf --libs eigen3 openblas)"

build_app() {
    g++ $CCOPT -o a.out ttt.cc $LDOPT
}

build_tests() {
    g++ $CCOPT -DTESTING -o tests \
    test/test.cpp ttt.cc \
    $LDOPT -lgtest -lgtest_main -lpthread
}

case "$1" in
    "test") build_tests ;;
    *) build_app ;;
esac

