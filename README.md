# Pymeshcorr
Fast ensemble average for mesh array

Program to determine the correlation function in the given grid of size 'n x n x n' using MPI4PY

#Instructions

Compile the cython code first. Can be done using the following two commands:

    cython -a MPI_correlation_binning.pyx
    gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o  MPI_correlation_binning.so MPI_correlation_binning.c


# License 
Pymeshcorr is a free software made available under the GPL3 License. For details see the LICENSE file.

Copyright 2015 Joseph Kuruvilla
