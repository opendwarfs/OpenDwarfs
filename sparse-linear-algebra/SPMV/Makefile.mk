#
# Copyright 2010 by Virginia Polytechnic Institute and State
# University. All rights reserved. Virginia Polytechnic Institute and
# State University (Virginia Tech) owns the software and its
# associated documentation.
#

bin_PROGRAMS += csr
bin_PROGRAMS += createcsr

csr_LDFLAGS = -lm 
csr_SOURCES = sparse-linear-algebra/SPMV/src/csr.c sparse-linear-algebra/SPMV/src-common/sparse_formats.c sparse-linear-algebra/SPMV/src-common/ziggurat.c sparse-linear-algebra/SPMV/src-common/common.c
createcsr_SOURCES = sparse-linear-algebra/SPMV/src-test/createcsr.c sparse-linear-algebra/SPMV/src-common/sparse_formats.c sparse-linear-algebra/SPMV/src-common/ziggurat.c sparse-linear-algebra/SPMV/src-common/common.c

##createcsr does not need to be linked with any of the opencl common files
createcsr_LDADD = include/common_args.o include/rdtsc.o opts/opts.o
createcsr_LINK = $(CCLD) -o $@ -lm

all_local += csr-all-local
exec_local += csr-exec-local

csr-all-local:
	cp $(top_srcdir)/sparse-linear-algebra/SPMV/src/spmv_kernel.cl .

csr-exec-local:
	cp $(top_srcdir)/sparse-linear-algebra/SPMV/src/spmv_kernel.cl ${DESTDIR}${bindir}
	
