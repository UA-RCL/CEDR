CC=clang

LIBRARIES = -lgsl -lgslcblas

LDFLAGS ?= 

objects ?= RF_convergence_LU_projection.c RF_convergence_support.c

arm:
	$(CC) \
	  -march=armv8-a -target aarch64-linux-gnu \
	  -I /usr/aarch64-linux-gnu/include \
	  -I /usr/aarch64-linux-gnu/include/c++/7/aarch64-linux/gnu \
	  -lm \
	  ../../lib/libclang_rt.builtins-aarch64.a \
	  $(objects) \
	  ../../lib/libgsl.a \
	  ../../lib/libgslcblas.a \
	  -o RF_convergence-aarch64.standalone

.PHONY: clean

clean:
	rm -rf *.standalone
