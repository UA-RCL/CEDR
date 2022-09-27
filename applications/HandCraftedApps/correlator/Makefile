x86:
	mkdir -p ../target/x86
	clang++ \
                -shared \
                -fPIC \
		correlator.cpp \
		-lfftw3f \
		-o "../target/x86/correlator.so" 
	cp "correlator.json" ../target/x86

arm:
	mkdir -p ../target/ARM
	clang++ \
                -march=armv8-a \
                -target aarch64-linux-gnu \
                -I/usr/aarch64-linux-gnu/include/c++/7.2.1/aarch64-linux-gnu \
                -I/usr/aarch64-linux-gnu/include/c++/7.5.0/aarch64-linux-gnu \
                -I/usr/aarch64-linux-gnu/include \
                -L/usr/aarch64-linux-gnu/lib64/ \
                -shared -fPIC \
                -DARM \
                correlator.cpp \
                ../../../toolchains/aarch64-linux-gnu/lib/libfftw3f.a \
                -lm \
                -o "../target/ARM/correlator.so"
	cp "correlator.json" ../target/ARM

# Not really needed anymore, but leaving in case some poor soul needs a reference to rebuild libfftw at some point
# 
# ../../kernels/lib/install-dir/lib/libfftw3f.a:
# 	mkdir -p ../../kernels/lib/fftw-3.3.8/build
# 	cd ../../kernels/lib/fftw-3.3.8/build && \
#           ../configure \
#           --enable-single \
#           --enable-neon \
#           --prefix=`realpath ../../install-dir` \
#           --host=aarch64-linux-gnu \
#           "CC=clang -march=armv8-a --target=aarch64-linux-gnu -fPIC -I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu -I/usr/aarch64-linux-gnu/include" && \
#           $(MAKE) -C . && \
#           $(MAKE) -C . install

clean:
	-rm ../target/ARM/correlator.* 2>/dev/null
	-rm ../target/x86/correlator.* 2>/dev/null
