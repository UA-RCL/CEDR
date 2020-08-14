.PHONY: x86
.PHONY: arm
.PHONY: clean

x86:
	clang++-9 \
		-shared \
		-fPIC \
		fft.cpp fft_dma.cpp \
		-lfftw3f \
		-o fft-2048-x86.so

arm: ../lib/install-dir/lib/libfftw3f.a
	clang++-9 \
		-march=armv8-a \
		-target aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/9/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include \
		-I../lib/install-dir/include \
		-L../lib/install-dir/lib \
		-shared \
		-fPIC \
		-DARM \
		../lib/install-dir/lib/libfftw3f.a \
		fft.cpp fft_dma.cpp \
		-lfftw3f \
		-lm \
		-o fft-2048-aarch64.so

../lib/install-dir/lib/libfftw3f.a:
	mkdir -p ../lib/fftw-3.3.8/build
	cd ../lib/fftw-3.3.8/build && \
	  ../configure \
	  --enable-single \
	  --enable-neon \
	  --disable-fortran \
	  --with-slow-timer \
	  --prefix=`realpath ../../install-dir` \
	  --host=aarch64-linux-gnu \
	  "CC=clang -march=armv8-a --target=aarch64-linux-gnu -fPIC -I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu -I/usr/aarch64-linux-gnu/include/c++/9/aarch64-linux-gnu -I/usr/aarch64-linux-gnu/include" && \
	  $(MAKE) -C . -j 4 && \
	  $(MAKE) -C . install

clean:
	if ls *.so 1>/dev/null 2>&1; then rm *.so; fi
