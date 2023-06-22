arm:  ../../kernels/lib/install-dir/lib/libfftw3f.a
	mkdir -p ../target/ARM
	clang++ \
                -march=armv8-a \
                -target aarch64-linux-gnu \
                -I/usr/aarch64-linux-gnu/include/c++/5.4.0/aarch64-linux-gnu \
                -I/usr/aarch64-linux-gnu/include \
                -L/usr/aarch64-linux-gnu/lib64/ \
                -shared -fPIC \
                -DARM \
                fft_acc_ds.cpp \
                -o "../target/ARM/fft_acc_ds.so"
	cp "fft_acc_ds.json" ../target/ARM

arm_debug:  ../../kernels/lib/install-dir/lib/libfftw3f.a
	mkdir -p ../target/ARM
	clang++ \
                -march=armv8-a \
                -target aarch64-linux-gnu \
                -I/usr/aarch64-linux-gnu/include/c++/5.4.0/aarch64-linux-gnu \
                -I/usr/aarch64-linux-gnu/include \
                -L/usr/aarch64-linux-gnu/lib64/ \
                -shared -fPIC \
                -DARM \
                -DDEBUG \
                fft_acc_ds.cpp \
                -o "../target/ARM/fft_acc_ds.so"
	cp "fft_acc_ds.json" ../target/ARM

../../kernels/lib/install-dir/lib/libfftw3f.a:
	mkdir -p ../../kernels/lib/fftw-3.3.8/build
	cd ../../kernels/lib/fftw-3.3.8/build && \
          ../configure \
          --enable-single \
          --enable-neon \
          --prefix=`realpath ../../install-dir` \
          --host=aarch64-linux-gnu \
          "CC=clang -march=armv8-a --target=aarch64-linux-gnu -fPIC -I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu -I/usr/aarch64-linux-gnu/include" && \
          $(MAKE) -C . && \
          $(MAKE) -C . install



clean:
	-rm ../target/ARM/fft_acc_ds.* 2>/dev/null
	-rm ../target/x86/fft_acc_ds.* 2>/dev/null
