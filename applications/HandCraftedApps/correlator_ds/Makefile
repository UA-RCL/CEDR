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

clean:
	-rm ../target/ARM/correlator.* 2>/dev/null
	-rm ../target/x86/correlator.* 2>/dev/null
