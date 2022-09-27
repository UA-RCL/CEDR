x86:
	mkdir -p ../target/x86
	clang++ \
		-shared \
		-fPIC \
		-o "../target/x86/pulse_doppler-x86.so" \
		pulse_doppler.cpp \
		-lm \
		-lfftw3f
	cp "pulse_doppler.json" ../target/x86/pulse_doppler-x86.json
	sed -i "s/pulse_doppler.so/pulse_doppler-x86.so/g" ../target/x86/pulse_doppler-x86.json

arm:
	mkdir -p ../target/ARM
	clang++ \
		-march=armv8-a \
		-target aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include \
		-I../include \
		-shared \
		-g \
		-fPIC \
		-DARM \
		-o "../target/ARM/pulse_doppler-aarch64.so" \
		pulse_doppler.cpp \
		../../../toolchains/aarch64-linux-gnu/lib/libfftw3f.a \
		-lm
	cp "pulse_doppler.json" ../target/ARM/pulse_doppler-aarch64.json
	sed -i "s/pulse_doppler.so/pulse_doppler-aarch64.so/g" ../target/ARM/pulse_doppler-aarch64.json

cuda:
	mkdir -p ../target/cuda
	clang++ \
		-shared \
		-fPIC \
		-DCUDA \
		-o "../target/cuda/pulse_doppler-cuda.so" \
		pulse_doppler.cpp \
		-lm \
		-lfftw3f
	cp "pulse_doppler.json" ../target/cuda/pulse_doppler-cuda.json
	sed -i "s/pulse_doppler.so/pulse_doppler-cuda.so/g" ../target/cuda/pulse_doppler-cuda.json

standalone:
	clang++ \
		-I../../../extern/include \
		-DINCLUDE_MAIN \
		-o "pulse_doppler.json_gen.out" \
		pulse_doppler.cpp \
		-lm \
		-lfftw3f

clean:
	-rm ./pulse_doppler.json_gen.out
	-rm ../target/ARM/pulse_doppler-aarch64.* 2>/dev/null
	-rm ../target/x86/pulse_doppler-x86.* 2>/dev/null
