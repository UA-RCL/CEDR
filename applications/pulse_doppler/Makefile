x86:
	mkdir -p ../target/x86
	clang++ \
		-I../../include \
		-shared \
		-fPIC \
		-ggdb \
		-O0 \
		-o "../target/x86/pulse_doppler.so" \
		pulse_doppler.cpp \
		-lm \
		-lfftw3f
	cp "pulse_doppler.json" ../target/x86

#x86_inst:
#	mkdir -p ../target/x86
#	clang++ \
#		-shared \
#		-static \
#		-o "../target/x86/pulse_doppler.so" \
#		pulse_doppler.cpp \
#		-lm
#	clang++ \
#		-static \
#		-o "../target/x86/pulse_doppler.o" \
#		pulse_doppler.cpp \
#		-lm
#	python ../json_parse/instrument_binary.py -b "pulse_doppler.o" -d "pulse_doppler.json" -o "../target/x86/pulse_doppler.o.inst"

arm: ../lib/install-dir/lib/libfftw3f.a
	mkdir -p ../target/ARM
	clang++ \
		-march=armv8-a \
		-target aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include \
		-I../../include \
		-I../lib/install-dir/include \
		-L../lib/install-dir/lib \
		-shared \
		-fPIC \
		-ggdb \
		-DARM \
		-O0 \
		-o "../target/ARM/pulse_doppler.so" \
		pulse_doppler.cpp \
		-lm \
		-lfftw3f
	cp "pulse_doppler.json" ../target/ARM

../lib/install-dir/lib/libfftw3f.a:
	mkdir -p ../lib/fftw-3.3.8/build
	cd ../lib/fftw-3.3.8/build && \
	../configure \
        --enable-single \
        --enable-neon \
        --prefix=`realpath ../../install-dir` \
        --host=aarch64-linux-gnu \
        "CC=clang -march=armv8-a --target=aarch64-linux-gnu -fPIC -I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu -I/usr/aarch64-linux-gnu/include" && \
	$(MAKE) -C . && \
	$(MAKE) -C . install

standalone:
	clang++ \
		-I../../include \
		-I../../extern \
		-DINCLUDE_MAIN \
		-o "pulse_doppler.json_gen.out" \
		pulse_doppler.cpp \
		../../src/fft_hs_PD.cpp \
		../../src/dma.cpp \
		../../src/fft_hwa.cpp \
		-lm \
		-lfftw3f

clean:
	-rm ./pulse_doppler.json_gen.out
	-rm ../target/ARM/pulse_doppler.* 2>/dev/null
	-rm ../target/x86/pulse_doppler.* 2>/dev/null
