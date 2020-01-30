x86:
	mkdir -p ../target/x86
	clang++ \
		-I../../include \
		-shared \
		-fPIC \
		-ggdb \
		-O0 \
		-o "../target/x86/range_detection.so" \
		range_detection.cpp \
		-lm \
		-lfftw3f
	cp "range_detection.json" ../target/x86

#x86_inst:
#	mkdir -p ../target/x86
#	clang++ \
#		-shared \
#		-static \
#		-o "../target/x86/range_detection.so" \
#		range_detection.cpp \
#		-lm
#	clang++ \
#		-static \
#		-o "../target/x86/range_detection.o" \
#		range_detection.cpp \
#		-lm
#	python ../json_parse/instrument_binary.py -b "range_detection.o" -d "range_detection.json" -o "../target/x86/range_detection.o.inst"

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
		-o "../target/ARM/range_detection.so" \
		range_detection.cpp \
		-lm \
		-lfftw3f
	cp "range_detection.json" ../target/ARM

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

clean:
	-rm ../target/ARM/range_detection.* 2>/dev/null
	-rm ../target/x86/range_detection.* 2>/dev/null
