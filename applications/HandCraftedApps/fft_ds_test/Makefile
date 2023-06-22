arm:
	mkdir -p ../target/ARM
	clang++ \
		-march=armv8-a \
		-target aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include \
		-I/usr/aarch64-linux-gnu/include/c++/7.2.1/ \
		-I/usr/aarch64-linux-gnu/include/c++/7.2.1/aarch64-linux-gnu/ \
		-L/usr/aarch64-linux-gnu/lib64/ \
		-shared -fPIC \
		-o "../target/ARM/fft_streaming_test.so" \
		fft_streaming_test.cpp
	cp "fft_streaming_test.json" ../target/ARM

clean:
	-rm ../target/ARM/fft_streaming_test.* 2>/dev/null
	-rm ../target/x86/fft_streaming_test.* 2>/dev/null