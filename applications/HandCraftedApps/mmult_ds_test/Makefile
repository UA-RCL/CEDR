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
		-o "../target/ARM/mmult_streaming_test.so" \
		mmult_streaming_test.cpp
	cp "mmult_streaming_test.json" ../target/ARM

clean:
	-rm ../target/ARM/mmult_streaming_test.* 2>/dev/null
	-rm ../target/x86/mmult_streaming_test.* 2>/dev/null