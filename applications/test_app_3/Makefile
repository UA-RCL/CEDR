x86:
	mkdir -p ../target/x86
	clang++ \
		-fPIC \
		-shared \
		-o "../target/x86/test_app_3.so" \
		test_app_3.cpp
	cp "test_app_3.json" ../target/x86

arm:
	mkdir -p ../target/ARM
	clang++ \
		-march=armv8-a \
		-target aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include \
		-shared \
		-o "../target/ARM/test_app_3.so" \
		test_app_3.cpp
	cp "test_app_3.json" ../target/ARM

clean:
	-rm ../target/ARM/test_app_3.* 2>/dev/null
	-rm ../target/x86/test_app_3.* 2>/dev/null
