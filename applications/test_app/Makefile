x86:
	mkdir -p ../target/x86
	clang++ \
		-shared \
		-o "../target/x86/test_app.so" \
		test_app.cpp
	cp "test_app.json" ../target/x86

arm:
	mkdir -p ../target/ARM
	clang++ \
		-march=armv8-a \
		-target aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include \
		-shared \
		-o "../target/ARM/test_app.so" \
		test_app.cpp
	cp "test_app.json" ../target/ARM

clean:
	-rm ../target/ARM/test_app.* 2>/dev/null
	-rm ../target/x86/test_app.* 2>/dev/null
