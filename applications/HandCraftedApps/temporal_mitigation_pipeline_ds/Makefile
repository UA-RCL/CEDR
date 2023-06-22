x86:
	mkdir -p ../target/x86
	clang++ \
		-shared -fPIC\
		-o "../target/x86/temporal_mitigation.so" \
		temporal_mitigation.cpp
	cp "temporal_mitigation.json" ../target/x86

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
		-o "../target/ARM/temporal_mitigation.so" \
		temporal_mitigation.cpp
	cp "temporal_mitigation.json" ../target/ARM

clean:
	-rm ../target/ARM/temporal_mitigation.* 2>/dev/null
	-rm ../target/x86/temporal_mitigation.* 2>/dev/null
