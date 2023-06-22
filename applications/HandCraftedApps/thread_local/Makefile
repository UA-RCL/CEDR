x86:
	mkdir -p ../target/x86
	clang++ \
		-shared -fPIC\
		-o "../../target/x86/thread_local.so" \
		thread_local.cpp
	cp "thread_local.json" ../../target/x86

arm:
	mkdir -p ../target/ARM
	clang++ \
		-march=armv8-a \
		-target aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/7.2.1/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include \
		-shared \
		-o "../../target/ARM/thread_local.so" \
		thread_local.cpp
	cp "thread_local.json" ../../target/ARM

test:
	clang thread_local_test.c \
		-ldl -lpthread \
		-o thread_local_test.out

clean:
	-rm ../../target/ARM/thread_local.* 2>/dev/null
	-rm ../../target/x86/thread_local.* 2>/dev/null
	-rm thread_local_test.out 2>/dev/null
