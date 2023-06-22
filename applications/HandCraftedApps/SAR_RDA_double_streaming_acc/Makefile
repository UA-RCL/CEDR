
CC=clang++-9
COMPILERRT=../../lib/libclang_rt.builtins-aarch64.a

x86:
	$(CC) -shared -fPIC -ggdb -I./include -I../../TraceAtlasApps/include -o SAR_RDA.so SAR_RDA.cpp -lgsl -lgslcblas -lm

arm:
	$(CC) \
			-march=armv8-a \
			-target aarch64-linux-gnu \
			-I /usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
			-I /usr/aarch64-linux-gnu/include \
			-I/usr/aarch64-linux-gnu/include/c++/7.2.1/ \
			-I/usr/aarch64-linux-gnu/include/c++/7.2.1/aarch64-linux-gnu/ \
			-L/usr/aarch64-linux-gnu/lib64/ \
			-I../../TraceAtlasApps/include \
			-fuse-ld=lld-9 \
			$(COMPILERRT) SAR_RDA.cpp \
			-shared -fPIC \
			../../../toolchains/aarch64-linux-gnu/lib/libgsl*.a \
			-lm \
			-o SAR_RDA-aarch64.so

.PHONY: clean

clean:
	-rm -rf ./SAR_RDA*.so 2>/dev/null
