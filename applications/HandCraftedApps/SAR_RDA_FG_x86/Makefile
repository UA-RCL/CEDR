
CC=clang++-9
CFLAGS += -DDASH_DATA=\"./\"
COMPILERRT=../lib/libclang_rt.builtins-aarch64.a

x86:
	$(CC) -shared -fPIC -ggdb -I./include -o SAR_RDA.so SAR_RDA.cpp -lgsl -lgslcblas -lm -DDASH_DATA=\"./\"

standalone:
	clang++ \
		-g \
		-I../../extern/include \
		-o "gen.json_gen.out" \
		gen_x86.cpp \
		-lm
		
arm:
	$(CC) \
			-march=armv8-a \
			-target aarch64-linux-gnu \
			-I /usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
			-I /usr/aarch64-linux-gnu/include \
			-I/usr/aarch64-linux-gnu/include/c++/7.2.1/ \
			-I/usr/aarch64-linux-gnu/include/c++/7.2.1/aarch64-linux-gnu/ \
			-L/usr/aarch64-linux-gnu/lib64/ \
			-I./include -fuse-ld=lld-9 \
			$(COMPILERRT) SAR_RDA.cpp \
			-shared -fPIC \
			../lib/libgsl*.a \
			-lm -DDASH_DATA=\"./\" \
			-o SAR_RDA-aarch64.so

.PHONY: clean

clean:
	-rm -rf ./*.standalone 2>/dev/null
