
CC=clang-9
CFLAGS += -DDASH_DATA=\"./\"
COMPILERRT=../../lib/libclang_rt.builtins-aarch64.a

standalone:
	$(CC) -I../include -o SAR_RDA.standalone SAR_RDA.c -lgsl -lgslcblas -lm -DDASH_DATA=\"./\"

standalone-aarch64:
	$(CC) -march=armv8-a -target aarch64-linux-gnu -I /usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu -I /usr/aarch64-linux-gnu/include -I../include -fuse-ld=lld-9 $(COMPILERRT) SAR_RDA.c ../../lib/libgsl*.a -lm -DDASH_DATA=\"./\" -o SAR_RDA-aarch64.standalone

.PHONY: clean

clean:
	-rm -rf ./*.standalone 2>/dev/null
