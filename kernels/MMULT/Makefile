.PHONY: x86
.PHONY: aarch64

x86: | outdir_x86
	clang-9 \
		-shared \
		-fPIC \
		mmult.c \
		-o ./x86/mmult-x86.so


standalone_x86: | outdir_x86
	clang-9 \
		mmult.c \
		-o ./x86/mmult-x86.standalone

outdir_x86:
	mkdir -p x86

aarch64: | outdir_aarch64
	clang-9 \
		-march=armv8-a \
		-target aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
		-I/usr/aarch64-linux-gnu/include \
		-shared \
		-fPIC \
		mmult.c \
		-o ./aarch64/mmult-aarch64.so

standalone_aarch64: | outdir_aarch64
	clang-9 \
	      -march=armv8-a \
	      -target aarch64-linux-gnu \
	      -I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu \
	      -I/usr/aarch64-linux-gnu/include \
	      -g \
	      mmult.c \
	      -o ./aarch64/mmult-aarch64.standalone

outdir_aarch64:
	mkdir -p aarch64

