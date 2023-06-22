all: vector_add vector_add_gpu

vector_add:
	clang++ vector_add.cpp -shared -fPIC -o vector_add.so

vector_add_gpu:
	nvcc vector_add_gpu.cu -shared -Xcompiler -fPIC -o vector_add_gpu.so