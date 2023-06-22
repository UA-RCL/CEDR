ARCH ?= x86
INCLUDES = -I ../0_include -I ../../../extern/include/xenium/
LIBS = -L ../0_lib_dash

ifeq ($(ARCH),x86)
CXX = g++
endif

ifeq ($(ARCH),aarch64)
#CXX = clang++ -march=armv8-a --target=aarch64-linux-gnu -I/usr/aarch64-linux-gnu/include/ -I/usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu
CXX = aarch64-linux-gnu-g++-7
endif

all: ../0_lib_dash/libdash-$(ARCH).a dummy_runtime_$(ARCH)

../0_lib_dash/libdash-$(ARCH).a:
	$(MAKE) -C ../0_lib_dash 

dummy_runtime_$(ARCH):
	$(CXX) $(INCLUDES) $(LIBS) -Wl,--dynamic-list=exported.txt dummy_runtime.cpp -l:libdash-$(ARCH).a -ldl -lpthread -o dummy_runtime_$(ARCH).out

clean:
	-rm dummy_runtime_*.out
