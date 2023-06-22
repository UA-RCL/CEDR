INCLUDES = ../../../libdash
RUNTIME_LIBDIR = ../../../build/libdash
STANDALONE_LIBDIR = $(RUNTIME_LIBDIR)

ifeq ($(ARCH), arm)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	RUNTIME_LIBDIR = ../../../../build-arm/libdash
endif

DEFINES = #-DPRINT_IMAGES

all: serial.so serial.out fft.so fft.out 2Dconv.so 2Dconv.out zip.so zip.out gemm.out gemm.so

serial: serial.so serial.out 

nonblocking: fft_nb.so fft_nb.out

fft: fft.so fft.out fft_nb.so fft_nb.out

2Dconv: 2Dconv.so 2Dconv.out

zip: zip.so zip.out

gemm: gemm.out gemm.so

fft_nb.so: track.cpp conv_nb.cpp
	$(CXX) -shared -I $(INCLUDES) -shared -fPIC -DFFT track.cpp conv_nb.cpp -o track_nb.so

fft_nb.out: conv_nb.cpp track.cpp
	$(CC) -I $(INCLUDES) -L $(STANDALONE_LIBDIR) -DFFT $(DEFINES) conv_nb.cpp track.cpp -l:libdash.a -lstdc++ -lm -o track_nb.out -g

fft.so: track.cpp conv.cpp
	$(CXX) -shared -I $(INCLUDES) -shared -fPIC -DFFT track.cpp conv.cpp -o track.so

fft.out: conv.cpp track.cpp
	$(CC) -I $(INCLUDES) -L $(STANDALONE_LIBDIR) -DFFT conv.cpp $(DEFINES) track.cpp -l:libdash.a -lstdc++ -lm -o track.out -g

serial.so: track.cpp conv.cpp
	$(CXX) -shared -I $(INCLUDES) -shared -fPIC -DSERIAL track.cpp conv.cpp -o track.so

serial.out: conv.cpp track.cpp
	$(CC) -I $(INCLUDES) -L $(STANDALONE_LIBDIR) -DSERIAL conv.cpp track.cpp -l:libdash.a -lstdc++ -lm -o track.out -g

zip.so: track.cpp conv.cpp
	$(CXX) -shared -I $(INCLUDES) -shared -fPIC -DZIP track.cpp conv.cpp -o track.so

zip.out: conv.cpp track.cpp
	$(CC) -I $(INCLUDES) -L $(STANDALONE_LIBDIR) -DZIP conv.cpp track.cpp -l:libdash.a -lstdc++ -lm -o track.out -g

gemm.so: track.cpp conv.cpp
	$(CXX) -shared -I $(INCLUDES) -shared -fPIC -DGEMM track.cpp conv.cpp -o track.so

gemm.out: conv.cpp track.cpp
	$(CC) -I $(INCLUDES) -L $(STANDALONE_LIBDIR) -DGEMM conv.cpp track.cpp -l:libdash.a -lstdc++ -lm -o track.out -g

2Dconv.so: track.cpp conv.cpp
	$(CXX) -shared -I $(INCLUDES) -shared -fPIC -DCONV track.cpp conv.cpp -o track.so

2Dconv.out: conv.cpp track.cpp
	$(CC) -I $(INCLUDES) -L $(STANDALONE_LIBDIR) -DCONV conv.cpp track.cpp -l:libdash.a -lstdc++ -lm -o track.out -g

clean:
	-rm *.so
	-rm *.out
	-rm *.png
	-rm o*.jpeg

