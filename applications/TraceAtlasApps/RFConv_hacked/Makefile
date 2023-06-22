# define variable
#CC		:= gcc -std=gnu99
#CC		:= aarch64-linux-gnu-gcc
CC := clang

WIFI_SOURCES=$(wildcard src/wifi/*.c)
WIFI_OBJECTS=$(patsubst %.c, %.o, $(WIFI_SOURCES))

TEMPMIT_SOURCES=$(wildcard src/temporal_mitigation/*.cpp)
TEMPMIT_OBJECTS=$(patsubst %.c, %.o, $(TEMPMIT_SOURCES))

rfconv:
	$(CC) \
	src/main.c \
	src/wifi/baseband_lib.c src/wifi/channel_Eq.c src/wifi/channel_Eq_initialize.c src/wifi/channel_Eq_terminate.c \
	src/wifi/crc.c src/wifi/CyclicPrefix.c src/wifi/datatypeconv.c src/wifi/decode.c src/wifi/detection.c src/wifi/diag.c src/common/dma.c \
	src/wifi/equalizer.c src/wifi/fft.c src/wifi/fft_hs.c src/common/fft_hwa.c src/wifi/interleaver_deintleaver.c src/wifi/pilot.c \
	src/wifi/Preamble_ST_LG.c src/wifi/qpsk_Mod_Demod.c src/wifi/rdivide.c src/wifi/rfInf.c src/wifi/rtGetInf.c src/wifi/rtGetNaN.c \
	src/wifi/rt_nonfinite.c src/wifi/RX.c src/wifi/scrambler_descrambler.c src/wifi/txData.c src/wifi/viterbi.c src/wifi/TX.c \
	src/common/log_data.c \
	src/temporal_mitigation/adjoint.cpp src/temporal_mitigation/alternateinverse.cpp src/temporal_mitigation/determinant.cpp \
	src/temporal_mitigation/display.cpp src/temporal_mitigation/divide.cpp src/temporal_mitigation/getcofactor.cpp \
	src/temporal_mitigation/hermitian.cpp src/temporal_mitigation/imagpart.cpp src/temporal_mitigation/inverse.cpp \
	src/temporal_mitigation/mmadd.cpp src/temporal_mitigation/mmult.cpp src/temporal_mitigation/mmult4.cpp \
	src/temporal_mitigation/mmult64.cpp src/temporal_mitigation/mmultiply.cpp src/temporal_mitigation/msub.cpp \
	src/temporal_mitigation/realpart.cpp src/temporal_mitigation/scalableinverse.cpp \
	src/range_detection/range_detection.c \
	-lm -lgsl -lgslcblas -lfftw3f -lpthread \
	-Isrc -Isrc/common -Isrc/pulse_doppler -Isrc/range_detection -Isrc/temporal_mitigation -Isrc/wifi \
	-DWIFI_RX -DWIFI_TX -DTEMP_MIT -DRANGE_DETECTION -DNUM_FRAMES=1 \
	-DKERN_ENTER\(str\) -DKERN_EXIT\(str\) \
	-o rfconv

clean:
	rm rfconv

