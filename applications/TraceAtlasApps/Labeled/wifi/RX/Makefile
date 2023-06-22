# define variable
#
	#$(CC) $(LDFLAGS) -static -I../src/wifi/  -I../src/common/ -I ../src/ -I ../src/range_detection/ 
	#../src/common/log_data.c ../src/main.c -lfftw3f -lm -ldl -pthread -Wno-deprecated-declarations -o DASH 
	#
	#

# define the mode of the code
# -DWIFI_TX for wifi_tx
# -DWIFI_RX for wifi_rx
MODE?=-DWIFI_RX

# define number of packets to send
SIZE?=-DNUM_FRAMES=32

CC?= aarch64-linux-gnu-gcc -std=gnu99

INCLUDE1=-I/mnt/nobackup-09/Dash/Sources/include/
INCLUDE2=-I../src/common/
INCLUDE3=-I../src/
INCLUDE4=-I../src/range_detection/
INCLUDE5=-I../src/wifi/
INCLUDES=$(INCLUDE1) $(INCLUDE2) $(INCLUDE3) $(INCLUDE4) $(INCLUDE5) 

ALIBRARY_PATH=/mnt/nobackup-09/Dash/Sources/alib/
LIB1=$(LIBRARY_PATH)libfftw3f.a

SRC1=../src/wifi/baseband_lib.c
SRC2=../src/wifi/channel_Eq.c
SRC3=../src/wifi/channel_Eq_initialize.c
SRC4=../src/wifi/channel_Eq_terminate.c 
SRC5=../src/wifi/crc.c
SRC6=../src/wifi/CyclicPrefix.c
SRC7=../src/wifi/datatypeconv.c
SRC8=../src/wifi/decode.c
SRC9=../src/wifi/detection.c
SRC10=../src/wifi/diag.c
SRC11=../src/common/dma.c
SRC12=../src/wifi/equalizer.c
SRC13=../src/wifi/fft.c
SRC14=../src/wifi/fft_hs.c
SRC15=../src/pulse_doppler/fft_hs_PD.c
SRC16=../src/common/fft_hwa.c
SRC17=../src/wifi/interleaver_deintleaver.c
SRC18=../src/wifi/pilot.c
SRC19=../src/wifi/Preamble_ST_LG.c
SRC20=../src/wifi/qpsk_Mod_Demod.c
SRC21=../src/wifi/rdivide.c
SRC22=../src/wifi/rfInf.c
SRC23=../src/wifi/rtGetInf.c
SRC24=../src/wifi/rtGetNaN.c
SRC25=../src/wifi/rt_nonfinite.c
SRC26=../src/wifi/RX.c
SRC27=../src/wifi/scrambler_descrambler.c
SRC28=../src/wifi/txData.c
SRC29=../src/wifi/viterbi.c
SRC30=../src/wifi/TX.c
SRC31=../src/range_detection/range_detection.c
SRC32=../src/pulse_doppler/pulse_doppler.c
SRC33=../src/common/log_data.c
SRC34=../src/main.c 
SOURCES=$(SRC1) $(SRC2) $(SRC3) $(SRC4) $(SRC5) $(SRC6) $(SRC7) $(SRC8) $(SRC9) $(SRC10) $(SRC11) $(SRC12) $(SRC13) $(SRC14) $(SRC15) $(SRC16) $(SRC17) $(SRC18) $(SRC19) $(SRC20) $(SRC21) $(SRC22) $(SRC23) $(SRC24) $(SRC25) $(SRC26) $(SRC27) $(SRC28) $(SRC29) $(SRC30) $(SRC31) $(SRC32) $(SRC33) $(SRC34)

all:
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIB1) $(SOURCES) $(MODE) $(SIZE) -DPAPI -o wifiRX.bc

run:
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIB1) $(SOURCES) $(MODE) $(SIZE) -o wifi.bc
	opt-8 -load /mnt/nobackup-09/Dash/Tracer/lib/DashPasses.so -EncodedTrace wifi.bc -o wifi.tra -t wifi.trc -TC 9
	clang++-8 -lz -lpthread -lpapi wifi.tra -o wifi.native /mnt/nobackup-09/Dash/Tracer/lib/libDashTracer.a -fuse-ld=lld-8
	./wifi.native 

clean:
	rm -f WIFI WIFI-TX WIFI-RX WIFI-CH log ../Input/ARM_TX_out.dat ../Input/input.dat ../Input/txdata.dat ../log/*
	rm -rf *.bc *.tr* *.ll *.native *.o ./-1* build*

