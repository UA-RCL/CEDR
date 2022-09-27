ARCH ?= x86

LIBDASH_INC ?= ../../../../libdash
LIBDASH_LIB ?= ../../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = wifi-tx

C_SOURCES = ../src/crc.c ../src/txData.c ../src/baseband_lib.c ../src/viterbi.c \
	../src/interleaver_deintleaver.c ../src/scrambler_descrambler.c ../src/qpsk_Mod_Demod.c \
	../src/CyclicPrefix.c ../src/Preamble_ST_LG.c ../src/diag.c ../src/rfInf.c ../src/pilot.c
CPP_SOURCES = ../src/TX.cpp ../src/fft_hs.cpp
CPP_SOURCES_NB = ../src/TX_nb.cpp ../src/fft_hs.cpp

SOURCES = $(C_SOURCES) $(CPP_SOURCES)
SOURCES_NB = $(C_SOURCES) $(CPP_SOURCES_NB)

$(APP_NAME)-$(ARCH).so:$(CPP_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

nonblocking: $(APP_NAME)-NB-$(ARCH).so
$(APP_NAME)-NB-$(ARCH).so:$(CPP_SOURCES_NB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_NB) -o $(APP_NAME)-nb-$(ARCH).so

release:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC -O2 $(SOURCES) -o $(APP_NAME)-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
