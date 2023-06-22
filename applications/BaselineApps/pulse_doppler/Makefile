LIB_DIR=$(DASH_ROOT)/alib/

LIBRARIES=\
$(LIB_DIR)libgsl.a\
$(LIB_DIR)libgslcblas.a\

CFLAGS += -DDASH_DATA=\"$(DASH_DATA)\"

objects ?= pulse_doppler.c

INCLUDE = -I$(DASH_ROOT)/include/

LDFLAGS ?=

pulse_doppler.bc : $(objects)
			$(CC) -o pulse_doppler.bc $(INCLUDE) $(objects) $(CFLAGS) $(LDFLAGS) $(LIBRARIES)

.PHONY: clean

clean:
		rm -rf *.bc *.native *.tr*
