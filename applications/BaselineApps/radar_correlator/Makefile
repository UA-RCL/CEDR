LIB_DIR=$(DASH_ROOT)/alib/

LIBRARIES=\
$(LIB_DIR)libgsl.a\
$(LIB_DIR)libgslcblas.a\

CFLAGS += -DDASH_DATA=\"$(DASH_DATA)\"

INCLUDE=$(DASH_ROOT)/include/

LDFLAGS ?= 

objects ?= radar_correlator.c

radar_corr.bc : $(objects)
			$(CC) -I$(INCLUDE) -o radar_corr.bc $(objects) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean :	
	rm -rf *.bc *.native *.tr*
