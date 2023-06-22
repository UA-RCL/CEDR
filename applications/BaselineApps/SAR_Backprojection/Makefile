LIB_DIR=$(DASH_ROOT)/alib/

LIBRARIES=\
$(LIB_DIR)libgsl.a\
$(LIB_DIR)libgslcblas.a\
$(LIB_DIR)libfftw3.a\

CFLAGS += -DDASH_DATA=\"$(DASH_DATA)\"

INCLUDE=$(DASH_ROOT)/include/

objects ?= BP_SAR.c

LDFLAGS ?= 

SAR_BP.bc : $(objects)
			$(CC) -I$(INCLUDE) -o SAR_BP.bc $(objects) $(CFLAGS) $(LDFLAGS) $(LIBRARIES)

.PHONY: clean

clean :	
	rm -rf *.bc *.native *.tr*
