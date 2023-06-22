LIB_DIR=$(DASH_ROOT)/alib/

LIBRARIES=\
$(LIB_DIR)libgsl.a\
$(LIB_DIR)libgslcblas.a\

CFLAGS += -DDASH_DATA=\"$(DASH_DATA)\"

INCLUDE=$(DASH_ROOT)/include/

LDFLAGS ?= 

objects ?= RF_convergence_LU_projection.c

RF_convergence.bc : $(objects)
	$(CC) -I$(INCLUDE) -o RF_convergence.bc $(objects) $(LIBRARIES) $(LDFLAGS) $(CFLAGS)

.PHONY: clean

clean:
	rm -rf *.bc *.native *.tr*
