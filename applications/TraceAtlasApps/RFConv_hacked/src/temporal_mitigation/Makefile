LIB_DIR=$(DASH_ROOT)/alib/

CFLAGS += -DDASH_DATA=\"$(DASH_DATA)\"

INCLUDE=$(DASH_ROOT)/include/

LDFLAGS ?= 

objects ?= temporalmitigation.cpp adjoint.cpp alternateinverse.cpp determinant.cpp display.cpp divide.cpp getcofactor.cpp hermitian.cpp imagpart.cpp inverse.cpp mmadd.cpp mmult.cpp mmult4.cpp mmult64.cpp mmultiply.cpp msub.cpp realpart.cpp scalableinverse.cpp

temporal_mitigation.bc : $(objects)
	$(CC) -I$(INCLUDE) -o temporal_mitigation.bc $(objects) $(LIBRARIES) $(LDFLAGS) $(CFLAGS)

.PHONY: clean

clean:
	rm -rf *.bc *.native *.tr*
