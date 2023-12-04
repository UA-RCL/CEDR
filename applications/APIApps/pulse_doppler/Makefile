ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

ifeq ($(ARCH), aarch64)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	RUNTIME_LIBDIR = ../../../../build-arm/libdash
endif

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = pulse_doppler
SOURCES = pulse_doppler.c
SOURCES_NB = pulse_doppler_nb.c
SOURCES_NB_STATIC_INPUT = pulse_doppler_nb_static_input.c

$(APP_NAME)-$(ARCH).so: $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so -O3

static: $(SOURCES_NB_STATIC_INPUT)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_NB_STATIC_INPUT) -o $(APP_NAME)-nb-static-$(ARCH).so
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_NB_STATIC_INPUT) -l:libdash.a -lm -o $(APP_NAME)-static-$(ARCH).out

nonblocking: $(APP_NAME)-NB-$(ARCH).so
$(APP_NAME)-NB-$(ARCH).so:$(SOURCES_NB)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_NB) -o $(APP_NAME)-nb-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out: $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -lm -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
