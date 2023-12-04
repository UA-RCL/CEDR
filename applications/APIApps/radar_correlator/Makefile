ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = radar_correlator

non-api:
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_non_api.c -l:libdash.a -lm -o $(APP_NAME)_non_api-$(ARCH).out

api:
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(APP_NAME)_fft.c -o $(APP_NAME)_fft-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)_fft-$(ARCH).out
$(APP_NAME)_fft-$(ARCH).out:
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_fft.c -l:libdash.a -lm -o $(APP_NAME)_fft-$(ARCH).out

clean:
	-rm $(APP_NAME)*.so
	-rm $(APP_NAME)*.out