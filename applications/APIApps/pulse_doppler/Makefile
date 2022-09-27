ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = pulse_doppler
SOURCES = pulse_doppler.c

$(APP_NAME)-$(ARCH).so:
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out:
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -lm -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
