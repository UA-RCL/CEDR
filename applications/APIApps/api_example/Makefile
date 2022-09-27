ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = api_example
SOURCES = api_example.cpp

$(APP_NAME)-$(ARCH).so:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out