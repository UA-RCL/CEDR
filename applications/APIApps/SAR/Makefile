ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = SAR
SOURCES = SAR.cpp
NB_SOURCES = SAR_nb.cpp
NB_SOURCES_STATIC_INPUT = SAR_nb_static_input.cpp

static: $(NB_SOURCES_STATIC_INPUT)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(NB_SOURCES_STATIC_INPUT) -o $(APP_NAME)_nb-static-$(ARCH).so
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(NB_SOURCES_STATIC_INPUT) -l:libdash.a -o $(APP_NAME)_nb-static-$(ARCH).out

$(APP_NAME)-$(ARCH).so: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -o $(APP_NAME)-$(ARCH).out

nonblocking: $(APP_NAME)_nb-$(ARCH).so
$(APP_NAME)_nb-$(ARCH).so: $(NB_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(NB_SOURCES) -o $(APP_NAME)_nb-$(ARCH).so

standalone_nb: $(APP_NAME)_nb-$(ARCH).out
$(APP_NAME)_nb-$(ARCH).out: $(NB_SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(NB_SOURCES) -l:libdash.a -o $(APP_NAME)_nb-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
