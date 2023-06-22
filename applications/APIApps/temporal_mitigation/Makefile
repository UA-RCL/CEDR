ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = temporal_mitigation
SOURCES = main.cpp adjoint.cpp alternateinverse.cpp determinant.cpp display.cpp divide.cpp getcofactor.cpp hermitian.cpp imagpart.cpp inverse.cpp mmadd.cpp mmult.cpp mmult4.cpp mmult64.cpp mmultiply.cpp msub.cpp realpart.cpp scalableinverse.cpp

$(APP_NAME)-$(ARCH).so:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
