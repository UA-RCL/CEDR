
ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash


ifeq ($(ARCH), jetson)
	PYTHON_ENV_PATH = ../../../../build/torch_jetson
endif

PYTHON_VERSION := $(shell python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VERSION_NUMBER := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_ENV_PATH := $(strip $(shell python3 -c "import sys; print(sys.prefix)")) 

PYTHON_INC = $(shell ${PYTHON_VERSION}-config --cflags)
PYTHON_LD = $(shell ${PYTHON_VERSION}-config --ldflags)

PYBIND_INC = $(strip $(PYTHON_ENV_PATH))/lib/${PYTHON_VERSION}/site-packages/pybind11/include
PYBIND_LIB = $(strip $(PYTHON_ENV_PATH))/lib/${PYTHON_VERSION}/site-packages/pybind11 

INCLUDES = -I $(LIBDASH_INC) -I ${PYTHON_INC} -I ${PYBIND_INC}
LIBS = -L $(LIBDASH_LIB) -L ${PYTHON_LD} -L ${PYBIND_LIB}

APP_NAME = pytorch_app

CPP_SOURCES = ./main.cpp 

SOURCES = $(CPP_SOURCES)
SOURCES_NB = $(C_SOURCES)

ifeq ($(ARCH), aarch64)
        CXX = aarch64-linux-gnu-g++
        CC = aarch64-linux-gnu-gcc
        RUNTIME_LIBDIR = ../../../../build-arm/libdash
endif

default: blocking

blocking: $(SOURCES)
	@echo "Python version used is $(PYTHON_VERSION)"
	@echo "Current Python environment path: $(PYTHON_ENV_PATH)"
	@echo "PYBIND_INC: $(PYBIND_INC)"
	@echo "PYBIND_LIB: $(PYBIND_LIB)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DX86 -DPYTHON_VERSION=\"$(PYTHON_VERSION_NUMBER)\" -DPYTHON_ENV_PATH=\"$(strip $(PYTHON_ENV_PATH))\" -shared -fPIC $(SOURCES) -lpython3.10 -o $(APP_NAME)-$(ARCH).so -rdynamic $(PYTHON_LD)

libpython:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC -rdynamic $(SOURCES) -o $(APP_NAME)-$(ARCH).so $(PYTHON_LD)

zcu:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DZCU102 -shared -fPIC -rdynamic $(SOURCES) -o $(APP_NAME)-$(ARCH).so $(PYTHON_LD)

jetson:
	@echo "Python version used is $(PYTHON_VERSION)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DJETSON -DPYTHON_VERSION=\"$(PYTHON_VERSION_NUMBER)\" -shared -fPIC -rdynamic $(SOURCES) -lpython3.8 -o $(APP_NAME)-$(ARCH).so $(PYTHON_LD)

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -lpython3.10 -l:libdash.a -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
