INCLUDES := -I../src-dag/include -I../extern/include
CPP_DEPENDENCIES := ../src-dag/include/keras2c/RT_resource_model_10.cpp ../src-dag/include/keras2c/nn_ml_sched.cpp
CATCH_SOURCES := ./catch-runner.cpp ./catch-tests-utils.cpp
SRC_FILES := $(wildcard ../src-dag/*.cpp)
SRC_FILES := $(filter-out ../src-dag/main.cpp, $(SRC_FILES))

.PHONY: all

all:
# All tests
# Make sure threads.cpp is the last one
	g++ \
		-pthread \
		-Wl,--no-as-needed \
		-ldl \
		$(INCLUDES) \
		$(CPP_DEPENDENCIES) \
		../src-dag/config_manager.cpp \
		../src-dag/scheduler.cpp \
		../src-dag/IL.cpp \
		../src-dag/threads.cpp \
		$(CATCH_SOURCES) \
		./catch-tests-scheduler.cpp \
		./catch-tests-config.cpp \
		./catch-tests-threads.cpp \
		-o test-runner

threads:
	# threads.cpp tests
	g++ \
		-pthread \
		-Wl,--no-as-needed \
		-ldl \
		$(INCLUDES) \
		$(CPP_DEPENDENCIES) \
		../src-dag/config_manager.cpp \
		../src-dag/threads.cpp \
		$(CATCH_SOURCES) \
		./catch-tests-threads.cpp \
		-o test-runner

scheduler:
	# scheduler.cpp tests
	g++ \
		$(INCLUDES) \
		$(CPP_DEPENDENCIES) \
		../src-dag/config_manager.cpp \
		../src-dag/scheduler.cpp \
		../src-dag/IL.cpp \
		$(CATCH_SOURCES) \
		./catch-tests-scheduler.cpp \
		-o test-runner

config:
	# config_manager.cpp tests
	g++ \
		$(INCLUDES) \
		$(CPP_DEPENDENCIES) \
		../src-dag/config_manager.cpp \
		$(CATCH_SOURCES) \
		./catch-tests-config.cpp \
		-o test-runner

clean:
ifneq ("$(wildcard test-runner)", "")
	rm test-runner
endif
