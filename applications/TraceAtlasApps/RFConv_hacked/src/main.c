#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "CyclicPrefix.h"
#include "IFFT_FFT.h"
#include "Preamble_ST_LG.h"
#include "baseband_lib.h"
#include "common.h"
#include "datatypeconv.h"
#include "dma.h"
#include "fft_hs.h"
#include "fft_hwa.h"
#include "functions_common.h"
#include "interleaver_deintleaver.h"
#include "pilot.h"
#include "qpsk_Mod_Demod.h"
#include "rfInf.h"
#include "rf_interface.h"
#include "scrambler_descrambler.h"
#include "txData.h"
#include "viterbi.h"

//---- START  OF
//PAPI----------------------------------------------------------------------------------------------------

// Specific to PAPI calls
#ifdef PAPI
#include <papi.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "log_data.h"

#ifdef PAPI
int retval, EventSet = PAPI_NULL, EventCode;
char EventCodeStr[PAPI_MAX_STR_LEN];
long_long values[35];
float real_time, proc_time, mflops;

struct timespec begin, current;
long long start, elapsed, microseconds, previoustime = 0;
#define NANOS 1000000000LL

// Read PAPI at the end
int elab_papi_end(const char *format, ...) {
	// Store temparory value of timestamp
	long long microseconds_temp = 0, timestamp;

	if (PAPI_read(EventSet, values) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// Get elapsed time
	if (clock_gettime(CLOCK_MONOTONIC, &current)) {
		/* getting clock time failed, what now? */
		exit(EXIT_FAILURE);
	}

	// Elapsed time in nanoseconds
	elapsed = current.tv_sec * NANOS + current.tv_nsec - start;
	microseconds_temp = elapsed / 1000 + (elapsed % 1000 >= 500);  // round up halves
	timestamp = microseconds_temp;

	// Previous time
	microseconds = microseconds_temp - previoustime;
	previoustime = microseconds_temp;

	// Logging performance data
	log_data(values, 0, microseconds, timestamp);

	return 1;
}

// Initialize PAPI
void papi_init() {
	retval = PAPI_library_init(PAPI_VER_CURRENT);

	if (retval != PAPI_VER_CURRENT) {
		fprintf(stderr, "PAPI library init error!\n");
		exit(1);
	}

	if (PAPI_create_eventset(&EventSet) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// Instructions
	if (PAPI_event_name_to_code("INST_RETIRED", &EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	if (PAPI_add_event(EventSet, EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// CPU Cycles
	if (PAPI_event_name_to_code("CPU_CYCLES", &EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	if (PAPI_add_event(EventSet, EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// Branch Miss Prediction
	if (PAPI_event_name_to_code("BRANCH_MISPRED", &EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	if (PAPI_add_event(EventSet, EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// Level 2 cache misses
	if (PAPI_event_name_to_code("PAPI_L2_TCM", &EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	if (PAPI_add_event(EventSet, EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// Data Memory Access
	if (PAPI_event_name_to_code("DATA_MEM_ACCESS", &EventCode) != PAPI_OK)
		printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	if (PAPI_add_event(EventSet, EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// NONCACHE_EXTERNAL_MEMORY_REQUEST
	if (PAPI_event_name_to_code("NONCACHE_EXTERNAL_MEMORY_REQUEST", &EventCode) != PAPI_OK)
		printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	if (PAPI_add_event(EventSet, EventCode) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// Start PAPI calls.
	if (PAPI_start(EventSet) != PAPI_OK) printf("%s:%d\t ERROR\n", __FILE__, __LINE__);

	// Start the timer
	if (clock_gettime(CLOCK_MONOTONIC, &begin)) {
		exit(EXIT_FAILURE);
	}

	// Start time in nanoseconds
	start = begin.tv_sec * NANOS + begin.tv_nsec;
}

//---- END OF PAPI----------------------------------------------------------------------------------------------------
#endif

//############################
//## FFT variable declarations
//############################
extern TYPE *udmabuf1_base_addr;
extern int dma1_control_fd;
extern unsigned int *dma1_control_base_addr;
extern int fd_udmabuf1;
extern int fft1_control_fd;
extern unsigned int *fft1_control_base_addr;
extern TYPE *udmabuf2_base_addr;
extern int dma2_control_fd;
extern unsigned int *dma2_control_base_addr;
extern int fd_udmabuf2;
extern int fft2_control_fd;
extern unsigned int *fft2_control_base_addr;
extern unsigned int udmabuf1_phys_addr;
extern unsigned int udmabuf2_phys_addr;

TYPE *udmabuf1_base_addr;
int dma1_control_fd;
unsigned int *dma1_control_base_addr;
int fd_udmabuf1;
int fft1_control_fd;
unsigned int *fft1_control_base_addr;
TYPE *udmabuf2_base_addr;
int dma2_control_fd;
unsigned int *dma2_control_base_addr;
int fd_udmabuf2;
int fft2_control_fd;
unsigned int *fft2_control_base_addr;
unsigned int udmabuf1_phys_addr;
unsigned int udmabuf2_phys_addr;

extern sem_t mutex;
sem_t mutex;

void *wifi_tx();
void *wifi_rx();
void *pulse_doppler();
void *range_detection();
void *temp_mit();

int main() {
	int rc;

	// Thread related variable declarations
	pthread_t thread_wifi_tx, thread_wifi_rx, thread_temp_mit, thread_range_detection;

	int core_id = 4;

	// Setting thread affinity
	int s1, s2, s3, s4;
	pthread_attr_t attr_thread_wifi_tx, attr_thread_wifi_rx, attr_thread_temp_mit, attr_thread_range_detection;
	cpu_set_t cpuset1, cpuset2, cpuset3, cpuset4;

	pthread_attr_init(&attr_thread_wifi_tx);
	CPU_ZERO(&cpuset1);
	CPU_SET(core_id, &cpuset1);

	pthread_attr_init(&attr_thread_wifi_rx);
	CPU_ZERO(&cpuset2);
	CPU_SET(core_id, &cpuset2);

	pthread_attr_init(&attr_thread_temp_mit);
	CPU_ZERO(&cpuset3);
	CPU_SET(core_id, &cpuset3);

	pthread_attr_init(&attr_thread_range_detection);
	CPU_ZERO(&cpuset4);
	CPU_SET(core_id, &cpuset4);

	s1 = pthread_attr_setaffinity_np(&attr_thread_wifi_tx, sizeof(cpuset1), &cpuset1);
	s2 = pthread_attr_setaffinity_np(&attr_thread_wifi_rx, sizeof(cpuset2), &cpuset2);
	s3 = pthread_attr_setaffinity_np(&attr_thread_temp_mit, sizeof(cpuset3), &cpuset3);
	s4 = pthread_attr_setaffinity_np(&attr_thread_range_detection, sizeof(cpuset4), &cpuset4);

	//#################################################################
	//## Initialize Semaphores, DMA and FFT IPs
	//#################################################################

	if (sem_init(&mutex, 0, 1) != 0) {
		printf("[ERROR] Semaphore creation failed ...\n");
		exit(-1);
	}

#ifdef FFT1_HW
	// Virtual Address to DMA Control Slave
	init_dma1();

	init_fft1();

	// Virtual Address to udmabuf Buffer
	init_udmabuf1();

#endif

#ifdef FFT2_HW
	// Virtual Address to DMA Control Slave
	init_dma2();

	init_fft2();

	// Virtual Address to udmabuf Buffer
	init_udmabuf2();

#endif
	//#################################################################
	//## Create threads for each of the applications
	//#################################################################
#ifdef WIFI_RX
	// Create thread for WIFI-RX
	// Check status of thread create
#ifdef PAPI
	wifi_rx();
#elseif FORCE_CPU_AFFINITY
	if (pthread_create(&thread_wifi_rx, &attr_thread_wifi_rx, wifi_rx, NULL) != 0) {
#else
	if (pthread_create(&thread_wifi_rx, NULL, wifi_rx, NULL) != 0) {
		printf("[ERROR] WIFI-RX Thread Creation FAILED ...\n");
		exit(-1);
	} else {
		printf("[ INFO] WIFI-RX Thread successfully created ...\n");
	}
	// Join created threads
	if (pthread_join(thread_wifi_rx, NULL) != 0) {
		printf("[ERROR] WIFI-RX Thread Join FAILED ...\n");
		exit(-1);
	}
#endif  // #ifdef PAPI
#endif  // #ifdef WIFI_RX

#ifdef WIFI_TX

#ifdef PAPI
	wifi_tx();
#elseif FORCE_CPU_AFFINITY
	if (pthread_create(&thread_wifi_tx, &attr_thread_wifi_tx, wifi_tx, NULL) != 0) {
#else
	if (pthread_create(&thread_wifi_tx, NULL, wifi_tx, NULL) != 0) {
		printf("[ERROR] WIFI-TX Thread Creation FAILED ...\n");
		exit(-1);
	} else {
		printf("[ INFO] WIFI-TX Thread successfully created ...\n");
	}
	// Join created threads
	if (pthread_join(thread_wifi_tx, NULL) != 0) {
		printf("[ERROR] WIFI-TX Thread Join FAILED ...\n");
		exit(-1);
	}	
#endif  // #ifdef PAPI
#endif  // #ifdef WIFI_TX

#ifdef TEMP_MIT

#ifdef PAPI
	temp_mit();
#elseif FORCE_CPU_AFFINITY
	if (pthread_create(&thread_temp_mit, &attr_thread_temp_mit, temp_mit, NULL) != 0) {
#else
	if (pthread_create(&thread_temp_mit, NULL, temp_mit, NULL) != 0) {
		printf("[ERROR] TEMP-MIT Thread Creation FAILED ...\n");
		exit(-1);
	} else {
		printf("[ INFO] TEMP-MIT Thread successfully created ...\n");
	}
	// Join created threads
	if (pthread_join(thread_temp_mit, NULL) != 0) {
		printf("[ERROR] TEMP-MIT Thread Join FAILED ...\n");
		exit(-1);
	}
#endif  // #ifdef PAPI
#endif  // #ifdef TEMP_MIT

#ifdef RANGE_DETECTION
	// Create thread for Range Detection
	// Check status of thread create
#ifdef PAPI
	range_detection();
#elseif FORCE_CPU_AFFINITY
	if (pthread_create(&thread_range_detection, &attr_thread_range_detection, range_detection, NULL) != 0) {
#else
	if (pthread_create(&thread_range_detection, NULL, range_detection, NULL) != 0) {
		printf("[ERROR] Range Detection Thread Creation FAILED ...\n");
		exit(-1);
	} else {
		printf("[ INFO] Range Detection Thread successfully created ...\n");
	}
	// Join created threads
	if (pthread_join(thread_range_detection, NULL) != 0) {
		printf("[ERROR] Lag Detection Thread Join FAILED ...\n");
		exit(-1);
	}
#endif  // #ifdef PAPI
#endif  // #ifdef RANGE_DETECTION

	//#################################################################
	//## Destroy Semaphore
	//#################################################################

	if (sem_destroy(&mutex) != 0) {
		printf("[ERROR] Semaphore destroy failed ...\n");
		exit(-1);
	}

	//#################################################################
	//## Close FFT hardware calls
	//#################################################################

#ifdef FFT1_HW
	close_dma1();
	close_fft1();
	munmap(udmabuf1_base_addr, 8192);
	close(fd_udmabuf1);
#endif

#ifdef FFT2_HW
	close_dma2();
	close_fft2();
	munmap(udmabuf2_base_addr, 8192);
	close(fd_udmabuf2);
#endif

	return 0;
}
