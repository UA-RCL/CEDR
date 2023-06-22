#define _GNU_SOURCE
#include "pilot.h"

#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "common.h"

#define PILOT_LEN 16

static int pilotLoc[PILOT_LEN] = {7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127};

static comp_t pilotData[PILOT_LEN] = {{-1, 1}, {1, 1}, {1, 1}, {1, 1}, {-1, 1}, {1, 1}, {1, 1}, {1, 1},
                                      {-1, 1}, {1, 1}, {1, 1}, {1, 1}, {-1, 1}, {1, 1}, {1, 1}, {1, 1}};

#ifndef THREAD_PER_TASK
void pilotInsertion(comp_t idata[], comp_t odata[]) {
#else
void *pilotInsertion(void *input) {
	comp_t *idata = ((struct args_pilot *)input)->idata;
	comp_t *odata = ((struct args_pilot *)input)->odata;
#endif

#ifdef DISPLAY_CPU_ASSIGNMENT
	printf("[INFO] TX-Pilot-Insertion assigned to CPU: %d\n", sched_getcpu());
#endif

	int i, j, k;

	for (i = 0, j = 0, k = 0; i < 128; i++) {
		if (pilotLoc[j] == i) {
			odata[i] = pilotData[j];
			j++;
		} else {
			odata[i] = idata[k];
			k++;
		}
	}
}

#ifndef THREAD_PER_TASK
void pilotExtract(comp_t idata[], float pilot_data[]) {
#else
void *pilotExtract(void *input) {
	comp_t *idata = ((struct args_pilotex *)input)->idata;
	float *pilot_data = ((struct args_pilotex *)input)->pilot_data;
#endif

#ifdef DISPLAY_CPU_ASSIGNMENT
	printf("[INFO] RX-Pilot-Extraction assigned to CPU: %d\n", sched_getcpu());
#endif

	int i, j;

	for (i = 0, j = 0; i < PILOT_LEN; i++) {
		pilot_data[j++] = idata[pilotLoc[i]].real;
		pilot_data[j++] = idata[pilotLoc[i]].imag;
	}
}

#ifndef THREAD_PER_TASK
void pilotRemove(int len, comp_t idata[], comp_t odata[]) {
#else
void *pilotRemove(void *input) {
	int len = ((struct args_pilotrm *)input)->len;
	comp_t *idata = ((struct args_pilotrm *)input)->idata;
	comp_t *odata = ((struct args_pilotrm *)input)->odata;
#endif

#ifdef DISPLAY_CPU_ASSIGNMENT
	printf("[INFO] RX-Pilot-Removal assigned to CPU: %d\n", sched_getcpu());
#endif

	int i, j, k;

	for (i = 0, j = 0, k = 0; i < len; i++) {
		if (i != pilotLoc[j])
			odata[k++] = idata[i];
		else
			j++;
	}
}
