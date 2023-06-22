#ifndef __PILOT_H__
#define __PILOT_H__

/**
 * Necessary structure for pthreads
 * when sending arguments to the 
 * child thread
 */
struct args_pilot {
	comp_t *idata;
	comp_t *odata;
};

struct args_pilotrm {
    int len;
    comp_t *idata;
    comp_t *odata;
};

struct args_pilotex {
    comp_t *idata;
    float *pilot_data;
};

#ifndef THREAD_PER_TASK
void pilotInsertion(comp_t idata[], comp_t odata[]);
#else
void * pilotInsertion(void *input);
#endif

#ifndef THREAD_PER_TASK
void pilotRemove(int len, comp_t idata[], comp_t odata[]);
#else
void* pilotRemove(void *input);
#endif

#ifndef THREAD_PER_TASK
void pilotExtract(comp_t idata[], float pilot_data[]);
#else
void* pilotExtract(void *input);
#endif

#endif
