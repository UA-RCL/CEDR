#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef short dash_re_int_type;

typedef struct dash_cmplx_int_type {
dash_re_int_type im;
dash_re_int_type re;
} dash_cmplx_int_type;

typedef float dash_re_flt_type;

typedef struct dash_cmplx_flt_type {
dash_re_flt_type re;
dash_re_flt_type im;
} dash_cmplx_flt_type;

typedef struct cedr_barrier {
  pthread_cond_t* cond;
  pthread_mutex_t* mutex;
  uint32_t* completion_ctr;
} cedr_barrier_t;



#ifdef __cplusplus
} // Close 'extern "C"'
#endif
