#ifndef _LOG_DATA_FILE
#define _LOG_DATA_FILE

// Function declaration      
void log_data(long long *values, int tid, long long microseconds, long long timestamp);
void write_data(long long *values,long long microseconds);
int elab_papi_read(const char* format,...);
int elab_papi_end(const char* format,...);

#endif
