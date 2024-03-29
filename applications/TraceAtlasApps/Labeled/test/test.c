
#include "/localhome/jmack2545/rcl/DASH-SoC/CEDR_private/TraceAtlas/TraceInfrastructure/include/Backend/BackendTrace.h"
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

void KernelEnter(char *label);
void KernelExit(char *label);

#ifdef TRC
#define KERN_ENTER(str) KernelEnter(str);
#define KERN_EXIT(str) KernelExit(str);
#else
#define KERN_ENTER(str);
#define KERN_EXIT(str);
#endif

char *make_label(const char *fmt, ...);

int main(void) {
    int i, a, b, c;

    for (i = 0; i < 1; i++) {}

    a = 1; b = 2; c = 3;
    KERN_ENTER(make_label("FFT[1D][1024][float]"));
    for (i = 0; i < 1024; i++) {
        a = b * c;
    }
    KERN_EXIT(make_label("FFT[1D][1024][float]"));

    return 0;
}

char *make_label(const char *fmt, ...) {
    int size = 0;
    char *p = NULL;
    va_list ap;

    va_start(ap, fmt);
    size = vsnprintf(p, size, fmt, ap);
    va_end(ap);

    if (size < 0) return NULL;

    size++;
    p = (char*) malloc(size);
    if (p == NULL) return NULL;

    va_start(ap, fmt);
    size = vsnprintf(p, size, fmt, ap);
    va_end(ap);

    if (size < 0) {
        free(p);
        return NULL;
    }
    return p;
}