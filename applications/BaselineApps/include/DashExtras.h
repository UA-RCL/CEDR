#pragma once

#ifndef DASHEXTRAS_H
#define DASHEXTRAS_H

#ifdef ENABLE_TRACING

#include <Backend/BackendTrace.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#define KERN_ENTER(str) TraceAtlasMarkovKernelEnter(str);
#define KERN_EXIT(str) TraceAtlasMarkovKernelExit(str);

static char *make_label(const char *fmt, ...) {
	int size = 0;
	char *p = NULL;
	va_list ap;

	va_start(ap, fmt);
	size = vsnprintf(p, size, fmt, ap);
	va_end(ap);

	if (size < 0) return NULL;

	size++;
	p = (char*)malloc(size);
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

#else
#define KERN_ENTER(str)
#define KERN_EXIT(str)
#endif

#endif
