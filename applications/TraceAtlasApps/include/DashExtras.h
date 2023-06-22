#pragma once

#ifndef DASHEXTRAS_H
#define DASHEXTRAS_H

#ifdef ENABLE_TRACING

//#include <Backend/BackendTrace.h>
#include "BackendTrace.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// Causes issues with JR-based kernel extraction because the if statement leads to a branch either into (i) start of node or (ii) one block further into the node
//#define KERN_ENTER(str) { if (str != "") { KernelEnter(str); } };
//#define KERN_EXIT(str) { if (str != "") { KernelExit(str); } };
#define KERN_ENTER(str) { KernelEnter(str); };
#define KERN_EXIT(str) { KernelExit(str); };

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
