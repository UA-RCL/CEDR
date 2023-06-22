/*
 * SPI testing utility (using spidev driver)
 *
 * Copyright (c) 2007  MontaVista Software, Inc.
 * Copyright (c) 2007  Anton Vorontsov <avorontsov@ru.mvista.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License.
 *
 * Cross-compile with cross-gcc -I/path/to/cross-kernel/include
 */

#include "p2vconv.h"

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <linux/spi/spidev.h>
#include <linux/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#endif

static int mem_fd = -1;
static int top_mem_idx;

static struct tagMEMDESC {
	void *ptr;
	unsigned int size;
} mem_ptr[1024];

int p2vconv_init() {
	mem_fd = open("/dev/mem", O_RDWR);
	if (mem_fd < 0) {
		fprintf(stderr, "Can't open /dev/mem, errno: %d (%s)\n", errno, strerror(errno));
	}

	top_mem_idx = 0;

	return mem_fd;
}

void *p2vconv(unsigned int paddr, unsigned int size) {
	unsigned int offset;
	void *vaddr;

	if (top_mem_idx >= ARRAY_SIZE(mem_ptr)) {
		fprintf(stderr, "Saved memory pointers array is full\n");
		return (void *)0;
	}

	offset = paddr & 0xFFFF;
	paddr &= 0xFFFF0000;
	size += offset;

	// printf("map size=%x paddr=%x\n", size, paddr);

	vaddr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, paddr);
	if (vaddr == MAP_FAILED) {
		fprintf(stderr, "Can't mmap /dev/mem at address 0x%08x, errno: %d (%s)\n", paddr, errno, strerror(errno));
	}

	mem_ptr[top_mem_idx].size = size;
	mem_ptr[top_mem_idx++].ptr = vaddr;

	return (char *)vaddr + offset;
}

int p2vconv_cleanup() {
	int i;

	for (i = 0; i < top_mem_idx; i++) {
		// printf("munmap(%x,%x)\n", mem_ptr[i].ptr, mem_ptr[i].size);
		if (munmap((void *)mem_ptr[i].ptr, mem_ptr[i].size) == -1) {
			fprintf(stderr, "Error un-mmapping 0x%x 0x%x", (unsigned int)mem_ptr[i].ptr, mem_ptr[i].size);
		}
	}
	close(mem_fd);
	return 0;
}
