/*
 * Copyright(c) 2007-2014 Intel Corporation. All rights reserved.
 *
 *   This program is free software; you can redistribute it and/or modify
 * it under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
 * General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.
 * The full GNU General Public License is included in this distribution 
 * in the file called LICENSE.GPL.
 *
 * Contact Information:
 * Intel Corporation
 */

#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <sys/mman.h>

#define CRAM_BASEADDR   0xF3000000
#define CRAM_SIZE       0x000C0000

int cram_load(unsigned int cram_addr, unsigned char *p_buf, unsigned int buf_size)
{
    int i, fd;
    unsigned char *map;
    unsigned int cram_offset = cram_addr - CRAM_BASEADDR;

    // Load CEVA DSP code to CRAM
    fd = open("/dev/mem", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Can't open /dev/mem, errno: %d (%s)\n", errno, strerror(errno));
        return -1;
    }
    map = mmap(NULL, CRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, CRAM_BASEADDR);
    if (map == MAP_FAILED) {
        fprintf(stderr, "Can't mmap /dev/mem at address %08X, errno: %d (%s)\n", CRAM_BASEADDR, errno, strerror(errno));
		close(fd);
        return -1;
    }
    for (i = 0; i < buf_size; i++)
        map[cram_offset+i] = p_buf[i];

    if (munmap(map, CRAM_SIZE) == -1) {
        fprintf(stderr, "Error un-mmapping the file");
    }
    close(fd);

    return 0;
}

int cram_unload(unsigned int cram_addr, unsigned char *p_buf, unsigned buf_size)
{
    int i, fd;
    unsigned char *map;
    unsigned int cram_offset = cram_addr - CRAM_BASEADDR;

    // Load CEVA DSP code to CRAM
    fd = open("/dev/mem", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Can't open /dev/mem, errno: %d (%s)\n", errno, strerror(errno));
        return -1;
    }
    map = mmap(NULL, CRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, CRAM_BASEADDR);
    if (map == MAP_FAILED) {
        fprintf(stderr, "Can't mmap /dev/mem at address %08X, errno: %d (%s)\n", CRAM_BASEADDR, errno, strerror(errno));
		close(fd);
        return -1;
    }
    for (i = 0; i < buf_size; i++)
        p_buf[i] = map[cram_offset+i];

    if (munmap(map, CRAM_SIZE) == -1) {
        fprintf(stderr, "Error un-mmapping the file");
    }
    close(fd);

    return 0;
}
