#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdint.h>

static int fd = -1;

int mmap_init() {

        int i;

        if(fd != -1) return 0;

        if ((fd = open ("/dev/mem", O_RDWR | O_SYNC) ) < 0) {
                printf("Unable to open /dev/mem: %s\n", (char *)strerror(errno));
                return -1;
        }

        return 0;
}


unsigned int *memory_map(int paddr, int size) {

        unsigned int *vaddr;

        if(fd == -1) mmap_init();

        vaddr = (uint32_t *)mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, paddr);
        if (vaddr == MAP_FAILED){
                printf("Mmap failed on physical address %x: %s\n", paddr, (char *)strerror(errno));
                return ((unsigned int *)-1);
        }

        return vaddr;
}
