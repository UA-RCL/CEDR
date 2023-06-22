#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tdsp.h"
#include "tcb.h"
#include "cevacode.h"

//#define DEBUG 0

int ceva_init(cevadata *ptx)
{
	int rc;
	unsigned int size;
	ptx->CevaDevHandle = open(HW_CEVA_DEV, O_RDWR);
	if (ptx->CevaDevHandle < 0) {
		printf("failed to open device\n");
		return -1;
	}
	size = 1024;
	rc = ioctl(ptx->CevaDevHandle, TDSP_IOCTL_ALLOC_CRAM, &size);
	if (size == 0) {
		printf("Failed to allocate input CRAM buffer\n");
		rc = -2;
		return rc;
	}
	ptx->cramptr = (void *)size;
	return 0;
}

int ceva_deinit(cevadata *ptx)
{
	int rc;
	unsigned int size;
	if(ptx->cramptr != NULL) {
		size = (unsigned int)ptx->cramptr;
		ioctl(ptx->CevaDevHandle, TDSP_IOCTL_FREE_CRAM, &size);
	}
	if(ptx->CevaDevHandle)
		close(ptx->CevaDevHandle);
	return 0;
}

int ceva_end(int taskid)
{
    unsigned int pTCB;
    int rc;
    TCB Tcb, tcbr;
    pTCB = TRANSCEDE_CRAM_BASE + TCB_OFFSET; 
    memset(&Tcb, 0, sizeof(Tcb));
        
    Tcb.TaskID = taskid;
    Tcb.Status = STATUS_READY;

    rc = cram_load(pTCB, &Tcb, sizeof (Tcb));
 
    while (1) {
	    cram_unload(pTCB, &tcbr, sizeof(tcbr));
	    if(tcbr.Status == STATUS_READY)
		    usleep(250);
	    else {
		    break;
	    }
    }
    return 0;
}

int ceva_test_ioctl(int taskid)
{
    int i, rc, fd;
    unsigned int p1, p2, pTCB;
    int Buf[10];
    TCB Tcb;
    rc = 0;
    fd = open(HW_CEVA_DEV, O_RDWR);
    if (fd < 0) {
	    printf("failed\n", HW_CEVA_DEV);
	    return -1;
    }

    for (i=0;i<10;i++) {
	    Buf[i] = i;
	    printf("%d ", Buf[i]);
    }
    printf("\n");
    p1 = 0xF30A0000;
    p2 = 0xF30A8000;

    memset(&Tcb, 0, sizeof(Tcb));
    cram_load(p1, Buf, sizeof(Buf));
    Tcb.TaskID = taskid;
    Tcb.Status = STATUS_READY;
    Tcb.InputDataLen = sizeof(Buf);
    Tcb.OutputDataLen =Tcb.InputDataLen;
    Tcb.InputDataPtr =(void *)p1;
    Tcb.OutputDataPtr =(void *)p2;

    rc = ioctl(fd, TDSP_IOCTL_RUN, &Tcb);
    //rc = ioctl(fd, TDSP_IOCTL_WAIT_RUN_COMP, &Tcb);
    printf("IOCTL Status: %x ExecTicks=%d\n", rc, Tcb.ExecTicks);

    cram_unload(p2, Buf, sizeof(Buf));
    for (i=0;i<10;i++) {
    	printf("%d ", Buf[i]);
    }
    printf("\n");
    close(fd);
    return rc;
}


int ceva_fft(cevadata *ptx, int taskid, int nfft, int fft_size, void *input, void *output)
{
    int i, rc, fd, status;
    unsigned int p1, p2, pTCB;
    short int InBuf[nfft*fft_size*SZF_CMPX];
    short int OutBuf[nfft*fft_size*SZF_CMPX];
    float InVit[fft_size];
    int OutVit[fft_size/2];

    TCB Tcb, tcbr;
    pTCB = TRANSCEDE_CRAM_BASE + TCB_OFFSET; 
    memset(&Tcb, 0, sizeof(Tcb));
/*    fd = open(HW_CEVA_DEV, O_RDWR);
    if (fd < 0) {
	    printf("failed\n", HW_CEVA_DEV);
	    return -1;
    }
*/
    switch(taskid) {
	    case TASKID_ASU_VITERBIK7:
		    {
			    memset(InVit, 0, sizeof(InVit));
			    if (input) {
				    memcpy(InVit, input, sizeof(InVit));
			    }
			    Tcb.TaskID = taskid;
			    Tcb.Status = STATUS_READY;
			    Tcb.InputDataLen = sizeof(InVit);
			    Tcb.OutputDataLen = sizeof(OutVit);
			    break;
		    }
	    default:
		    {
			    memset(InBuf, 0, sizeof(InBuf));
			    if (input) {
				    memcpy(InBuf, input, sizeof(InBuf));
			    }
			    Tcb.TaskID = taskid;
			    Tcb.Status = STATUS_READY;
			    Tcb.InputDataLen = sizeof(InBuf);
			    Tcb.OutputDataLen = sizeof(OutBuf);
			    break;
		    }
    }

    p1 = (unsigned int) ptx->cramptr;
    /*status = ioctl(fd, TDSP_IOCTL_ALLOC_CRAM, &p1);
    if (p1 == 0) {
	    printf("Failed to allocate input CRAM buffer\n");
	    rc = -2;
	    return rc;
    }*/
    p2 = p1 + Tcb.InputDataLen;

#ifdef DEBUG
    printf("insize = %d\n", Tcb.InputDataLen);
    printf("addr1 = %u, addr2 = %u, tcb = %u\n",p1,p2, pTCB);
    printf("addr1 = %x, addr2 = %x, tcb = %x\n",p1,p2, pTCB);
#endif
    Tcb.InputDataPtr =(void *)p1;
    Tcb.OutputDataPtr =(void *)p2;
    switch(taskid) {
	    case TASKID_ASU_VITERBIK7:
		    {
			    cram_load(p1, InVit, sizeof(InVit));
			    memset(OutVit, 0, Tcb.OutputDataLen);
			    cram_load(p2, OutVit, Tcb.OutputDataLen);
			    cram_load(pTCB, &Tcb, sizeof (Tcb));
			    break;
		    }
	    default:
		    {
			    cram_load(p1, InBuf, sizeof(InBuf));
			    memset(OutBuf, 0, Tcb.OutputDataLen);
			    cram_load(p2, OutBuf, Tcb.OutputDataLen);
			    cram_load(pTCB, &Tcb, sizeof (Tcb));
			    break;
		    }
    }
		    

    while (1) {
	    cram_unload(pTCB, &tcbr, sizeof(tcbr));
	    if(tcbr.Status == STATUS_READY)
		    usleep(250);
	    else {
		    break;
	    }
    }
    switch(taskid) {
	    case TASKID_ASU_VITERBIK7:
		    {
			    cram_unload(p2, OutVit, Tcb.OutputDataLen);
			    if (output) {
				    memcpy(output, OutVit, sizeof(OutVit));
			    }
			    break;
		    }
    
	    default:
		    {
			    cram_unload(p2, OutBuf, Tcb.OutputDataLen);
			    if (output) {
				    memcpy(output, OutBuf, sizeof(OutBuf));
			    }
			    break;
		    }
    }
//    ioctl(fd, TDSP_IOCTL_FREE_CRAM, &p1);
//    close(fd);
    return 0;
}

