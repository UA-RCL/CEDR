#include "ipc.h"
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <plog/Log.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void mark_slot_free(struct process_ipc_struct *ipc_cmd) {
  const static char cmd_s0_free[] = CMD_SLOT0_FREE;
  memcpy(ipc_cmd->sh_mem_virt_addr, cmd_s0_free, sizeof(CMD_SLOT0_FREE));
}

void initialize_sh_mem_server(struct process_ipc_struct *ipc_cmd) {
  unsigned int value = 1;
  mode_t mode = 0777;
  mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
  int oflag = O_CREAT | O_EXCL;
  int status;
  char cmd_s0_free[] = CMD_SLOT0_FREE;

  ipc_cmd->cmd_type = NOPE;

  strcpy(ipc_cmd->semname, SEMAPHORE_NAME0);
  if (sem_unlink(ipc_cmd->semname) != 0) {
    perror("sem_unlink failed in server initialization");
  }

  ipc_cmd->sem_handler = sem_open(ipc_cmd->semname, oflag, mode, value);
  if (ipc_cmd->sem_handler == SEM_FAILED) {
    perror("sem_open failed in server initialization");
    exit(0);
  }

  strcpy(ipc_cmd->sh_mem_name, SHARED_MEM0);
  if (shm_unlink(ipc_cmd->sh_mem_name) != 0) {
    perror("shm_unlink in server initializtion");
  }

  ipc_cmd->sh_memd = shm_open(ipc_cmd->sh_mem_name, (O_CREAT | O_RDWR | O_EXCL), 0666);
  if (ipc_cmd->sh_memd == -1) {
    perror("shm_open failed in server initializtion");
    exit(0);
  }

  ipc_cmd->sh_mem_pg_size = sysconf(_SC_PAGE_SIZE);
  if ((ftruncate(ipc_cmd->sh_memd, ipc_cmd->sh_mem_pg_size)) == -1) { /* Set the size */
    perror("ftruncate failure in server initializtion");
    exit(0);
  }

  /* Map one page */

  ipc_cmd->sh_mem_virt_addr = (caddr_t)mmap(0, ipc_cmd->sh_mem_pg_size, PROT_WRITE | PROT_READ, MAP_SHARED, ipc_cmd->sh_memd, 0);
  if (ipc_cmd->sh_mem_virt_addr == MAP_FAILED) {
    perror("mmap failure in server initializtion");
    exit(0);
  }

  status = sem_wait(ipc_cmd->sem_handler);
  if (status == -1) {
    perror("sem_wait() failure in server initializtion");
    exit(0);
  }

  mark_slot_free(ipc_cmd);
  // sprintf(ipc_cmd->sh_mem_virt_addr,"%s", "READY");

  // sleep(5);
  if (sem_post(ipc_cmd->sem_handler) == -1) {
    perror("sem_post() failure in server initializtion");
    exit(0);
  }

  /*
  //sleep(5);
  if( sem_close(ipc_cmd->sem_handler) == -1){
          perror("sem_close() failure in server initializtion");
          exit(0);

  }
  if(munmap(ipc_cmd->sh_mem_virt_addr, ipc_cmd->sh_mem_pg_size) == -1){
          perror("munmap() failure in server initializtion");
          exit(0);
  }
  close(ipc_cmd->sh_memd);
  shm_unlink(ipc_cmd->sh_mem_name);
  if(shm_unlink(ipc_cmd->sh_mem_name) != 0){
          perror("shm_unlink failed in server initializtion");
          exit(0);
  }

  */
}

void sh_mem_server_process_cmd(struct process_ipc_struct *ipc_cmd) {
  char cmd_s0_busy[] = CMD_SLOT0_BUSY;
  char cmd_terminate[] = CMD_TERMINATE;
  int status;

  status = sem_trywait(ipc_cmd->sem_handler);
  if (status != 0 && errno == EAGAIN) {
    // Semaphore is currently unavailable, check again later
    return;
  }
  // printf("%s\n", ipc_cmd->sh_mem_virt_addr);
  if (memcmp(ipc_cmd->sh_mem_virt_addr, cmd_s0_busy, sizeof(CMD_SLOT0_BUSY)) == 0) {

    // Copy the received IPC command into our local buffer
    memcpy(ipc_cmd->message, ipc_cmd->sh_mem_virt_addr, sizeof(struct ipc_message_struct));

    // Indicate we've received a DAG
    ipc_cmd->cmd_type = SH_OBJ;

    // And mark the command slot as free again
    mark_slot_free(ipc_cmd);

  } else if (memcmp(ipc_cmd->sh_mem_virt_addr, cmd_terminate, sizeof(CMD_TERMINATE)) == 0) {
    mark_slot_free(ipc_cmd);

    if (sem_post(ipc_cmd->sem_handler) == -1) {
      perror("sem_post() failure in server initializtion");
      exit(0);
    }

    if (sem_close(ipc_cmd->sem_handler) == -1) {
      perror("sem_close() failure in server initializtion");
      exit(0);
    }
    if (munmap(ipc_cmd->sh_mem_virt_addr, ipc_cmd->sh_mem_pg_size) == -1) {
      perror("munmap() failure in server initializtion");
      exit(0);
    }
    close(ipc_cmd->sh_memd);
    // shm_unlink(ipc_cmd->sh_mem_name);
    if (shm_unlink(ipc_cmd->sh_mem_name) != 0) {
      perror("shm_unlink failed in server initializtion");
      exit(0);
    }
    if (sem_unlink(ipc_cmd->semname) != 0) {
      perror("sem_unlink failed in server initializtion");
      exit(0);
    }
    ipc_cmd->cmd_type = SERVER_EXIT;
    return;
    // exit(0);
  }
  if (sem_post(ipc_cmd->sem_handler) == -1) {
    perror("sem_post() failure in server initializtion");
    exit(0);
  }
}

void initialize_sh_mem_client(struct process_ipc_struct *ipc_cmd) {
  unsigned int value = 1;
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
  int oflag = 0;
  int flag_cmd_status = 0;
  char cmd_s0_free[] = CMD_SLOT0_FREE;
  int status;

  strcpy(ipc_cmd->semname, SEMAPHORE_NAME0);

  ipc_cmd->sem_handler = sem_open(ipc_cmd->semname, oflag, mode, value);
  if (ipc_cmd->sem_handler == SEM_FAILED) {
    // fprintf(stderr,"sem_open failed in server with errorno %d\n",errno)
    perror("sem_open failed in client initialization");
    exit(0);
  }

  strcpy(ipc_cmd->sh_mem_name, SHARED_MEM0);
  ipc_cmd->sh_memd = shm_open(ipc_cmd->sh_mem_name, O_RDWR, 0666);
  if (ipc_cmd->sh_memd == -1) {
    perror("shm_open failed in client initialization");
    exit(0);
  }
  ipc_cmd->sh_mem_pg_size = sysconf(_SC_PAGE_SIZE);
  // if((ftruncate(ipc_cmd->sh_memd, pg_size)) == -1){    /* Set the size */
  //	perror("ftruncate failure");
  //	exit(0);
  //}
  /* Map one page */

  ipc_cmd->sh_mem_virt_addr = (caddr_t)mmap(nullptr, ipc_cmd->sh_mem_pg_size, PROT_WRITE | PROT_READ, MAP_SHARED, ipc_cmd->sh_memd, 0);
  if (ipc_cmd->sh_mem_virt_addr == MAP_FAILED) {
    perror("mmap failure in client initializtion");
    exit(0);
  }

  while (flag_cmd_status != 1) {

    status = sem_wait(ipc_cmd->sem_handler);
    if (status == -1) {
      perror("sem_wait() failure in server initializtion");
      exit(0);
    }

    if (memcmp(ipc_cmd->sh_mem_virt_addr, cmd_s0_free, sizeof(CMD_SLOT0_FREE)) == 0) {

      memcpy(ipc_cmd->sh_mem_virt_addr, ipc_cmd->message, sizeof(struct ipc_message_struct));
      flag_cmd_status = 1;
    }

    // printf("%s\n", ipc_cmd->sh_mem_virt_addr);
    sem_post(ipc_cmd->sem_handler);
  }

  sem_close(ipc_cmd->sem_handler);

  munmap(ipc_cmd->sh_mem_virt_addr, ipc_cmd->sh_mem_pg_size);
  close(ipc_cmd->sh_memd);
}
