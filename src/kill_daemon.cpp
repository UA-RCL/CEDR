#include "ipc.h"
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
int main(int argc, char **argv) {
  struct process_ipc_struct ipc_cmd;
  int option;
  char *dag_path = NULL;
  char *binary_path = NULL;
  int dflag = 0, sflag = 0;
  char cmd_terminate[] = CMD_TERMINATE;
  size_t dag_len;
  size_t bin_path_len;

  ipc_cmd.message_len = sizeof(CMD_TERMINATE);
  ipc_cmd.message = (char *)malloc((ipc_cmd.message_len) * sizeof(char));
  memcpy(ipc_cmd.message, cmd_terminate, sizeof(CMD_TERMINATE));

  initialize_sh_mem_client(&ipc_cmd);
  // client_submission(&ipc_cmd);
  free(ipc_cmd.message);

  return 0;
}
