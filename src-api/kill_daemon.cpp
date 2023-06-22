#include "ipc.h"
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv) {
  struct process_ipc_struct ipc_cmd {};
  char cmd_terminate[] = CMD_TERMINATE;

  ipc_cmd.message = (ipc_message_struct *)malloc(sizeof(ipc_message_struct));
  memcpy(ipc_cmd.message->CMD, cmd_terminate, sizeof(CMD_TERMINATE));

  initialize_sh_mem_client(&ipc_cmd);
  // client_submission(&ipc_cmd);
  free(ipc_cmd.message);

  return 0;
}
