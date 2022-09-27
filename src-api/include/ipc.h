#define SEMAPHORE_NAME0 "sem_submission_queue"
#define SHARED_MEM0 "mem1"
#define CMD_SLOT0_FREE "SLOT0_FREE"
#define CMD_SLOT0_BUSY "SLOT0_BUSY"
#define CMD_TERMINATE "TERMINATE"

#define IPC_MAX_PATH_LEN 256
#define IPC_MAX_APPS 8

#include <semaphore.h>
#include <sys/shm.h>
#include <cstdint>

enum CMD_TYPES_enum {NOPE,SH_OBJ,SERVER_EXIT};
typedef char* caddr_t;

/*
 * Note: This struct likely can't contain any pointers as the data of this struct will be communicated across processes
 * As such, most dynamically allocated data structures will simply be copied by value as an address
 * And, in the memory space of the destination process, those addresses don't mean anything (and just cause segfaults)
 *
 * Could be wrong here, but that's my understanding based on attempts to use i.e. char* path_to_dag with a buffer I calloc'd
 */
struct ipc_message_struct {
  // IMPORTANT: THIS CMD ARR MUST REMAIN AT THE FRONT OF THIS STRUCT DUE TO THE IPC MESSAGING PROTOCOL
  char CMD[12];
  // Feel free to add/remove struct members from here onward, but we check whether an application has arrived by mem-comparing
  // the "is there a new DAG?" string with a memory buffer containing one of these structs.
  // As such, we need the initial bytes of this struct to be interpreted as the command value.
  // If you REALLY want to move it around, then YOU get to write the logic to offset that memcmp call
  char path_to_so[IPC_MAX_APPS][IPC_MAX_PATH_LEN];
  //char path_to_tracefile[8][256];
  uint64_t num_instances[IPC_MAX_APPS];
  uint64_t periodicity[IPC_MAX_APPS];
};
struct process_ipc_struct{
  sem_t *sem_handler;
  char semname[255];
  int sh_memd;
  caddr_t sh_mem_virt_addr;
  char sh_mem_name[255];
  long sh_mem_pg_size;

  CMD_TYPES_enum cmd_type;
  struct ipc_message_struct* message;
};

extern void initialize_sh_mem_server(struct process_ipc_struct *ipc_cmd);
extern void sh_mem_server_process_cmd(struct process_ipc_struct *ipc_cmd);
extern void initialize_sh_mem_client(struct process_ipc_struct *ipc_cmd);
