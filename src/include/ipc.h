#define SEMAPHORE_NAME0 "sem_submission_queue"
#define SHARED_MEM0 "mem1"
#define CMD_LEN 20
#define CMD_SLOT0_FREE "SLOT0_FREE"
#define CMD_SLOT0_BUSY "SLOT0_BUSY"
#define CMD_TERMINATE "TERMINATE" 
#include <semaphore.h>
#include <sys/shm.h>

enum CMD_TYPES_enum {NOPE,DAG_OBJ,SERVER_EXIT};
struct process_ipc_struct{
	sem_t *sem_handler;
	char semname[255];
	int sh_memd;
	caddr_t sh_mem_virt_addr;
	char sh_mem_name[255];
	long sh_mem_pg_size;
	char *message;
	size_t message_len;

	CMD_TYPES_enum cmd_type;
	char *path_to_dag;
	char *path_to_shared_obj;
};

extern void initialize_sh_mem_server(struct process_ipc_struct *ipc_cmd);
extern void sh_mem_server_process_cmd(struct process_ipc_struct *ipc_cmd);
extern void initialize_sh_mem_client(struct process_ipc_struct *ipc_cmd);
