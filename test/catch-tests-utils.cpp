#include "catch.hpp"
#include "catch-tests-utils.hpp"

void initMutexes(unsigned int resource_count, pthread_mutex_t *mutexes) {
  for (int i = 0; i < resource_count; i++) {
    pthread_mutex_init(&mutexes[i], nullptr);
  }
}

void pretendWeOnlyHaveCPUs(unsigned int resource_count, worker_thread* hardware_thread_handles) {
  for (int i = 0; i < resource_count; i++) {
    hardware_thread_handles[i].todo_task_dequeue = std::deque<task_nodes *>();
    hardware_thread_handles[i].completed_task_dequeue = std::deque<task_nodes *>();
    hardware_thread_handles[i].task = nullptr;
    hardware_thread_handles[i].resource_state = 0;
    hardware_thread_handles[i].resource_name = "Core " + std::to_string(i+1);
    hardware_thread_handles[i].thread_resource_type = resource_type::cpu;
    hardware_thread_handles[i].resource_cluster_idx = i;
    hardware_thread_handles[i].todo_dequeue_time = 0;
    hardware_thread_handles[i].thread_avail_time = 0;
  }
}

void populateFakeCPUTask(task_nodes* task, std::string task_name, long long estimated_exec) {
  task->task_id = 0;
  task->app_id = 0;
  task->app_pnt = nullptr;
  task->succ = nullptr;
  task->succ_count = 0;
  task->pred = nullptr;
  task->pred_count = 0;
  task->task_name = task_name;
  task->iter = 0;
  task->complete_flag = false;
  task->in_ready_queue = true;
  task->running_flag = false;
  task->actual_execution_time = 0;
  task->assigned_resource_type = resource_type::cpu;
  task->assigned_resource_name = "Unassigned";
  task->supported_resources = std::set<resource_type>{resource_type::cpu};
  task->actual_resource_cluster_idx = 0;
  task->estimated_execution[(uint8_t) resource_type::cpu] = estimated_exec;
}
