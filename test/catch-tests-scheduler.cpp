#define CEDR_UNIT_TESTING
#include "catch.hpp"
#include "catch-tests-utils.hpp"
#include "scheduler.hpp"
#undef CEDR_UNIT_TESTING

#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Log.h>

#define LOG_LEVEL "NONE"

TEST_CASE("On a CPU-only system, each scheduler behaves as expected", "[CPU Tests]") {
  static plog::ConsoleAppender<plog::TxtFormatter> appender;
  plog::init(plog::severityFromString(LOG_LEVEL), &appender);

  ConfigManager cedr_config = ConfigManager();
  cedr_config.resource_array[(uint8_t) resource_type::cpu] = 3;
  cedr_config.resource_array[(uint8_t) resource_type::fft] = 0;
  cedr_config.resource_array[(uint8_t) resource_type::mmult] = 0;
  
  uint32_t available_resources = cedr_config.getTotalResources();

  std::deque<task_nodes *> ready_queue;
  worker_thread hardware_thread_handles[cedr_config.getTotalResources()];
  pthread_mutex_t resource_mutex[cedr_config.getTotalResources()];

  initMutexes(cedr_config.getTotalResources(), resource_mutex);
  pretendWeOnlyHaveCPUs(cedr_config.getTotalResources(), hardware_thread_handles);

  task_nodes *tasks = (task_nodes*) calloc(4, sizeof(task_nodes));

  populateFakeCPUTask(&tasks[0], "Task 1", ((long long) 5) * SEC2NANOSEC);
  populateFakeCPUTask(&tasks[1], "Task 2", ((long long) 10) * SEC2NANOSEC);
  populateFakeCPUTask(&tasks[2], "Task 3", ((long long) 0.5) * SEC2NANOSEC);
  populateFakeCPUTask(&tasks[3], "Task 4", ((long long) 50) * SEC2NANOSEC);
  ready_queue.push_back(&tasks[0]);
  ready_queue.push_back(&tasks[1]);
  ready_queue.push_back(&tasks[2]);
  ready_queue.push_back(&tasks[3]);

  SECTION("SIMPLE behaves as expected") {
    cedr_config.scheduler = "SIMPLE";
    performScheduling(cedr_config, ready_queue, hardware_thread_handles, resource_mutex, available_resources);

    REQUIRE(tasks[0].assigned_resource_name == "Core 1");
    REQUIRE(tasks[0].actual_resource_cluster_idx == 0);
    REQUIRE(tasks[1].assigned_resource_name == "Core 2");
    REQUIRE(tasks[1].actual_resource_cluster_idx == 1);
    REQUIRE(tasks[2].assigned_resource_name == "Core 3");
    REQUIRE(tasks[2].actual_resource_cluster_idx == 2);
    REQUIRE(tasks[3].assigned_resource_name == "Core 1");
    REQUIRE(tasks[3].actual_resource_cluster_idx == 0);
  }

  SECTION("MET behaves as expected") {
    cedr_config.scheduler = "MET";
    performScheduling(cedr_config, ready_queue, hardware_thread_handles, resource_mutex, available_resources);

    REQUIRE(tasks[0].assigned_resource_name == "Core 1");
    REQUIRE(tasks[0].actual_resource_cluster_idx == 0);
    REQUIRE(tasks[1].assigned_resource_name == "Core 2");
    REQUIRE(tasks[1].actual_resource_cluster_idx == 1);
    REQUIRE(tasks[2].assigned_resource_name == "Core 3");
    REQUIRE(tasks[2].actual_resource_cluster_idx == 2);
    REQUIRE(tasks[3].assigned_resource_name == "Core 1");
    REQUIRE(tasks[3].actual_resource_cluster_idx == 0);
  }

  SECTION("HEFT_RT behaves as expected") {
    cedr_config.scheduler = "HEFT_RT";
    performScheduling(cedr_config, ready_queue, hardware_thread_handles, resource_mutex, available_resources);

    REQUIRE(tasks[0].assigned_resource_name == "Core 3");
    REQUIRE(tasks[0].actual_resource_cluster_idx == 2);
    REQUIRE(tasks[1].assigned_resource_name == "Core 2");
    REQUIRE(tasks[1].actual_resource_cluster_idx == 1);
    REQUIRE(tasks[2].assigned_resource_name == "Core 3");
    REQUIRE(tasks[2].actual_resource_cluster_idx == 2);
    REQUIRE(tasks[3].assigned_resource_name == "Core 1");
    REQUIRE(tasks[3].actual_resource_cluster_idx == 0);
  }

  SECTION("ETF behaves as expected") {
    cedr_config.scheduler = "ETF";
    performScheduling(cedr_config, ready_queue, hardware_thread_handles, resource_mutex, available_resources);

    REQUIRE(tasks[0].assigned_resource_name == "Core 2");
    REQUIRE(tasks[0].actual_resource_cluster_idx == 1);
    REQUIRE(tasks[1].assigned_resource_name == "Core 3");
    REQUIRE(tasks[1].actual_resource_cluster_idx == 2);
    REQUIRE(tasks[2].assigned_resource_name == "Core 1");
    REQUIRE(tasks[2].actual_resource_cluster_idx == 0);
    REQUIRE(tasks[3].assigned_resource_name == "Core 1");
    REQUIRE(tasks[3].actual_resource_cluster_idx == 0);
  }

  SECTION("EFT behaves as expected") {
    cedr_config.scheduler = "EFT";
    performScheduling(cedr_config, ready_queue, hardware_thread_handles, resource_mutex, available_resources);

    REQUIRE(tasks[0].assigned_resource_name == "Core 3");
    REQUIRE(tasks[0].actual_resource_cluster_idx == 2);
    REQUIRE(tasks[1].assigned_resource_name == "Core 2");
    REQUIRE(tasks[1].actual_resource_cluster_idx == 1);
    REQUIRE(tasks[2].assigned_resource_name == "Core 1");
    REQUIRE(tasks[2].actual_resource_cluster_idx == 0);
    REQUIRE(tasks[3].assigned_resource_name == "Core 1");
    REQUIRE(tasks[3].actual_resource_cluster_idx == 0);
  }
  SECTION("RANDOM behaves as expected") {
    cedr_config.scheduler = "RANDOM";
    cedr_config.random_seed = 0;
    performScheduling(cedr_config, ready_queue, hardware_thread_handles, resource_mutex, available_resources);

    REQUIRE(tasks[0].assigned_resource_name == "Core 2");
    REQUIRE(tasks[0].actual_resource_cluster_idx == 1);
    REQUIRE(tasks[1].assigned_resource_name == "Core 1");
    REQUIRE(tasks[1].actual_resource_cluster_idx == 0);
    REQUIRE(tasks[2].assigned_resource_name == "Core 2");
    REQUIRE(tasks[2].actual_resource_cluster_idx == 1);
    REQUIRE(tasks[3].assigned_resource_name == "Core 3");
    REQUIRE(tasks[3].actual_resource_cluster_idx == 2);
  }
}
