#define CEDR_UNIT_TESTING
#include "catch.hpp"
#include "catch-tests-utils.hpp"
#include "threads.hpp"
#undef CEDR_UNIT_TESTING

#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Log.h>

#define LOG_LEVEL "NONE"

TEST_CASE("Testing hardware threads", "[CPU ONLY]") {
  ConfigManager cedr_config = ConfigManager();
  cedr_config.resource_array[(uint8_t) resource_type::cpu] = 2; 
  cedr_config.loosen_thread_permissions = true;

  const unsigned int totalResources = cedr_config.getTotalResources();
  pthread_t resource_handle[totalResources];
  worker_thread hardware_thread_handle[totalResources];
  pthread_mutex_t resource_mutex[totalResources];
 
  SECTION("Threads initialized as Expected") {
    initializeThreads(cedr_config, resource_handle, hardware_thread_handle, resource_mutex);
    REQUIRE(hardware_thread_handle[0].resource_name == "Core 1");
    REQUIRE(hardware_thread_handle[0].thread_resource_type == resource_type::cpu);
    REQUIRE(hardware_thread_handle[0].resource_state == 0);
    REQUIRE(hardware_thread_handle[0].task == nullptr);
    REQUIRE(hardware_thread_handle[1].resource_name == "Core 2");
    REQUIRE(hardware_thread_handle[1].thread_resource_type == resource_type::cpu);
    REQUIRE(hardware_thread_handle[1].resource_state == 0);
    REQUIRE(hardware_thread_handle[1].task == nullptr);
  }
}
