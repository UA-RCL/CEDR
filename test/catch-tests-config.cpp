#define CEDR_UNIT_TESTING
#include "catch.hpp"
#include "catch-tests-utils.hpp"
#undef CEDR_UNIT_TESTING

#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Log.h>

#define LOG_LEVEL "NONE"

TEST_CASE("Testing Json parser (config_manager.cpp) with dummy json") {
  ConfigManager cedr_config = ConfigManager();

  SECTION("Default values parse as expected") {
    REQUIRE(cedr_config.getTotalResources() == 3);
    REQUIRE(cedr_config.getResourceArray()[(uint8_t) resource_type::cpu] == 3);
    REQUIRE(cedr_config.getResourceArray()[(uint8_t) resource_type::fft] == 0);
    REQUIRE(cedr_config.getResourceArray()[(uint8_t) resource_type::mmult] == 0);
    REQUIRE(cedr_config.getResourceArray()[(uint8_t) resource_type::gpu] == 0);
    REQUIRE(cedr_config.getScheduler() == "SIMPLE");
    REQUIRE(!cedr_config.getCacheSchedules());
    REQUIRE(cedr_config.getEnableQueueing());
    REQUIRE(!cedr_config.getUsePAPI());
    REQUIRE(!cedr_config.getLoosenThreadPermissions());
    REQUIRE(cedr_config.getFixedPeriodicInjection());
    REQUIRE(!cedr_config.getExitWhenIdle());
    REQUIRE(cedr_config.getRandomSeed() == 0);
  }

  cedr_config.parseConfig("test_config.json");

  SECTION("Resources counter works as expected") {
    REQUIRE(cedr_config.getTotalResources() == 6);
    REQUIRE(cedr_config.getResourceArray()[(uint8_t) resource_type::cpu] == 3);
    REQUIRE(cedr_config.getResourceArray()[(uint8_t) resource_type::fft] == 1);
    REQUIRE(cedr_config.getResourceArray()[(uint8_t) resource_type::mmult] == 1);
    REQUIRE(cedr_config.getResourceArray()[(uint8_t) resource_type::gpu] == 1);
  }
 
  SECTION("Features values parse as expected") {
    REQUIRE(cedr_config.getScheduler() == "NONEXISTENT");
    REQUIRE(cedr_config.getEnableQueueing());
    REQUIRE(!cedr_config.getCacheSchedules());
    REQUIRE(cedr_config.getEnableQueueing());
    REQUIRE(cedr_config.getUsePAPI());
    REQUIRE(!cedr_config.getLoosenThreadPermissions());
    REQUIRE(cedr_config.getFixedPeriodicInjection());
    REQUIRE(cedr_config.getExitWhenIdle());
    REQUIRE(cedr_config.getRandomSeed() == 5000);
  }

  SECTION("PAPI counters values parse as expected") {
    std::list<std::string> x = cedr_config.getPAPICounters();
    std::list<std::string>::iterator it = x.begin();
    REQUIRE(*it == "ONE");
    std::advance(it,1);
    REQUIRE(*it == "TWO");
    std::advance(it,1);
    REQUIRE(*it == "THREE");
  }
}
