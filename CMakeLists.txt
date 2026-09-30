cmake_minimum_required(VERSION 3.22)
project(columnar-analytics-engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(columnar_analytics_core src/version.cpp)
target_include_directories(columnar_analytics_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(columnar_analytics_core PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(columnar_analytics_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(columnar-analytics-engine src/main.cpp)
target_link_libraries(columnar-analytics-engine PRIVATE columnar_analytics_core)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
