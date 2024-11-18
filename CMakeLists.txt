cmake_minimum_required (VERSION 3.11)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(newton-project
	VERSION 1.0.0
	DESCRIPTION "Parallel newton fractal")

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../cmake)

find_package(AdaptiveCpp CONFIG REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT ACPP_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(ACPP_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(ACPP_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})

if(WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "[^/]+/main.cpp")
add_executable(newton src/main.cpp)
add_library(newton_lib ${SOURCES})
target_include_directories(newton_lib PUBLIC include)
#add_sycl_to_target(TARGET newton_lib SOURCES ${SOURCES})
add_sycl_to_target(TARGET newton SOURCES src/main.cpp)
target_link_libraries(newton PUBLIC newton_lib)
target_compile_options(newton_lib PUBLIC 
	$<$<CXX_COMPILER_ID:Clang,GNU>:-Wall -Wextra -Wpedantic>
	$<$<CONFIG:Debug>:-g>
	$<$<CONFIG:Debug>:-O3>
)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB TESTS "test/*.cpp")
enable_testing()
add_executable(utest ${TESTS})
target_compile_options(utest PUBLIC -g -O0)
target_link_libraries(utest newton_lib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(utest)
