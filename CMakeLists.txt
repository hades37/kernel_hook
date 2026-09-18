cmake_minimum_required(VERSION 3.16)
project(msprof_hook CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(msprof_hook SHARED src/msprof_hook.cpp)
target_include_directories(msprof_hook PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(msprof_hook PRIVATE -O2 -Wall -Wextra -fPIC)
target_link_libraries(msprof_hook PRIVATE dl)

# keep symbol names exactly as the exported CANN API
set_target_properties(msprof_hook PROPERTIES
    OUTPUT_NAME msprof_hook
    PREFIX "lib"
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)
