# CMakeLists do app — compila o llama.cpp junto com a ponte JNI.
# O código do llama.cpp é clonado por android/scripts/fetch-llama-cpp.sh
cmake_minimum_required(VERSION 3.22)
project(gptmachado CXX)

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(gptmachado SHARED native-lib.cpp)
target_link_libraries(gptmachado PRIVATE llama log)
