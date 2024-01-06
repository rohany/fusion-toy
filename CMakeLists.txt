cmake_minimum_required(VERSION 3.13.4)
project(fusion-toy LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ Standard")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIR.cmake in: ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

llvm_map_components_to_libnames(llvm_libs support core)

add_executable(test test.cpp)
target_link_libraries(test
    PRIVATE 
      ${dialect_libs}
      ${conversion_libs}
      ${llvm_libs}
      MLIRIR
      MLIROptLib
      MLIRSupport
      MLIRParser
      MLIRPass
      MLIRTransforms
      # The principled way to do this is to link in just the LLVM
      # libraries we need, but I think just using MLIRExecutionEngine
      # will give us all the transitive dependencies we need.
      MLIRExecutionEngine
)
# TODO (rohany): This came from https://groups.google.com/g/llvm-dev/c/NEnHjC9MdJk, which
#  is defined in llvm-config --cxxflags. I don't know the best way to thread this through
#  to cmake build, but that's a problem for another day. Adding this fixed some missing
#  symbol errors.
target_compile_options(test PRIVATE -fno-rtti)

add_executable(legate_toy legate_toy.cpp)
target_link_libraries(legate_toy
        PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        ${llvm_libs}
        MLIRIR
        MLIROptLib
        MLIRSupport
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRExecutionEngine
)
target_compile_options(legate_toy PRIVATE -fno-rtti)

add_executable(users_repro users_repro.cpp)
target_link_libraries(users_repro
        PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        ${llvm_libs}
        MLIRIR
        MLIROptLib
        MLIRSupport
        MLIRParser
        MLIRPass
        MLIRTransforms
        # The principled way to do this is to link in just the LLVM
        # libraries we need, but I think just using MLIRExecutionEngine
        # will give us all the transitive dependencies we need.
        MLIRExecutionEngine
        )
# TODO (rohany): This came from https://groups.google.com/g/llvm-dev/c/NEnHjC9MdJk, which
#  is defined in llvm-config --cxxflags. I don't know the best way to thread this through
#  to cmake build, but that's a problem for another day. Adding this fixed some missing
#  symbol errors.
target_compile_options(users_repro PRIVATE -fno-rtti)
