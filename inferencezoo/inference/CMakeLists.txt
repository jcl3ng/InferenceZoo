cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SonicInfer)

# set(CMAKE_PREFIX_PATH "C:/include/libtorch/share/cmake/Torch")
# list(APPEND CMAKE_PREFIX_PATH "C:/libtorch")
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${TORCH_LIBRARY}") 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(SonicInfer torchscript.cpp)
target_link_libraries(SonicInfer "${TORCH_LIBRARIES}")
set_property(TARGET SonicInfer PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET deepsmlm_interface
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:deepsmlm_interface>)
endif (MSVC)
# cmake -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" ..