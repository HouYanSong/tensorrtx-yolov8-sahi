cmake_minimum_required(VERSION 3.10)

project(yolov8_sahi)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)

include_directories("/usr/include/eigen3")

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# ============= tensorrt ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
  include_directories(/usr/include/x86_64-linux-gnu)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif()

add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)
set_target_properties(myplugins PROPERTIES CUDA_ARCHITECTURES "87")

# ============ opencv ============
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# =========== bytetrack lib ===========
include_directories(${PROJECT_SOURCE_DIR}/bytetrack/include)
file(GLOB_RECURSE SRCS01 ${PROJECT_SOURCE_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${SRCS01})
target_link_libraries(bytetrack cudart nvinfer ${OpenCV_LIBS})
set_target_properties(bytetrack PROPERTIES CUDA_ARCHITECTURES "87")

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(yolov8_sahi ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})
add_executable(yolov8_sahi_track ${PROJECT_SOURCE_DIR}/track.cpp ${SRCS})

target_link_libraries(yolov8_sahi nvinfer)
target_link_libraries(yolov8_sahi cudart)
target_link_libraries(yolov8_sahi myplugins)
target_link_libraries(yolov8_sahi ${OpenCV_LIBS})
set_target_properties(yolov8_sahi PROPERTIES CUDA_ARCHITECTURES "87")

target_link_libraries(yolov8_sahi_track nvinfer)
target_link_libraries(yolov8_sahi_track cudart)
target_link_libraries(yolov8_sahi_track myplugins)
target_link_libraries(yolov8_sahi_track ${OpenCV_LIBS})
target_link_libraries(yolov8_sahi_track bytetrack)
set_target_properties(yolov8_sahi_track PROPERTIES CUDA_ARCHITECTURES "87")
