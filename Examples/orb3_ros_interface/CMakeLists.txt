cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

find_package(OpenCV 4.6.0 REQUIRED)
# find_package(OpenCV 3.2 REQUIRED)
# if(NOT OpenCV_FOUND)
#     message(WARNING "OpenCV 4.4.0 is not found")
#    find_package(OpenCV 2.4.3 QUIET)
#    if(NOT OpenCV_FOUND)
#       message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#    endif()
# endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_package(realsense2 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_ros REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../
${PROJECT_SOURCE_DIR}/../../include
${PROJECT_SOURCE_DIR}/../../include/CameraModels
${PROJECT_SOURCE_DIR}/../../Thirdparty/Sophus
${Pangolin_INCLUDE_DIRS}
)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../lib/libORB_SLAM3.so
-lboost_system
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


include_directories(${PROJECT_NAME}
${realsense_INCLUDE_DIR}
)
# target_link_libraries(${PROJECT_NAME}
# ${realsense2_LIBRARY}
# )
# # Node for monocular camera
# rosbuild_add_executable(Mono
# src/ros_mono.cc
# )

# target_link_libraries(Mono
# ${LIBS}
# )

# Node for monocular camera (Augmented Reality Demo)
#rosbuild_add_executable(MonoAR
#src/AR/ros_mono_ar.cc
#src/AR/ViewerAR.h
#src/AR/ViewerAR.cc
#)

#target_link_libraries(MonoAR
#${LIBS}
#)

# # Node for stereo camera
# rosbuild_add_executable(Stereo
# src/ros_stereo.cc
# )

# target_link_libraries(Stereo
# ${LIBS}
# )

# Node for RGB-D camera
rosbuild_add_executable(RGBD
src/ros_rgbd.cc
)

target_link_libraries(RGBD
${LIBS}
)
set_target_properties(RGBD PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")

rosbuild_add_executable(rgbd_realsense
src/ros_rgbd_realsense_D435i.cc
)

target_link_libraries(rgbd_realsense
${LIBS}
${realsense2_LIBRARY}
${PCL_LIBRARIES}
)
set_target_properties(rgbd_realsense PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")

rosbuild_add_executable(rgbd_sub_topic
src/ros_rgbd_sub_topic.cc
)

target_link_libraries(rgbd_sub_topic
${LIBS}
${realsense2_LIBRARY}
${PCL_LIBRARIES}
)
set_target_properties(rgbd_sub_topic PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")

rosbuild_add_executable(mono_sub_topic
src/ros_mono_sub_topic.cc
)

target_link_libraries(mono_sub_topic
${LIBS}
${realsense2_LIBRARY}
${PCL_LIBRARIES}
)
set_target_properties(mono_sub_topic PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")

# # Node for monocular-inertial camera
# rosbuild_add_executable(Mono_Inertial
# src/ros_mono_inertial.cc
# )

# target_link_libraries(Mono_Inertial
# ${LIBS}
# )

# # Node for stereo-inertial camera
# rosbuild_add_executable(Stereo_Inertial
# src/ros_stereo_inertial.cc
# )

# target_link_libraries(Stereo_Inertial
# ${LIBS}
# )
