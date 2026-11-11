cmake_minimum_required(VERSION 3.16)
project(omni_upper_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(REPO "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
include_directories(${REPO} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS}
                    ${REPO}/io/serial/include ${REPO}/io/hikrobot/include)
add_executable(omni_handoff_policy_test ../omni_handoff_policy_test.cpp)
add_executable(omni_geometry_test ../omni_geometry_test.cpp
               ${REPO}/tasks/omniperception/geometry.cpp ${REPO}/tasks/auto_aim/armor.cpp)
target_link_libraries(omni_geometry_test PRIVATE ${OpenCV_LIBS} yaml-cpp Eigen3::Eigen)
# Compile the actual modified integration translation units, without linking devices/OpenVINO.
# No ROS2 macro here: hardware-free syntax validation of the non-ROS entry point.
add_library(omni_latest_syntax OBJECT ${REPO}/src/sentry_multithread.cpp
                                    ${REPO}/tasks/omniperception/perceptron.cpp)
target_link_libraries(omni_latest_syntax PRIVATE fmt::fmt spdlog::spdlog
                      nlohmann_json::nlohmann_json Eigen3::Eigen yaml-cpp)
enable_testing()
add_test(NAME handoff_policy COMMAND omni_handoff_policy_test)
add_test(NAME calibrated_geometry COMMAND omni_geometry_test "${CMAKE_BINARY_DIR}/fixtures")
