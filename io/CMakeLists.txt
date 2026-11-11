cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "HikRobot process isolation requires Linux")
endif()

add_subdirectory(serial)

# 创建目标 io
add_library(io STATIC 
    hikrobot/hikrobot.cpp    
    mindvision/mindvision.cpp  
    usbcamera/usbcamera.cpp  
    camera.cpp
    cboard.cpp
    dm_imu/dm_imu.cpp
    gimbal/gimbal.cpp
)

# hikrobot
target_include_directories(io PUBLIC hikrobot/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC hikrobot/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC hikrobot/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

# mindvision
target_include_directories(io PUBLIC mindvision/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_compile_definitions(io PRIVATE _GNU_SOURCE)
target_link_libraries(io MVSDK usb-1.0 yaml-cpp serial Threads::Threads)

# MVS is loaded only by the exec-isolated SDK worker, never by the application.
add_executable(hikrobot_worker
  hikrobot/worker_main.cpp
  hikrobot/capture_worker.cpp
  hikrobot/usb_reset.cpp
  ../tools/logger.cpp
)
target_include_directories(hikrobot_worker PRIVATE hikrobot/include)
target_compile_definitions(hikrobot_worker PRIVATE _GNU_SOURCE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(hikrobot_worker PRIVATE hikrobot/lib/amd64)
else()
  target_link_directories(hikrobot_worker PRIVATE hikrobot/lib/arm64)
endif()
target_link_libraries(hikrobot_worker PRIVATE
  MvCameraControl usb-1.0 ${OpenCV_LIBS} spdlog::spdlog fmt::fmt Threads::Threads)
set_target_properties(hikrobot_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(hikrobot_usb_reset
  hikrobot/reset_main.cpp hikrobot/usb_reset.cpp ../tools/logger.cpp)
target_link_libraries(hikrobot_usb_reset PRIVATE usb-1.0 spdlog::spdlog fmt::fmt Threads::Threads)
set_target_properties(hikrobot_usb_reset PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(hikrobot_launcher hikrobot/worker_launcher.cpp)
target_compile_definitions(hikrobot_launcher PRIVATE _GNU_SOURCE)
target_compile_definitions(hikrobot_usb_reset PRIVATE _GNU_SOURCE)
set_target_properties(hikrobot_launcher PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_dependencies(io hikrobot_worker hikrobot_usb_reset hikrobot_launcher)

################## 检查 ROS 环境 ##################
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)

# 如果找不到 ROS 环境，则跳过 ROS 相关的部分
if(NOT ament_cmake_FOUND OR NOT rclcpp_FOUND OR NOT std_msgs_FOUND OR NOT rosidl_typesupport_cpp_FOUND OR NOT sp_msgs_FOUND)
    message(WARNING "ROS2 not found, skipping ROS2 specific code.")
else()
    message(STATUS "ROS2 environment found, compiling ROS2-related code.")
    
    # 添加 ROS2 相关的文件
    target_sources(io PRIVATE ros2/publish2nav.cpp ros2/ros2.cpp ros2/subscribe2nav.cpp)

    # 设置 ROS2 依赖
    include_directories(${std_msgs_INCLUDE_DIRS})
    ament_target_dependencies(io rclcpp std_msgs sp_msgs)
endif()