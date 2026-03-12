cmake_minimum_required(VERSION 3.8)
project(ur5_scan_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(xacro REQUIRED)


install(DIRECTORY
  launch
  urdf
  config
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/motion_planning_node.py
  scripts/camera_node.py
  scripts/reconstruct_3d.py
  scripts/image_processing_node.py
  scripts/reconstruction_node.py
  scripts/interactive_control_node.py
  scripts/view_results.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
