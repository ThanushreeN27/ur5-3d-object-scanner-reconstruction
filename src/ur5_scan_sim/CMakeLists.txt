cmake_minimum_required(VERSION 3.8)
project(ur5_scan_sim)

# --- THE SHOPPING LIST (Build System) ---
# This file tells the computer which tools and folders it needs 
# to "build" and run our ROS project.

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# STEP 1: Find the ROS 2 tools we need
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(xacro REQUIRED)


# STEP 2: Install Folders
# These are the folders that contain our settings, worlds, and 3D shapes.
install(DIRECTORY
  launch
  urdf
  config
  worlds
  DESTINATION share/${PROJECT_NAME}
)

# STEP 3: Install Python Scripts
# These are the actual programs (the "AI" and "Control" nodes)
install(PROGRAMS
  scripts/motion_planning_node.py
  scripts/camera_node.py
  scripts/reconstruct_3d.py
  scripts/image_processing_node.py
  scripts/reconstruction_node.py
  scripts/view_results.py
  DESTINATION lib/${PROJECT_NAME}
)

# STEP 4: Build Testing (Linter Checks)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # The lines below skip checks that are too strict for a learning project
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# STEP 5: Finish the Package
ament_package()
