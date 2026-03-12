import os
import sys

# --- STEP 1: Import the launch tools ---
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """
    This is the "DIRECTOR" of the show.
    It starts the simulation, the robot, the camera, and all the 3D map tools at the same time.
    """
    pkg_ur5_scan_sim = get_package_share_directory('ur5_scan_sim')
    
    # --- Part 1: File Paths ---
    # These tell the computer where to find our Robot shape and our 3D World
    urdf_xacro_file = os.path.join(pkg_ur5_scan_sim, 'urdf', 'ur5_with_camera.xacro')
    world_file = os.path.join(pkg_ur5_scan_sim, 'worlds', 'scan_world.sdf')
    
    # --- Part 2: Robot State Publisher ---
    # This node calculates exactly where every bone (joint) of the robot is.
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            urdf_xacro_file,
            " ",
            "name:=ur5",
            " ",
            "ur_type:=ur5",
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # --- Part 3: Plugin Path ---
    # Make sure the simulation can find its "brain" plugin
    if 'GZ_SIM_SYSTEM_PLUGIN_PATH' in os.environ:
        os.environ['GZ_SIM_SYSTEM_PLUGIN_PATH'] += ':/opt/ros/jazzy/lib'
    else:
        os.environ['GZ_SIM_SYSTEM_PLUGIN_PATH'] = '/opt/ros/jazzy/lib'

    # --- Part 4: Gazebo (The Game Engine) ---
    # This starts the actual 3D simulation window
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])]
        ),
        launch_arguments={"gz_args": f"-r -v 4 {world_file}"}.items(),
    )

    # --- Part 5: Spawn the Robot ---
    # This "drops" the virtual robot into the 3D world
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string", robot_description_content,
            "-name", "ur5",
            "-allow_renaming", "true",
            "-z", "0.1"
        ],
    )

    # --- Part 6: ROS-GZ Bridge ---
    # This acts like an "Interpreter" so Rviz can understand the simulation camera
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        output="screen"
    )

    # --- Part 7: Controllers ---
    # These "drivers" control the actual movement of the robot's arms
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )

    # Wait for the robot to exist before starting its movement drivers
    delay_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    delay_joint_trajectory_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[joint_trajectory_controller_spawner],
        )
    )

    # --- Part 8: Rviz (The Visualizer) ---
    # This opens the window where we see the 3D map and HUD
    rviz_config_file = os.path.join(pkg_ur5_scan_sim, 'config', 'ur5_scan.rviz')
    node_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": True}],
    )

    # --- Part 9: AI & Map Nodes ---
    # These only start if we say "--start_all:=true" when running the command
    start_all = LaunchConfiguration('start_all', default='false')

    camera_node = Node(
        package="ur5_scan_sim",
        executable="camera_node.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(start_all)
    )

    processing_node = Node(
        package="ur5_scan_sim",
        executable="image_processing_node.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(start_all)
    )

    reconstruction_node = Node(
        package="ur5_scan_sim",
        executable="reconstruction_node.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(start_all)
    )

    interactive_node = Node(
        package="ur5_scan_sim",
        executable="interactive_control_node.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    motion_node = Node(
        package="ur5_scan_sim",
        executable="motion_planning_node.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # --- FINISH: Combine everything into one big list ---
    return LaunchDescription(
        [
            DeclareLaunchArgument('start_all', default_value='false', description='Whether to start all data capture and reconstruction nodes'),
            gazebo,
            node_robot_state_publisher,
            spawn_entity,
            gz_bridge,
            node_rviz,
            delay_joint_state_broadcaster,
            delay_joint_trajectory_controller,
            camera_node,
            processing_node,
            reconstruction_node,
            interactive_node,
            motion_node,
        ]
    )
