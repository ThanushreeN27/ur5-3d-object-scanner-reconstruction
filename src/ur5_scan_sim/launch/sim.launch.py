import os

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
    pkg_ur5_scan_sim = get_package_share_directory('ur5_scan_sim')
    
    # Paths to files
    urdf_xacro_file = os.path.join(pkg_ur5_scan_sim, 'urdf', 'ur5_with_camera.xacro')
    world_file = os.path.join(pkg_ur5_scan_sim, 'worlds', 'scan_world.sdf')
    
    # Robot State Publisher
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

    # Make sure Gazebo can find the gz_ros2_control plugin
    set_env = ExecuteProcess(
        cmd=['export', 'GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH:/opt/ros/jazzy/lib'],
        shell=True
    )

    # Gazebo Harmonic (gz-sim)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])]
        ),
        launch_arguments={"gz_args": f"-r -v 4 {world_file}"}.items(),
    )

    # Spawn robot in Gazebo
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

    # ROS-GZ Bridge for Camera
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

    # Controllers Spawner
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

    # Delay spawner until robot is spawned
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

    rviz_config_file = os.path.join(pkg_ur5_scan_sim, 'config', 'ur5_scan.rviz')

    node_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": True}],
    )

    # Arguments for starting nodes automatically
    start_all = LaunchConfiguration('start_all', default='false')

    # Data & Reconstruction Nodes
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

    return LaunchDescription(
        [
            DeclareLaunchArgument('start_all', default='false', description='Whether to start all data capture and reconstruction nodes'),
            set_env,
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
        ]
    )
