#!/usr/bin/env python3

# --- STEP 1: Import the tools we need ---
import rclpy
from rclpy.node import Node
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from std_srvs.srv import Trigger

class InteractiveControlNode(Node):
    """
    This is the "REMOTE CONTROL" node.
    It puts a clickable green cube in the Rviz 3D window.
    When you click it, the robot starts its scanning movement.
    """
    def __init__(self):
        # Give this node a name
        super().__init__('interactive_control_node')
        # This creates the "Server" that handles 3D buttons
        self.server = InteractiveMarkerServer(self, 'ur5_scan_controls')
        
        # This is the "Phone Line" used to call the Motion Planner and say "START!"
        self.client = self.create_client(Trigger, '/ur5_scanner/start_scan')
        
        # Create the button on the screen
        self.create_interactive_marker()
        self.get_logger().info("Remote Control Server Started. Look for the GREEN CUBE in Rviz!")

    def create_interactive_marker(self):
        """This function draws the 3D button (Green Cube) in Rviz."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.name = "scan_control_marker"
        int_marker.description = "UR5 Scanner Controls"
        int_marker.pose.position.z = 1.0 # Float the cube 1 meter high in the air
        int_marker.scale = 0.2

        # --- Part A: The Visual Shape ---
        # Make a simple green cube
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = 0.1
        box_marker.scale.y = 0.1
        box_marker.scale.z = 0.1
        box_marker.color.r = 0.0
        box_marker.color.g = 1.0 # Green
        box_marker.color.b = 0.0
        box_marker.color.a = 1.0 # Solid (not see-through)

        # Connect the shape to the marker
        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.markers.append(box_marker)
        int_marker.controls.append(box_control)

        # --- Part B: The Click Logic ---
        # Make the cube react to mouse clicks
        button_control = InteractiveMarkerControl()
        button_control.name = "button_control"
        button_control.interaction_mode = InteractiveMarkerControl.BUTTON
        button_control.always_visible = True
        int_marker.controls.append(button_control)

        # Add the finished marker to the server
        self.server.insert(int_marker, feedback_callback=self.process_feedback)
        self.server.applyChanges()

    def process_feedback(self, feedback):
        """This function runs only when you click the button in Rviz."""
        if feedback.event_type == feedback.BUTTON_CLICK:
            self.get_logger().info("Green Cube clicked! Telling the robot to start...")
            
            # Check if the Motion Planner is actually running
            if not self.client.service_is_ready():
                self.get_logger().error("Motion Planner is NOT ready. Is the other node running?")
                return
            
            # Send the "START" command
            req = Trigger.Request()
            self.client.call_async(req)

def main(args=None):
    # Start ROS 2
    rclpy.init(args=args)
    # Start the remote control
    node = InteractiveControlNode()
    try:
        # Keep running
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Safely close
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
