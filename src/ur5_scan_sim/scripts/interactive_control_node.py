#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from std_srvs.srv import Trigger

class InteractiveControlNode(Node):
    def __init__(self):
        super().__init__('interactive_control_node')
        self.server = InteractiveMarkerServer(self, 'ur5_scan_controls')
        
        # Service client to trigger the robot motion
        self.client = self.create_client(Trigger, '/ur5_scanner/start_scan')
        
        self.create_interactive_marker()
        self.get_logger().info("Interactive Marker Server Started. Look for it in Rviz!")

    def create_interactive_marker(self):
        # Create an interactive marker for our server
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.name = "scan_control_marker"
        int_marker.description = "UR5 Scanner Controls"
        int_marker.pose.position.z = 1.0 # Float it in the air
        int_marker.scale = 0.2

        # Create a visual box marker
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = 0.1
        box_marker.scale.y = 0.1
        box_marker.scale.z = 0.1
        box_marker.color.r = 0.0
        box_marker.color.g = 1.0
        box_marker.color.b = 0.0
        box_marker.color.a = 1.0

        # Create a non-interactive control which contains the box
        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.markers.append(box_marker)
        int_marker.controls.append(box_control)

        # Create a control that reacts to mouse clicks
        button_control = InteractiveMarkerControl()
        button_control.name = "button_control"
        button_control.interaction_mode = InteractiveMarkerControl.BUTTON
        button_control.always_visible = True
        int_marker.controls.append(button_control)

        self.server.insert(int_marker, feedback_callback=self.process_feedback)
        self.server.applyChanges()

    def process_feedback(self, feedback):
        if feedback.event_type == feedback.BUTTON_CLICK:
            self.get_logger().info("Scan button clicked! Requesting scan service...")
            if not self.client.service_is_ready():
                self.get_logger().error("Scan service not available!")
                return
            
            req = Trigger.Request()
            self.client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
