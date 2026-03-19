import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse
from scipy.interpolate import splprep, splev

class InferenceNode(Node):
    DEFAULT_PROMPTS = ["traversable road", "dirt road", "off-road trail", "drivable terrain", "gravel path"]
    POINT_X, POINT_Y = 0.50, 0.75

    def __init__(self, model_type: str):
        super().__init__('perception_node')
        self.model_type = model_type.lower()
        self.subscription = self.create_subscription(Image, 'raw_frames', self.listener_callback, 10)
        self.publisher = self.create_publisher(Image, 'processed_frames', 10)
        self.bridge = CvBridge()

        self.history_path = None
        self.last_mask = None
        self.frame_count = 0

        if self.model_type == "yolo":
            from ultralytics import YOLO
            self.model = YOLO("yoloe-26n-seg.pt")
            self.model.set_classes(self.DEFAULT_PROMPTS)
            self.get_logger().info("YOLO Online")
        else:
            from ultralytics import SAM
            self.model = SAM("sam2.1_l.pt")
            self.model.to("cuda")
            self.get_logger().info("SAM Online")

    def listener_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        if self.model_type == "yolo":
            results = self.model.predict(cv_img, conf=0.20, verbose=False)
            if results and results[0].masks is not None:
                mask = np.any(results[0].masks.data.cpu().numpy(), axis=0).astype(np.uint8)
                self._publish_overlay(cv_img, mask)
        else:
            h, w = cv_img.shape[:2]
            cx, cy = int(w * self.POINT_X), int(h * self.POINT_Y)
            results = self.model.predict(cv_img, points=[[cx, cy]], labels=[1], verbose=False)
            if results and results[0].masks is not None:
                # Squeeze ensures shape is (H, W) for SAM
                mask = results[0].masks.data.cpu().numpy().squeeze().astype(np.uint8)
                self._publish_overlay(cv_img, mask)

    def _publish_overlay(self, cv_img, mask):
        h, w = cv_img.shape[:2]
        overlay = cv_img.copy()
        
        # 1. Highlight drivable area
        overlay[mask > 0] = [0, 100, 0] 

        # 2. Calculate Midline
        path_points = []
        # LONGER TRAJECTORY: Scan up to 65% of image height (was 75%)
        # Lowering this value makes the "look-ahead" longer.
        scan_limit = int(h * 0.65) 
        
        for y in range(h - 20, scan_limit, -20):
            nz = np.where(mask[y, :] > 0)[0]
            if len(nz) > 10:
                # Midpoint between leftmost and rightmost boundary
                midpoint = (np.min(nz) + np.max(nz)) // 2
                path_points.append([midpoint, y])

        # 3. Apply Trajectory
        if len(path_points) > 3:
            try:
                pts = np.array(path_points)
                # s=30.0 for a bit more flexibility over longer distances
                tck, u = splprep([pts[:, 0], pts[:, 1]], s=40.0)
                u_fine = np.linspace(0, 1, 40)
                smooth_x, smooth_y = splev(u_fine, tck)
                current_path = np.column_stack((smooth_x.astype(np.int32), smooth_y.astype(np.int32)))

                cv2.polylines(overlay, [current_path], False, (0, 255, 255), 2)
                cv2.circle(overlay, tuple(current_path[-1]), 4, (255, 255, 0), -1)
            except:
                pass

        # 4. Final Merge and Publish
        # Adjust weighted alpha (0.8/0.2) to make the mask less intrusive if desired
        result = cv2.addWeighted(cv_img, 0.8, overlay, 0.2, 0)
        self.publisher.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolo", "sam"], default="yolo")
    parsed, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = InferenceNode(model_type=parsed.model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()