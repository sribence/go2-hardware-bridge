#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2

pub = None

def on_depth(msg):
    global pub
    if pub is None or pub.get_num_connections() == 0:
        return

    try:
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        valid_mask = (depth > 150) & (depth < 5000)
        
        depth_scaled = np.zeros_like(depth, dtype=np.uint8)
        depth_scaled[valid_mask] = np.clip(255 - ((depth[valid_mask] - 150) * 255.0 / 4000.0), 0, 255).astype(np.uint8)
        
        colorized = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        colorized[~valid_mask] = [15, 15, 20]
        
        success, encoded = cv2.imencode(".jpg", colorized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if success:
            comp_msg = CompressedImage()
            comp_msg.header = msg.header
            comp_msg.format = "jpeg"
            comp_msg.data = encoded.tobytes()
            pub.publish(comp_msg)
    except Exception as e:
        rospy.logerr_throttle(2.0, "Depth colorizer error: " + str(e))

def main():
    global pub
    rospy.init_node("depth_colorizer", anonymous=True)
    pub = rospy.Publisher("/camera/depth/colorized/compressed", CompressedImage, queue_size=1)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, on_depth, queue_size=1, buff_size=2**22)
    rospy.loginfo("Depth colorizer running: /camera/depth/image_rect_raw -> /camera/depth/colorized/compressed")
    rospy.spin()

if __name__ == "__main__":
    main()
