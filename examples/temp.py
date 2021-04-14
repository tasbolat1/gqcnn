import rospy
import cv_bridge
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from perception import DepthImage
from visualization import Visualizer2D as vis

## DATA COLLECTION USING DEXNET PIPELINE


bridge = cv_bridge.CvBridge()

# probably need to change this in code somehow (they are using weird min/maxdepth)
min_depth = 0.25
max_depth = 1.25
BINARY_IM_MAX_VAL = np.iinfo(np.uint8).max
camera_frame = 'camera_depth_optical_frame'
ready = False

def callback(msg):
    global filename, ready
    if ready:
        depth = bridge.imgmsg_to_cv2(msg)
        depth_im = DepthImage(depth, frame=camera_frame)
        np.save('./collect/{}.npy'.format(filename), depth)
        vis.figure(size=(10, 10))
        vis.imshow(depth_im)
        vis.show('./collect/{}.png'.format(filename))
        ready = False
        print('saving ', filename)


def filename_callback(msg):
    global filename, ready
    filename = msg.data
    ready = True


filename_sub = rospy.Subscriber('filename', String, filename_callback)
depth_sub = rospy.Subscriber('/camera/depth/image_meters', Image, callback)


def main():
    '''Initializes and cleanup ros node'''
    rospy.init_node('image_feature')
    try:
        rospy.spin()

    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
if __name__ == '__main__':
    main()
