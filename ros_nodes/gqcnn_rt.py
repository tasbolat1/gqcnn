#!/usr/bin/env python

from __future__ import division, print_function

import rospy

import time
import os

#import torch
import numpy as np
import cv2

from std_msgs.msg import String

from tf import transformations as tft
from dougsm_helpers.timeit import TimeIt

from autolab_core import Point, Logger

from gqcnn.grasping import Grasp2D, SuctionPoint2D, GraspAction
from gqcnn.msg import GQCNNGrasp
from gqcnn.srv import GQCNNGraspPlanner, GQCNNGraspPlannerSegmask
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage

from sensor_msgs.msg import Image, CameraInfo
from visualization import Visualizer2D as vis
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

import cv_bridge
bridge = cv_bridge.CvBridge()

TimeIt.print_output = False

namespace  = 'gqcnn'
NUMBER_ITER = 5

class GqCNNRt:
    def __init__(self):
        # Get the camera parameters
        cam_info_topic = rospy.get_param('gqcnn/camera/info_topic')
        self.camera_info_msg = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.cam_K = np.array(self.camera_info_msg.K).reshape((3, 3))
        self.gripper = rospy.get_param("~gripper")


        #self.img_pub = rospy.Publisher('~visualisation', Image, queue_size=1)
        #self.cmd_pub = rospy.Publisher('~predict', Grasp, queue_size=1)

        self.base_frame = rospy.get_param('~camera/robot_base_frame')
        self.camera_frame = rospy.get_param('~camera/camera_frame')
        self.cam_fov = rospy.get_param('~camera/fov')

        #self.counter = 0
        self.curr_depth_img = None
        self.curr_img_time = 0
        #self.last_image_pose = None
        #self.prev_mp = None
        #rospy.Subscriber(rospy.get_param('~camera/depth_topic'), Image, self._depth_img_callback, queue_size=1)
        
        # For getting camera pose
        camera_pose = tfh.current_robot_pose(self.base_frame, self.camera_frame)
        self.camera_rot = tft.quaternion_matrix(tfh.quaternion_to_list(camera_pose.orientation))[0:3, 0:3]
        self.cam_p = camera_pose.position
        #self.imw, self.imh = None, None
        #self.depth = None

        # DEXNET SRV COMM:
        rospy.wait_for_service("%s/grasp_planner" % (namespace))
        rospy.wait_for_service("%s/grasp_planner_segmask" % (namespace))
        self.plan_grasp = rospy.ServiceProxy("%s/grasp_planner" % (namespace),
                                    GQCNNGraspPlanner)

        self.ok = True
        self.get_grasp = False


        self.grasp_publisher = rospy.Publisher('~grasp_output', PoseStamped, queue_size=10)
        rospy.Subscriber('~grasp_input', String, self.command_callback)

    def command_callback(self, msg):
        if msg.data == 'calculate_grasp':
            self.get_grasp = True
        elif msg.data == 'stop':
            self.get_grasp = False
            self.ok = False
        else:
            print('Bad command')

    def go(self):
        depth_raw = rospy.wait_for_message(rospy.get_param('~camera/depth_topic'), Image)
        depth = bridge.imgmsg_to_cv2(depth_raw)

        #self.imh, self.imw = depth.shape
        #self.depth = depth

        depth_im = DepthImage(depth, frame=self.camera_frame)
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                   3]).astype(np.uint8),
                         frame=self.camera_frame)

        
        grasp_resp = self.plan_grasp(color_im.rosmsg, depth_im.rosmsg,
                                self.camera_info_msg)
        grasp = grasp_resp.grasp
        grasp_type = grasp.grasp_type
        center = Point(np.array([grasp.center_px[0], grasp.center_px[1]]),
               frame=self.camera_frame)

        print(self.camera_info_msg)
        print(self.cam_K.shape)

        camera_intr = CameraIntrinsics("primesense_overhead",
                                                fx=self.cam_K[0][0],
                                                fy=self.cam_K[1][1],
                                                cx=self.cam_K[0][2],
                                                cy=self.cam_K[1][2],
                                                width=int(self.camera_info_msg.width),
                                                height=int(self.camera_info_msg.height))

        grasp_2d = Grasp2D(center,
                           grasp.angle,
                           grasp.depth,
                           width=0.08,
                           camera_intr=camera_intr)

        thumbnail = DepthImage(bridge.imgmsg_to_cv2(
            grasp.thumbnail, desired_encoding="passthrough"),
                               frame=self.camera_frame)
        #print(grasp.q_value)
        action = GraspAction(grasp_2d, grasp.q_value, thumbnail)

        #vis.figure(size=(10, 10))
        # vis.imshow(depth_im, vmin=0.6, vmax=0.9)
        # vis.imshow(depth_im)

        # vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        # vis.title("Planned grasp on depth %.3f (Q=%.3f)" % (grasp.depth, action.q_value))
        # vis.show()

        return grasp_2d, action, depth_im

    def draw_prediction(self, grasp, action, depth_im):
        vis.figure(size=(10, 10))
        vis.imshow(depth_im)
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp on depth %.3f (Q=%.3f)" % (grasp.depth, action.q_value))
        vis.show()

    def publish_grasp(self, grasp):
        # Create `GQCNNGrasp` return msg and populate it.
        gqcnn_grasp = GQCNNGrasp()
        gqcnn_grasp.q_value = grasp.q_value
        gqcnn_grasp.pose = grasp.grasp.pose().pose_msg
        if isinstance(grasp.grasp, Grasp2D):
            gqcnn_grasp.grasp_type = GQCNNGrasp.PARALLEL_JAW
        elif isinstance(grasp.grasp, SuctionPoint2D):
            gqcnn_grasp.grasp_type = GQCNNGrasp.SUCTION
        else:
            rospy.logerr("Grasp type not supported!")
            raise rospy.ServiceException("Grasp type not supported!")

        # Store grasp representation in image space.
        gqcnn_grasp.center_px[0] = grasp.grasp.center[0]
        gqcnn_grasp.center_px[1] = grasp.grasp.center[1]
        gqcnn_grasp.angle = grasp.grasp.angle
        gqcnn_grasp.depth = grasp.grasp.depth
        gqcnn_grasp.thumbnail = grasp.image.rosmsg

        # Create and publish the pose alone for easy visualization of grasp
        # pose in Rviz.
        pose_stamped = PoseStamped()

        # pose is taken from Grasp2D method. Returns pose in camera frame of reference(KIV)
        pose_stamped.pose = grasp.grasp.pose().pose_msg

        # Use simpler method for getting pose
        #pose = np.linalg.inv(cam_K)*np.array([gqcnn_grasp.center_px[0], gqcnn_grasp.center_px[1], 1.0]) + np.array([cam_p.x, cam_p.y, cam_p.z])
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.camera_frame
        pose_stamped.header = header
        self.grasp_publisher.publish(pose_stamped)

    def publish_pos(self, point, orientation):
        pose_stamped = PoseStamped()

        pose_stamped.pose.position.x = point[0]
        pose_stamped.pose.position.y = point[1]
        pose_stamped.pose.position.z = point[2]
        pose_stamped.pose.orientation = orientation

        # Use simpler method for getting pose
        #pose = np.linalg.inv(cam_K)*np.array([gqcnn_grasp.center_px[0], gqcnn_grasp.center_px[1], 1.0]) + np.array([cam_p.x, cam_p.y, cam_p.z])
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.camera_frame
        pose_stamped.header = header
        self.grasp_publisher.publish(pose_stamped)


    def run(self):
        while(self.ok):

            if self.get_grasp:
                q_values = []
                actions = []
                grasps = []
                depth_imgs = []
                for i in range(NUMBER_ITER):
                    print('Iteration: ', i+1)
                    grasp_2d, action, depth_im = self.go()
                    q_values.append(action.q_value)
                    grasps.append(grasp_2d)
                    actions.append(action)
                    depth_imgs.append(depth_im)

                best_indx = np.argmax(np.array(q_values))
                best_grasp = grasps[best_indx]
                best_action = actions[best_indx]
                best_depth = depth_imgs[best_indx]
                self.get_grasp = False

                print('Q values:', q_values)
                print('Best q', best_action.q_value)
                print('Angle: ', best_grasp.angle)
                print('Grasp Center: ', best_grasp.center)
                print('Grasp depth: ', best_grasp.depth)
                

                #imh, imw = self.imh, self.imw

                #x = ((np.vstack((np.linspace(imw // 2, imw // 2, self.imw, np.float), )*best_depth.shape[0]) - self.cam_K[0, 2])/self.cam_K[0, 0] * best_depth).flatten()
                #x = (best_grasp.center[0]- self.cam_K[0, 2]) / self.cam_K[0, 0]
                #y = (best_grasp.center[1]- self.cam_K[1, 2]) / self.cam_K[1, 1]
                #y = ((np.vstack((np.linspace(imh // 2, imh // 2, best_depth.shape[0], np.float), )*best_depth.shape[1]).T - self.cam_K[1,2])/self.cam_K[1, 1] * best_depth).flatten()
                #pos = np.dot(self.camera_rot, np.stack((x, y, best_grasp.depth))).T + np.array([[self.cam_p.x, self.cam_p.y, self.cam_p.z]])

                #print('x, y, pos:', x, y, pos)
                point_3d = best_grasp.depth * np.linalg.inv(self.cam_K).dot(np.array([best_grasp.center[0],best_grasp.center[1],1.0]))
                print('new point:', point_3d)
                point_3d[0] += self.cam_p.x
                point_3d[1] += self.cam_p.y
                point_3d[2] -= self.cam_p.z
                print('go to (x,y,z): ', point_3d[0], point_3d[1], point_3d[2])


                angle = best_grasp.angle
                angle -= np.arcsin(self.camera_rot[0, 1])  # Correct for the rotation of the camera
                angle = (angle + np.pi/2) % np.pi - np.pi/2  # Wrap [-np.pi/2, np.pi/2]
                angle_quat = tfh.list_to_quaternion(tft.quaternion_from_euler(np.pi, 0, ((angle%np.pi) - np.pi/2)))
                print('Angular quaternion', angle_quat)
                print('Angle(radian): ', angle)
                self.publish_pos(point_3d, angle_quat)
                self.draw_prediction(best_grasp, best_action, best_depth)
                



                # publish to the main planner
                #self.publish_grasp(action)

            rospy.sleep(1)


if __name__ == '__main__':
    rospy.init_node('gqcnn', anonymous=True)
    import dougsm_helpers.tf_helpers as tfh
    GqCNNRt = GqCNNRt()
    GqCNNRt.run()
    #rospy.spin()
