#!/usr/bin/env python
'''
plan:
- define the robotiq commands OK
- read grasp message OK

- convert to rbot frame  
> camera pose
> transforms in grasp2d.pose() to doublecheck
> offsets

- send command to robot
#
'''

# execute
from __future__ import division, print_function

import rospy      


import time
import os

import numpy as np
import cv2

#from std_msgs.msg import Stringgrasp

from tf import transformations as tft
#from dougsm_helpers.timeit import TimeIt

#from autolab_core import Point, Logger

#from gqcnn.grasping import Grasp2D, SuctionPoint2D, GraspAction
#from gqcnn.msg import GQCNNGrasp
#from gqcnn.srv import GQCNNGraspPlanner, GQCNNGraspPlannerSegmask
#from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage

#from sensor_msgs.msg import Image, CameraInfo
#from visualization import Visualizer2D as vis
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from gqcnn.msg import Grasp
from franka_msgs.msg import FrankaState, Errors as FrankaErrors


from franka_control_wrappers.panda_commander import PandaCommander
#from mvp_grasping.utils import correct_grasp

import dougsm_helpers.tf_helpers as tfh
from dougsm_helpers.ros_control import ControlSwitcher



class PandaOpenLoopGraspController(object):
    """
    Perform open-loop grasps from a single viewpoint using the Panda robot.
    """

    def __init__(self):
        self.gripper = rospy.get_param("~gripper", "panda")
        if self.gripper == "panda":
            self.LINK_EE_OFFSET = 0.1384
        elif self.gripper == "robotiq":
            self.LINK_EE_OFFSET = 0.32

        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher(
            "/cartesian_velocity_node_controller/cartesian_velocity",
            Twist,
            queue_size=1,
        )
        self.max_velo = 0.10
        self.curr_velo = Twist()
        self.best_grasp = None#GQCNNGrasp()

        self.cs = ControlSwitcher(
            {
                "moveit": "position_joint_trajectory_controller",
                "velocity": "cartesian_velocity_node_controller",
            }
        )
        self.cs.switch_controller("moveit")
        self.pc = PandaCommander(group_name="panda_arm", gripper=self.gripper)

        self.robot_state = None
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False
        rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            self.__robot_state_callback,
            queue_size=1,
        )

        self.grasper = rospy.Publisher('/gqcnn/grasp_input', String, queue_size=1)

    def calculate_grasp(self):
    	self.grasper.publish( 'calculate_grasp')

    def __recover_robot_from_error(self):
        rospy.logerr("Recovering")
        self.pc.recover()
        self.cs.switch_controller("moveit")
        self.pc.goto_saved_pose("start", velocity=0.1)
        rospy.logerr("Done")
        self.ROBOT_ERROR_DETECTED = False

    def __robot_state_callback(self, msg):
        self.robot_state = msg
        if any(self.robot_state.cartesian_collision):
            if not self.ROBOT_ERROR_DETECTED:
                rospy.logerr("Detected Cartesian Collision")
            self.ROBOT_ERROR_DETECTED = True
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                if not self.ROBOT_ERROR_DETECTED:
                    rospy.logerr("Robot Error Detected")
                self.ROBOT_ERROR_DETECTED = True

    def __execute_best_grasp(self):
        self.cs.switch_controller("moveit")

        print('Prepare planning')
        self.calculate_grasp()
        self.best_grasp = rospy.wait_for_message("/gqcnn/grasp_output", Grasp)
        print('Received msg')
        ############# TODO  #############################
        #self.best_grasp = correct_grasp(self.best_grasp, self.gripper)

        tfh.publish_pose_as_transform(self.best_grasp.pose, "panda_link0", "G", 0.5)

        # Offset for initial pose.(KIV)
        #offset = 0.05 + self.LINK_EE_OFFSET
        #gripper_width_offset = 0.03

        #self.best_grasp.pose.position.z += offset

        # Set approach gripper width (KIV)
        #self.pc.gripper.set_gripper(self.best_grasp.width + gripper_width_offset, wait=False)
        #self.pc.gripper.set_gripper(self.best_grasp.width , wait=False)

        rospy.sleep(0.1)
        print(self.best_grasp)
        self.pc.goto_pose(self.best_grasp.pose, velocity=0.1)

        # Reset the position
        offset = 0.05
        self.best_grasp.pose.position.z -= offset


        self.cs.switch_controller("velocity")
        v = Twist()
        v.linear.z = -0.05

        # Monitor robot state and descend
        while (
            self.robot_state.O_T_EE[-2] > self.best_grasp.pose.position.z
            and not any(self.robot_state.cartesian_contact)
            and not self.ROBOT_ERROR_DETECTED
        ):
            self.curr_velo_pub.publish(v)
            rospy.sleep(0.01)

        # Check for collisions
        if self.ROBOT_ERROR_DETECTED:
            return False

        rospy.sleep(1)
        self.cs.switch_controller("moveit")
        # close the fingers.
        rospy.sleep(0.2)
        self.pc.gripper.grasp(0, force=1)

        # Sometimes triggered by closing on something that pushes the robot
        if self.ROBOT_ERROR_DETECTED:
            return False

        return True

    def stop(self):
        self.pc.stop()
        self.curr_velo = Twist()
        self.curr_velo_pub.publish(self.curr_velo)

    def go(self):
        self.cs.switch_controller("moveit")
        #self.pc.goto_saved_pose("start", velocity=0.1)
        self.pc.gripper.set_gripper(0.1)

        grasp_ret = self.__execute_best_grasp()
        if not grasp_ret or self.ROBOT_ERROR_DETECTED:
            rospy.logerr("Something went wrong, aborting this run")
            if self.ROBOT_ERROR_DETECTED:
                self.__recover_robot_from_error()
        self.cs.switch_controller("moveit")

        #self.pc.goto_saved_pose("bin", velocity=0.1)

        self.cs.switch_controller("velocity")
        v = Twist()
        v.linear.z = -0.05

        # Monitor robot state and descend
        while (
            self.robot_state.O_T_EE[-2] > self.best_grasp.pose.position.z
            and not any(self.robot_state.cartesian_contact)
            and not self.ROBOT_ERROR_DETECTED
        ):
            self.curr_velo_pub.publish(v)
            rospy.sleep(0.01)
        rospy.sleep(1)
        self.cs.switch_controller("moveit")
        self.pc.gripper.set_gripper(0.1)
        self.pc.goto_saved_pose("start", velocity=0.1)


if __name__ == "__main__":
    rospy.init_node("panda_open_loop_grasp")
    pg = PandaOpenLoopGraspController()
    pg.go()
