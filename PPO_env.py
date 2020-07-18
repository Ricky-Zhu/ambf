import numpy as np
import sys
import random
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ambf_client import Client
from tf.transformations import quaternion_from_euler as qfe
from ambf_msgs.msg import ObjectCmd
from ambf_msgs.msg import ObjectState
from judge import Judge
from time import sleep
from math import *
import os


class GameState:
    def __init__(self):

        self.bridge = CvBridge()
        self.c = Client()
        self.c.connect()
        self.gripper_base = self.c.get_obj_handle('r_gripper_palm_link')
        self.sensor = self.c.get_obj_handle('Proximity0')
        self.actuator = self.c.get_obj_handle('Constraint0')
        self.s_1 = None  # image from camera 1
        self.s_1a = None  # image from camera 1a
        self.rot_block = None
        self.pos_block = None
        self.pos_palm = None
        self.rot_palm = None

        self.img1_sub = rospy.Subscriber("/ambf/image_data/camera1", Image, self.callback_img_1,
                                         queue_size=10, buff_size=2 ** 24)
        self.img1a_sub = rospy.Subscriber("/ambf/image_data/camera1a", Image, self.callback_img_1a,
                                          queue_size=10, buff_size=2 ** 24)
        self.palm_sub = rospy.Subscriber('/ambf/env/r_gripper_palm_link/State', ObjectState, self.callback_palm,
                                         queue_size=10,
                                         buff_size=2 ** 24)
        self.block_sub = rospy.Subscriber('/ambf/icl/PuzzleRed4/State', ObjectState, self.callback_block,
                                          queue_size=10, buff_size=2 ** 24)
        self.action_list = ['w', 's', 'a', 'd', 'h']  # contain 4 actions

        # parameters for control the gripper
        self.r = 0
        self.p = -90 * 3.14 / 180
        self.y = 0
        self.a = -0.0
        self.b = -0.0
        self.c = -0.5

        # fixed reset position
        self.r0 = 0
        self.p0 = -90 * 3.14 / 180
        self.y0 = 0
        self.a0 = -0.0
        self.b0 = -0.0
        self.c0 = -0.5

        # gripper reset posotion when puzzel reset
        self.a1 = -1.2
        self.b1 = -0.0
        self.c1 = -0.5

        # distance for the last step
        self.d0 = None

    def first_reset(self):
        self.gripper_base.set_rot(qfe(self.r0, self.p0, self.y0))
        self.gripper_base.set_pos(self.a0, self.b0, self.c0)
        while self.s_1 is None:
            self.step([0.001,0.001])
            sleep(0.1)
        self.d0 = self.check_dist()

    def reset(self, loc):
        self.gripper_base.set_pos(self.a1, self.b1, self.c1)
        sleep(1)
        os.system('{} {} {}'.format('python', '/home/dandan/ruiqi_ambf/Pyscript/puzzel_set.py', loc))
	self.a0 = random.uniform(-0.2,-0.1)
        self.b0 = random.uniform(-0.1,0.1)
        self.gripper_base.set_pos(self.a0, self.b0, self.c0)
        sleep(2)
        self.a, self.b, self.c = self.a0, self.b0, self.c0
        self.s_1, self.s_1a, _, _ = self.step([0.001,0.001])
        return self.s_1, self.s_1a
    def gri_reset(self):
	self.gripper_base.set_pos(self.a0, self.b0, self.c0)
	sleep(0.5)
	self.a, self.b, self.c = self.a0, self.b0, self.c0
	self.s_1, self.s_1a, _, _ = self.step([0.001,0.001])
	sleep(0.5)
	return self.s_1, self.s_1a

    def check_dist(self):
        d = sqrt((self.pos_block[0] - self.pos_palm[0]) ** 2 + (self.pos_block[1] - self.pos_palm[1]) ** 2)
        return d

    def callback_img_1(self, data):
        try:
            self.s_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # self.s = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret, self.s = cv2.threshold(self.s, 1, 255, cv2.THRESH_BINARY)
            # cv2.imshow("image", self.s)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

    def callback_img_1a(self, data):
        try:
            self.s_1a = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # self.s = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret, self.s = cv2.threshold(self.s, 1, 255, cv2.THRESH_BINARY)
            # cv2.imshow("image", self.s_1a)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

    def callback_block(self, data):
        self.pos_block = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.rot_block = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
                                   data.pose.orientation.w])

    def callback_palm(self, data):
        self.pos_palm = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.rot_palm = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z,
                                  data.pose.orientation.w])

    def move(self, a):

            x1 = a[0]
            y1 = a[1]
            self.a += x1
            self.b += y1
            self.gripper_base.set_pos(self.a, self.b, self.c)
	    sleep(0.2)

    def check_bound(self):
        check = False
        if (self.a < -0.72) or (self.a > 0.56):
            check = True
        if (self.b < -0.72) or (self.b > 1.2):
            check = True
        return check
    def step(self, action):
        reward = 0.1
        terminal = False
        self.move(action)
        
        if not self.pos_block is None:
            assert type(self.d0) != None

            d = self.check_dist()
            assert d != None

            if (d and self.d0) is not None:
                reward = 10 * (tanh(self.d0)-tanh(d))
            if d < 0.1:
                reward = 5.0
                terminal = True
        else:
            d = None
        self.d0 = d
        if self.check_bound():
            reward = -5.0
            terminal = True
        return self.s_1, self.s_1a, reward, terminal






