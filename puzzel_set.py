#!/usr/bin/env python
from ambf_msgs.msg import ObjectState
from ambf_msgs.msg import ObjectCmd
import rospy
from random import randint
from time import sleep
import numpy as np
import sys


class Puzzel_reset(object):
    def __init__(self):
        self.init = rospy.init_node('puzzel_init', anonymous=True)
        self.pub = rospy.Publisher('/ambf/icl/PuzzleRed4/Command', ObjectCmd, queue_size=10)
        self.puzzel_pose = ObjectCmd()
        self.puzzel_pose.enable_position_controller = True
        self.map = [[-0.321, -0.58, -0.894], [0.329, 0.038, -0.894], [-0.266, 1.06, -0.893]]
        self.x, self.y, self.z, self.rot = None, None, None, None
        while self.x == None:
            rospy.Subscriber('/ambf/icl/PuzzleRed4/State', ObjectState, self.callback, queue_size=1, buff_size=2 ** 24)
            self.puzzel_pose.pose.position.x = self.x
            self.puzzel_pose.pose.position.y = self.y
            self.puzzel_pose.pose.position.z = self.z
            self.puzzel_pose.pose.orientation = self.rot
        print('initialize done')

    def callback(self, data):
        self.x = data.pose.position.x
        self.y = data.pose.position.y
        self.z = data.pose.position.z
        self.rot = data.pose.orientation

    def reset(self,loc):
        cmd = randint(0, 2)
        pose = self.map[loc]
        vector=np.array([(pose[0]-self.puzzel_pose.pose.position.x),(pose[1]-self.puzzel_pose.pose.position.y)])
        unit = np.linalg.norm(vector, axis=0, keepdims=True)
        unit_vector=vector/unit
        dev_x=unit_vector[0]/1000
        dev_y=unit_vector[1]/1000
        while not rospy.is_shutdown():
            if self.puzzel_pose.pose.position.z < -0.2:
                self.puzzel_pose.pose.position.z += 0.025
                self.pub.publish(self.puzzel_pose)
                sleep(0.1)
            elif np.sqrt(np.power((pose[0]-self.puzzel_pose.pose.position.x),2)+np.power((pose[0]-self.puzzel_pose.pose.position.x),2))>0.001:
                self.puzzel_pose.pose.position.x += dev_x
                self.puzzel_pose.pose.position.y += dev_y
                self.pub.publish(self.puzzel_pose)
                #if abs(self.puzzel_pose.pose.position.x - pose[0]) > 0.05:
                    #print('x not reach')
                #if abs(self.puzzel_pose.pose.position.y - pose[1]) > 0.05:
                    #print('y not reach')
                sleep(0.005)

            else:
		sleep(0.8)
                break

        # puzzel_pose.pose.orientation.x = 0.0
        # puzzel_pose.pose.orientation.y = 0.0
        # puzzel_pose.pose.orientation.z = 0.0
        # puzzel_pose.pose.orientation.w = 0.0


pc = Puzzel_reset()
loc = int(sys.argv[1])
pc.reset(loc)

