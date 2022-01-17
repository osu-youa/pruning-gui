#!/usr/bin/env python

import socket
import numpy as np
import time
import rospy
from arm_utils.srv import HandlePosePlan, HandleJointPlan
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion


def process(received_array):

    success = 0
    if len(received_array) == 6:
        print('Received joint command')
        js = JointState()
        js.position = received_array
        rez = plan_joints_srv(js, True)
        success = int(rez.success)

    elif len(received_array) == 3:
        print('Received point command')
        print('Currently not implemented')

    elif len(received_array) == 7:
        print('Received pose command')
        pose = PoseStamped()
        pose.header.frame_id = rospy.get_param('base_frame')
        pose.pose.position = Point(*received_array[:3])
        pose.pose.orientation = Quaternion(*received_array[3:])
        rez = plan_pose_srv(pose, True)
        success = int(rez.success)

    else:
        print('Received unknown array of len {}, not taking any action'.format(len(received_array)))

    return success


if __name__ == '__main__':

    rospy.init_node('move_server')


    plan_joints_srv = rospy.ServiceProxy('plan_joints', HandleJointPlan)
    plan_pose_srv = rospy.ServiceProxy('plan_pose', HandlePosePlan)

    ADDRESS = 'localhost'
    PORT = 10000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = (ADDRESS, PORT)
    try:
        print('Starting up server on {}:{}'.format(*address))
        sock.bind(address)
        sock.listen(1)

        while True:
            print('Waiting for connection')
            connection, client_address = sock.accept()
            print('Connection accepted!')

            try:
                master_buffer = connection.recv(1024)
                received_array = np.frombuffer(master_buffer, dtype=np.float64)
                print(received_array)

                # Create an array and send it to the client
                success = process(received_array)
                connection.sendall(bytes(str(success), 'utf-8'))

            finally:
                connection.close()
                print('Connection terminated, waiting for new connection...')
    finally:
        sock.close()