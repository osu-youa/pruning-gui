#!/usr/bin/env python

import socket
import numpy as np
import time
import rospy
from arm_utils.srv import HandlePosePlan, HandleJointPlan
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, PointStamped, Quaternion, Pose, TransformStamped
from tf2_ros import TransformListener as TransformListener2, Buffer
from tf2_geometry_msgs import do_transform_point
from copy import deepcopy
from std_msgs.msg import Bool

BASE_TF = None
BASE_JOINTS = None


def process(received_array):
    code = int(received_array[0])
    msg = ''
    if code == 0:
        abort_pub.publish(Bool(True))
        msg = 'Abort message sent'

    elif code == 1:
        msg = 'Cutting service not implemented yet!'

    elif code == -1:
        msg = 'Connection has successfully been established!'

    print(msg)
    return msg



if __name__ == '__main__':

    rospy.init_node('move_server')

    tf_buffer = Buffer()
    tf_listener = TransformListener2(tf_buffer)
    rospy.sleep(1.0)

    plan_joints_srv = rospy.ServiceProxy('plan_joints', HandleJointPlan)
    plan_pose_srv = rospy.ServiceProxy('plan_pose', HandlePosePlan)

    abort_pub = rospy.Publisher('abort', Bool, queue_size=1)

    ADDRESS = '169.254.116.60'
    PORT = 10002

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = (ADDRESS, PORT)
    try:
        print('Starting up server on {}:{}'.format(*address))
        sock.bind(address)
        sock.listen(1)

        while True:
            print('Waiting for connection')
            try:
                connection, client_address = sock.accept()
            except socket.timeout:
                continue
            print('Connection accepted!')

            try:
                master_buffer = connection.recv(1024)
                received_array = np.frombuffer(master_buffer, dtype=np.float64)
                print(received_array)

                # Create an array and send it to the client
                success = process(received_array)
                connection.sendall(bytes(str(success)))

            finally:
                connection.close()
                print('Connection terminated, waiting for new connection...')
    finally:
        sock.close()
