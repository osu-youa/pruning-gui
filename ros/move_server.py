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

BASE_TF = None
BASE_JOINTS = None
PC_FRAME = '/camera_color_optical_frame'
PC_TF = None

def process(received_array):
    tool_frame = rospy.get_param('tool_frame')
    base_frame = rospy.get_param('base_frame')

    success = 0

    # All commanded arrays should be of the form [code, v1, v2, ...]
    # The code contains info on how to interpret the received array
    code = int(received_array[0])
    received_array = received_array[1:]

    if code == 0:  # RETRACT TO BASE POSE
        if BASE_JOINTS is not None:
            success = plan_joints_srv(BASE_JOINTS, True)
        else:
            print('Cannot retract to base joints when none has been defined!')

    elif code == 1:     # PLAN TO JOINTS, RESET BASE
        print('Received joint command')

        js = JointState()
        js.position = received_array
        rez = plan_joints_srv(js, True)
        success = int(rez.success)

        tf = retrieve_tf(tool_frame, base_frame)

        global BASE_TF
        global BASE_JOINTS
        global PC_TF
        BASE_TF = tf
        BASE_JOINTS = rospy.wait_for_message('/joint_states', JointState, timeout=0.5)
        PC_TF = retrieve_tf(PC_FRAME, base_frame)


    elif code == 2:
        print('Received pose command')
        pose = PoseStamped()
        pose.header.frame_id = rospy.get_param('base_frame')
        pose.pose.position = Point(*received_array[:3])
        pose.pose.orientation = Quaternion(*received_array[3:])
        rez = plan_pose_srv(pose, True)
        success = int(rez.success)

    elif code == 3:     # PLAN TO TOOL-FRAME POSE

        print('Received tool-frame point command')

        tf = BASE_TF
        pose = tf_to_pose(tf, keep_header=True)

        pt_world = apply_tf_to_point_array(tf, received_array, tool_frame)
        pose.pose.position = pt_world.point

        rez = plan_pose_srv(pose, True)
        success = int(rez.success)


    elif code == 4:

        print('Received camera-frame tool-frame point command with z-offset')

        pt_array = received_array[:3]
        z_offset = received_array[3]

        tf = PC_TF
        pose = tf_to_pose(tf, keep_header=True)

        # First modify the tf to move it by the z-offset
        pt_world = apply_tf_to_point_array(tf, np.array([0, 0, z_offset]), tool_frame).point
        tl = tf.transform.translation
        tl.x, tl.y, tl.z = pt_world.x, pt_world.y, pt_world.z

        # Using the transformed TF, process the received point
        pt_approach = apply_tf_to_point_array(tf, pt_array, tool_frame).point
        pose.pose.position = pt_approach

        rez = plan_pose_srv(pose, True)
        success = int(rez.success)


    else:
        print('Received unknown code {}, not taking any action'.format(code))

    return success


def apply_tf_to_point_array(tf, array, array_frame):
    pt = PointStamped()
    pt.point = Point(*array)
    pt.header.frame_id = array_frame
    tfed_pt = do_transform_point(pt, tf)
    return tfed_pt


def retrieve_tf(base_frame, target_frame, stamp=rospy.Time()):
    # Retrieves a TransformStamped with a child frame of base_frame and a target frame of target_frame
    # This transform can be applied to a point in the base frame to transform it to the target frame
    success = tf_buffer.can_transform(target_frame, base_frame, stamp, rospy.Duration(0.5))
    if not success:
        rospy.logerr("Couldn't look up transform between {} and {}!".format(target_frame, base_frame))
    tf = tf_buffer.lookup_transform(target_frame, base_frame, stamp)
    return tf


def tf_to_pose(tf, keep_header=False):
    header = None
    if isinstance(tf, TransformStamped):
        header = tf.header
        tf = tf.transform
    elif keep_header:
        raise ValueError("Cannot preserve the header of a non-stamped Transform!")
    t = tf.translation
    r = tf.rotation
    pose = Pose()
    pose.position = Point(t.x, t.y, t.z)
    pose.orientation = r
    if not keep_header:
        return pose
    assert header is not None
    pose_stamped = PoseStamped()
    pose_stamped.pose = pose
    pose_stamped.header = header
    return pose_stamped


if __name__ == '__main__':

    rospy.init_node('move_server')

    tf_buffer = Buffer()
    tf_listener = TransformListener2(tf_buffer)
    rospy.sleep(1.0)

    plan_joints_srv = rospy.ServiceProxy('plan_joints', HandleJointPlan)
    plan_pose_srv = rospy.ServiceProxy('plan_pose', HandlePosePlan)

    ADDRESS = '169.254.174.52'
    PORT = 10000

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
