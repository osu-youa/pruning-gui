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

def process(received_array):

    success = 0
    if len(received_array) == 6:
        print('Received joint command')
        js = JointState()
        js.position = received_array
        rez = plan_joints_srv(js, True)
        success = int(rez.success)

    elif len(received_array) == 3:

        print('Received tool-frame point command')

        tool_frame = rospy.get_param('tool_frame')
        base_frame = rospy.get_param('base_frame')

        import pdb
        pdb.set_trace()

        tf = retrieve_tf(tool_frame, base_frame)
        pose = tf_to_pose(tf, keep_header=True)

        pt = PointStamped()
        pt.point = Point(*received_array)
        pt.header.frame_id = tool_frame
        pt_world = do_transform_point(pt, tf)
        pose.pose.position = pt_world.point

        rez = plan_pose_srv(pose, True)
        success = int(rez.success)

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


def retrieve_tf(base_frame, target_frame, stamp = rospy.Time()):

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
                connection.sendall(bytes(str(success)))

            finally:
                connection.close()
                print('Connection terminated, waiting for new connection...')
    finally:
        sock.close()