#!/usr/bin/env python

import socket
import rospy
import numpy as np
from geometry_msgs.msg import Vector3Stamped, Vector3
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, Empty

CONTACT = False
ABORT = False

def handle_contact(msg):
    contact = msg.data
    global CONTACT
    CONTACT = contact

def handle_abort(msg):
    abort = msg.data
    global ABORT
    ABORT = abort


if __name__ == '__main__':

    rospy.init_node('vel_command_server')

    run_admittance_ctrl = rospy.ServiceProxy('run_admittance_controller', Trigger)
    pub = rospy.Publisher('/vel_command', Vector3Stamped, queue_size=1)
    rospy.Subscriber('/contact', Bool, handle_contact, queue_size=1)
    rospy.Subscriber('/abort', Bool, handle_abort, queue_size=1)
    servo_start = rospy.ServiceProxy('servo_activate', Empty)
    servo_stop = rospy.ServiceProxy('servo_stop', Empty)

    ADDRESS = 'localhost'
    PORT = 10000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    def shutdown():
        sock.close()
    rospy.on_shutdown(shutdown)

    address = (ADDRESS, PORT)
    print('Starting up server on {}:{}'.format(*address))
    sock.bind(address)
    sock.listen(1)

    while not rospy.is_shutdown():
        print('Waiting for connection')
        try:
            connection, client_address = sock.accept()
        except socket.timeout:
            continue
        print('Connection accepted!')

        servo_start()
        try:
            # Phase 1 - Connect to the client and output velocity commands
            try:
                while not ABORT and not CONTACT:
                    master_buffer = connection.recv(1024)
                    received_array = np.frombuffer(master_buffer, dtype=np.float64)
                    print(received_array)

                    msg = Vector3Stamped()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = 'cutpoint'
                    msg.vector = Vector3(*received_array)
                    pub.publish(msg)

                    success = 1
                    connection.sendall(bytes(str(success)))

            finally:
                connection.close()
                print('Connection terminated, waiting for new connection...')

            # Phase 2 - Based on the stop condition, take a different action
            if CONTACT:
                run_admittance_ctrl()

        finally:
            servo_stop()
            rospy.sleep(0.5)
            ABORT = False
            CONTACT = False
