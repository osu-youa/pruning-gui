import rospy
from std_srvs.srv import Empty
from pyfirmata import Arduino, util
from time import sleep, time

LAST_INIT_TIME = 0
MAX_CUT_ATTEMPTS = 3

def initialize(dig):

    global LAST_INIT_TIME
    cur_time = time()
    if cur_time - LAST_INIT_TIME >= 180:
        dig.write(0)
        sleep(.05)
        dig.write(1)
        sleep(.05)
        dig.write(0)
        sleep(.05)
        dig.write(1)
        sleep(.5)

        LAST_INIT_TIME = cur_time

def do_cut(*_, **__):

    initialize(dig0)
    dig0.write(0)  # Commands the Pruner to cut
    last_cut_time = time()
    sleep(.5)
    dig0.write(1)  # Sets the state of the Pruner to open

    cut_attempts = 0
    sensor_open = board.digital[5].read()  # Reads the Hall Effect sensor position to see if it is open or closed
    # Loop checks if the Hall Effect Sensor finished the cut (It reads False if the cut is complete)
    while sensor_open:
        sensor_open = board.digital[5].read()
        # If it did not complete the cut it will continue to try
        if time() - last_cut_time >= 1.5:
            cut_attempts += 1
            sleep(.5)
            dig0.write(0)
            sleep(.5)
            dig0.write(1)
            last_cut_time = time()
            if cut_attempts >= MAX_CUT_ATTEMPTS:  # Set to number of times to try and cut through branch again before moving on
                break
        rospy.sleep(0.1)

    sleep(.5)
    dig0.write(1)

    return []

def shutdown():
    reset.write(0)


if __name__ == '__main__':

    SERIAL_PORT = 'COM5'  # Change this to the COM port the Arduino is connected to

    # This sets up the Arduino board and defines the pins that we will be using to send and read signals
    board = Arduino(SERIAL_PORT)
    DIM = 300
    it = util.Iterator(board)
    it.setDaemon(True)
    it.start()
    dig0 = board.get_pin('d:3:o')
    lim0 = board.get_pin('d:5:i')
    reset = board.get_pin('d:12:o')

    LAST_INIT_TIME = time()  # If the Pruner's trigger is not "pulled" for more than 180 seconds it times out and needs to be
    # re-initialized, this tracks the time of the last time it was "pulled"

    rospy.init_node('cutter_node')
    rospy.on_shutdown(shutdown)
    rospy.Service('activate_cutters', Empty, do_cut)

    # When the Pruner is turned on it needs to be initialized by "pulling" the trigger twice quickly which this simulates
    initialize(dig0)

    rospy.spin()