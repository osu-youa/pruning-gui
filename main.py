from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QComboBox
from PyQt5.QtCore import QRect, QTimer, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPixmap, QImage
import numpy as np
import os
from functools import partial
from PIL import Image
from stable_baselines3.ppo import PPO
from image_processor import ImageProcessor
import socket
import time

class PruningGUI(QMainWindow):
    def __init__(self, config):
        QMainWindow.__init__(self)

        self.config = config
        self.test = self.config['test']

        # State variables for GUI
        self.waypoint_list = []
        self.current_waypoint = None
        self.detections = []
        self.current_box = None

        # Loaded images
        self.rgb_main = None
        self.rgb_alt = None
        self.depth_img = None
        self.flow_img = None
        self.mask_img = None
        self.mask_detections = None

        # Loaded utilites
        self.image_processor = None
        self.identifier = None
        self.rl_system = None

        # Boilerplate setup
        self.setWindowTitle('Pruning GUI')
        left_side_layout = self.init_left_layout()
        middle_layout = self.init_middle_layout()
        right_side_layout = self.init_right_layout()
        widget = QWidget()
        self.setCentralWidget(widget)
        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)
        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(middle_layout)
        top_level_layout.addLayout(right_side_layout)

        self.image_window = ImageDisplay([['Main RGB', 'Alt RGB'], ['Depth', 'Flow'], ['Mask', 'Boxes']],
                                         names=['RGB', 'Depth/Flow', 'Processed'])
        self.image_window.setWindowTitle('Images')
        self.image_window.show()

    def reset_waypoints(self, waypoint_list):

        self.reset_detections()

        self.current_waypoint = None
        self.waypoint_list = waypoint_list
        self.waypoint_menu.clear()
        for i, waypoint in enumerate(waypoint_list):
            txt = '{}: {}'.format(i+1, ','.join(['{:.2f}'.format(pt) for pt in waypoint]))
            self.waypoint_menu.addItem(txt, i)

        self.waypoint_menu.setCurrentIndex(0)
        self.status.setText('Waypoints loaded, please hit Move')

    def reset_detections(self, detections=None):

        self.detection_combobox.clear()
        if detections is None or len(detections) == 0:
            self.detections = []
            self.detection_status.setText('No detections')
            self.detection_combobox.setDisabled(True)
            self.detection_actions.disable_all()
            return

        self.detections = detections

        for i, detection in enumerate(detections):
            txt = '{}: {}'.format(i+1, ','.join(['{:.2f}'.format(pt) for pt in detection]))
            self.detection_combobox.addItem(txt)

        self.detection_combobox.setEnabled(True)
        self.detection_combobox.setCurrentIndex(0)
        self.detection_status.setText('Detected {} boxes'.format(len(detections)))
        self.detection_actions.reset()

    def load_networks(self):
        if self.image_processor is not None:
            return

        self.image_processor = ImageProcessor((424, 240), (128, 128), use_flow=True, gan_name=self.config['gan_name'])
        self.rl_system = PPO.load(self.config['rl_model_path'])

    def load(self):
        if self.test:
            waypoints = np.random.uniform(-1, 1, (10,3))
            self.reset_waypoints(waypoints)
        else:
            raise NotImplementedError()

    def move_robot_next(self):
        if not len(self.waypoint_list) or self.current_waypoint is None:
            return

        current_idx = self.current_waypoint
        max_idx = self.waypoint_menu.count() - 1
        if current_idx >= max_idx:
            return
        self.waypoint_menu.setCurrentIndex(current_idx + 1)
        self.move_robot_waypoint()


    def move_robot(self, pt):

        # If pt is a 6-vector, will be interpreted as a joint state



        if not self.test or (self.test and self.config['use_dummy_socket']):
            msg = pt.tobytes()
            ADDRESS = self.config.get('socket_ip', 'localhost')
            PORT = self.config['move_server_port']
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (ADDRESS, PORT)
            sock.connect(address)
            sock.sendall(msg)
            response = np.frombuffer(sock.recv(1024), dtype=np.uint8)
            sock.close()
            print('Move server response: {}'.format(response))
        print('Moved robot to: {}'.format(', '.join(['{:.3f}'.format(x) for x in pt])))

    def move_robot_waypoint(self):

        if not len(self.waypoint_list):
            return

        waypoint_index = self.waypoint_menu.currentData()

        self.move_robot(self.waypoint_list[waypoint_index])

        success = np.random.uniform() > 0.1
        if success:
            self.status.setText('At Waypoint {}'.format(waypoint_index + 1))
            self.actions_list.reset()
        else:
            self.status.setText('Failed to move to Waypoint {}'.format(waypoint_index + 1))
            waypoint_index = None
            self.actions_list.disable_all()

        self.current_waypoint = waypoint_index

        if waypoint_index is None:
            self.next_button.setDisabled(True)
        elif waypoint_index == len(self.waypoint_list) - 1:
            self.next_button.setDisabled(True)
        else:
            self.next_button.setEnabled(True)

    def acquire_images(self):

        if self.test:
            path = self.config['dummy_image_path']
            file_format = self.config['dummy_image_format']
            start_frame = np.random.randint(0, 100)

            self.rgb_alt = np.array(Image.open(os.path.join(path, file_format.format(frame=start_frame)))).astype(np.uint8)[:,:,:3]
            self.rgb_main = np.array(Image.open(os.path.join(path, file_format.format(frame=start_frame+1)))).astype(np.uint8)[:,:,:3]

        else:
            raise NotImplementedError()

        self.image_window.update_image('Main RGB', self.rgb_alt, 256)
        self.image_window.update_image('Alt RGB', self.rgb_main, 256)

    def compute_flow(self):

        self.load_networks()
        self.image_processor.reset()
        self.image_processor.process(self.rgb_alt)

        self.mask_img = self.image_processor.process(self.rgb_main)
        self.flow_img = self.image_processor.last_flow

        self.image_window.update_image('Flow', self.flow_img, 256)
        self.image_window.update_image('Mask', self.mask_img, 256)

    def detect_prune_points(self):
        num_boxes = np.random.randint(0,5)
        detections = np.random.uniform(-1, 1, (num_boxes, 3))
        self.reset_detections(detections)

    def move_to_box(self):

        idx = self.detection_combobox.currentIndex()
        if self.test:
            print('Moving to box {}'.format(idx+1))
        else:
            raise NotImplementedError()

    def execute_approach(self):

        sock = None
        if not self.test or (self.test and self.config['use_dummy_socket']):
            ADDRESS = self.config.get('socket_ip', 'localhost')
            PORT = self.config['vel_server_port']
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (ADDRESS, PORT)
            sock.connect(address)

        self.image_processor.reset()
        try:
            if self.test:
                path = self.config['dummy_image_path']
                file_format = os.path.join(path, self.config['dummy_image_format'])
                start_frame = np.random.randint(0, 100)

                for i in range(np.random.randint(10, 30)):
                    file_path = file_format.format(frame=start_frame + i)
                    img = np.array(Image.open(file_path), dtype=np.uint8)
                    seg = self.image_processor.process(img)

                    self.image_window.update_image('Main RGB', img)
                    self.image_window.update_image('Flow', self.image_processor.last_flow)
                    self.image_window.update_image('Mask', seg)

                    action = self.rl_system.predict(seg)[0]
                    if sock is not None:
                        sock.sendall(action.tobytes())
                        np.frombuffer(sock.recv(1024), dtype=np.uint8) # Synchronization
                    else:
                        print(action)

            else:
                while time.time() - start < duration:

                    raise NotImplementedError()
                    # Get image from camera
                    img = None
                    seg = self.image_processor.process(img)

                    self.image_window.update_image('Main RGB', img)
                    self.image_window.update_image('Flow', self.image_processor.last_flow)
                    self.image_window.update_image('Mask', seg)

                    action = self.rl_system.predict(seg)[0]
                    sock.sendall(action.tobytes())
                    response = np.frombuffer(sock.recv(1024), dtype=np.uint8)  # Synchronization
                    if response == 0:
                        break

        except Exception as e:
            print('Ran into an Exception!')
            print(e)
        finally:
            if sock is not None:
                sock.close()
        print('Approach finished!')


    def cut(self):
        if self.test:
            print('Cut!')
        else:
            raise NotImplementedError()





    def init_left_layout(self):
        layout = QVBoxLayout()
        save_button = QPushButton('Save')
        load_button = QPushButton('Load')

        all_widgets = [save_button, load_button]
        for widget in all_widgets:
            layout.addWidget(widget)

        load_button.clicked.connect(self.load)

        return layout

    def init_middle_layout(self):

        layout = QVBoxLayout()

        waypoint_widget = QGroupBox()
        waypoint_layout = QVBoxLayout()
        waypoint_widget.setLayout(waypoint_layout)




        combobox_layout = QHBoxLayout()
        self.waypoint_menu = QComboBox()
        move_button = QPushButton('Move')
        self.next_button = QPushButton('>>')

        self.status = QLabel('Current Status: None')

        waypoint_layout.addWidget(self.status)
        combobox_layout.addWidget(self.waypoint_menu)
        combobox_layout.addWidget(move_button)
        combobox_layout.addWidget(self.next_button)
        waypoint_layout.addLayout(combobox_layout)

        self.actions_list = SequentialButtonList(['Acquire Images', 'Compute Flow', 'Detect Prune Points'],
                                                 [self.acquire_images, self.compute_flow, self.detect_prune_points],
                                                 name='Actions')
        self.actions_list.disable_all()
        waypoint_layout.addWidget(self.actions_list)

        layout.addWidget(waypoint_widget)

        move_button.clicked.connect(self.move_robot_waypoint)
        self.next_button.clicked.connect(self.move_robot_next)

        return layout

    def init_right_layout(self):
        layout = QVBoxLayout()
        self.detection_widget = QVGroupBox('Detected Boxes')
        self.detection_status = QLabel('No detected boxes')
        self.detection_widget.addWidget(self.detection_status)
        layout.addWidget(self.detection_widget)

        self.detection_combobox = QComboBox()
        self.detection_actions = SequentialButtonList(['Move to Box', 'Execute Approach', 'Cut'],
                                                 [self.move_to_box, self.execute_approach, self.cut],
                                                 name='Actions')

        self.detection_combobox.currentIndexChanged.connect(lambda: self.detection_actions.reset())

        self.detection_widget.addWidget(self.detection_combobox)
        self.detection_widget.addWidget(self.detection_actions)
        self.reset_detections()

        return layout
#
# class ActionItemBar(QWidget):
#
#     def __init__(self, name, callback):
#         super().__init__()
#         layout = QHBoxLayout()
#         self.setLayout(layout)
#         layout.addWidget(QLabel(name))
#

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        for i in range(5):
            self.progress.emit(i+1)
        self.finished.emit()



class ImageDisplay(QWidget):

    @staticmethod
    def np_to_qimage(array):
        array = array[:,:,:3].copy()
        h, w, c = array.shape

        bytes_per_line = 3 * w
        qimg = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg

    def __init__(self, layout, names=None):

        # Layout is a list of lists with IDs for each image, which are also used as keys for updating
        # e.g. [['A', 'B', 'C'], ['D', 'E']] creates a two column layout with ABC in the first and DE in the second

        super(ImageDisplay, self).__init__()

        img_layout = QHBoxLayout()
        self.setLayout(img_layout)

        self.labels = {}

        if names is None:
            names = [''] * len(layout)

        for col_fields, group_name in zip(layout, names):
            col = QVGroupBox(group_name)
            for field in col_fields:
                self.labels[field] = QLabel()
                col.addWidget(QLabel(field))
                col.addWidget(self.labels[field])
            img_layout.addWidget(col)

    def update_image(self, label, img_array, width=None):
        if width is None:
            width = img_array.shape[1]

        qimg = self.np_to_qimage(img_array)
        pixmap = QPixmap(qimg).scaledToWidth(width)
        self.labels[label].setPixmap(pixmap)

    def clear(self):
        for label in self.labels.keys():
            label.setPixmap(QPixmap())

class QVGroupBox(QWidget):
    def __init__(self, name, vertical=True):
        super(QVGroupBox, self).__init__()

        container_layout = QVBoxLayout()
        self.setLayout(container_layout)

        self.main_widget = QGroupBox(name)
        self.main_layout = QVBoxLayout() if vertical else QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        container_layout.addWidget(self.main_widget)

    def addWidget(self, widget):
        self.main_layout.addWidget(widget)

    def clear_widgets(self):
        for i in reversed(range(self.main_layout.count())):
            widget = self.main_layout.itemAt(i).widget()
            self.main_layout.removeWidget(widget)
            widget.setParent(None)


class SequentialButtonList(QVGroupBox):

    def __init__(self, buttons, callbacks, name='', vertical=True):
        super().__init__(name, vertical=vertical)
        self.buttons = []
        self.callbacks = callbacks
        self.next_to_run = 0
        for i, (button_name, callback) in enumerate(zip(buttons, callbacks)):
            button = QPushButton(button_name)
            button.clicked.connect(partial(self.run_callbacks, i))
            self.buttons.append(button)
            self.addWidget(button)

    def run_callbacks(self, i):
        for action in range(i+1):
            if action < self.next_to_run:
                continue
            self.callbacks[action]()
            self.buttons[action].setDisabled(True)
        self.next_to_run = i+1

    def reset(self):
        self.next_to_run = 0
        for button in self.buttons:
            button.setEnabled(True)

    def disable_all(self):
        for button in self.buttons:
            button.setDisabled(True)


if __name__ == '__main__':

    config = {
        'test': True,
        # 'dummy_image_path': r'C:\Users\davijose\Pictures\TrainingData\GanTrainingPairsWithCutters\train',
        # 'dummy_image_format': 'render_{}_randomized_{:05d}.png',
        'dummy_image_path': r'C:\Users\davijose\Pictures\TrainingData\RealData\MartinDataCollection\20220109-141856-Downsized',
        'dummy_image_format': '{frame:04d}-C.png',
        
        
        'gan_name': 'orchard_cutterflowseg_pix2pix',
        'use_dummy_socket': False,
        'move_server_port': 10000,
        'vel_server_port': 10001,
        'rl_model_path': r'C:\Users\davijose\PycharmProjects\pybullet-test\best_model_1_0.zip',

    }

    procs = []
    if config['test'] and config['use_dummy_socket']:
        import subprocess, shlex
        procs.append(subprocess.Popen(shlex.split('python move_server_dummy.py'), stdout=subprocess.PIPE))
        procs.append(subprocess.Popen(shlex.split('python vel_command_server_dummy.py'), stdout=subprocess.PIPE))

    try:
        app = QApplication([])
        gui = PruningGUI(config)
        gui.show()
        app.exec_()
    finally:
        for proc in procs:
            proc.terminate()