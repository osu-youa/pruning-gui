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
from rs_camera import Camera
import cv2

class PruningGUI(QMainWindow):
    def __init__(self, config):
        QMainWindow.__init__(self)

        self.config = config
        self.test = self.config['test']
        self.cam = None
        if not self.config.get('test_camera', True):
            self.cam = Camera(640, 480)

        # State variables for GUI
        self.waypoint_list = []
        self.current_waypoint = None
        self.detections = []
        self.current_box = None
        self.last_click = None

        # Loaded images
        self.rgb_main = None
        self.rgb_alt = None
        self.depth_img = None
        self.pc = None
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
        self.image_window.register_click_callback('Mask', self.mask_click_callback)
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

    def reset_detections(self, clear=False):

        self.detection_combobox.clear()
        self.last_click = None

        if clear or len(self.detections) == 0:
            self.detections = []
            self.detection_status.setText('No detections')
            self.detection_combobox.setDisabled(True)
            self.detection_actions.disable_all()
            return

        if self.mask_img is None:
            raise Exception("Detections added but no mask has been generated!")

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        annotated_mask = cv2.resize(self.mask_img, (self.rgb_main.shape[1], self.rgb_main.shape[0]))

        for i, detection in enumerate(self.detections):

            target, pix = detection
            color = colors[i%len(colors)]

            x1, y1 = pix.min(axis=0)
            x2, y2 = pix.max(axis=0)

            annotated_mask = cv2.rectangle(annotated_mask, (x1, y1), (x2, y2), color=color, thickness=3)
            annotated_mask = cv2.putText(annotated_mask, str(i+1), (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                                         color=color, thickness=2)


            txt = '{}: {:.3f}, {:.3f}, {:.3f}'.format(i+1, *target)
            self.detection_combobox.addItem(txt)

        self.image_window.update_image('Boxes', annotated_mask, width=256)

        self.detection_combobox.setEnabled(True)
        self.detection_combobox.setCurrentIndex(0)
        self.detection_status.setText('Detected {} boxes'.format(len(self.detections)))
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
            print('Loading set waypoints for Millrace robot!')
            waypoints = [
                [1.069, -1.401, -1.649, -0.178, 1.152, 3.182],
                [0.804, -1.424, -1.634, -0.162, 1.416, 3.159],
                [0.587, -1.484, -1.582, -0.153, 1.633, 3.142],
                [0.333, -1.609, -1.462, -0.152, 1.886, 3.121],
            ]
            self.reset_waypoints(np.array(waypoints))

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

        # See move_server.py for defined behaviors based on array size
        # 6-vector, will be interpreted as a joint state
        # 3-vector, point relative to the last-defined base pose
        # 4-vector, point relative to base pose, but with z-offset

        if not self.test or (self.test and self.config['use_dummy_socket']):
            msg = pt.tobytes()
            ADDRESS = self.config.get('socket_ip', 'localhost')
            PORT = self.config['move_server_port']
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (ADDRESS, PORT)

            sock.connect(address)
            sock.sendall(msg)
            response = sock.recv(1024)
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

        if self.config.get('test_camera'):
            path = self.config['dummy_image_path']
            file_format = self.config['dummy_image_format']
            start_frame = np.random.randint(0, 100)

            self.rgb_alt = np.array(Image.open(os.path.join(path, file_format.format(frame=start_frame)))).astype(np.uint8)[:,:,:3]
            self.rgb_main = np.array(Image.open(os.path.join(path, file_format.format(frame=start_frame+1)))).astype(np.uint8)[:,:,:3]

        else:

            self.rgb_main, self.depth_img = self.cam.acquire_image()
            self.pc = self.cam.acquire_pc(return_rgb=False)

            self.move_robot(np.array([0.01, 0.0, 0]))
            # self.move_robot(np.array([0.0, 0.01, 0]))

            self.rgb_alt = self.cam.acquire_image()[0]
            self.move_robot(self.waypoint_list[self.current_waypoint])

            # For displaying depth image
            depth_display = (self.depth_img / self.depth_img.max() * 255).astype(np.uint8)
            self.image_window.update_image('Depth', np.dstack([depth_display] * 3), 256)

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


    def mask_click_callback(self, event):

        coord = self.convert_click_coord(event)
        print(coord)

        if self.last_click is None:
            self.last_click = coord
        else:
            x1, y1 = coord
            x2, y2 = self.last_click

            self.last_click = None

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            mask_upscaled = cv2.resize(self.mask_img, (self.rgb_main.shape[1], self.rgb_main.shape[0]))
            branch_binary = mask_upscaled[:, :, 0] > 128

            valid_pix = branch_binary[y1:y2,x1:x2]
            if valid_pix.sum() < 10:
                print('There are not enough branch pixels here to target!')
            else:

                yvals, xvals = np.where(valid_pix)
                pix = np.array([xvals, yvals]).T + [x1, y1]
                target = self.pc[pix[:,1], pix[:,0]].mean(axis=0)

                detection = (target, pix)
                print('New target at: {:.3f}, {:.3f}, {:.3f}'.format(*target))
                self.detections.append(detection)
                self.reset_detections()


    def convert_click_coord(self, event):
        click_x = event.pos().x()
        click_y = event.pos().y()

        pixmap_rect = self.image_window.labels['Mask'].pixmap().rect()
        pixmap_x = pixmap_rect.width()
        pixmap_y = pixmap_rect.height()

        im_y, im_x = self.rgb_main.shape[:2]

        return int(click_x * im_x / pixmap_x), int(click_y * im_y / pixmap_y)

    def detect_prune_points(self):
        print('Automatic detection of pruning is currently not implemented')
        print('You can click on the mask box instead to define boxes!')


    def move_to_box(self):

        idx = self.detection_combobox.currentIndex()
        to_send = np.zeros(4)
        to_send[:3] = self.detections[idx][0]
        to_send[3] = -0.20

        print('Sending move to box command {}'.format(to_send))
        self.move_robot(to_send)

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
        self.callbacks = {}

        if names is None:
            names = [''] * len(layout)

        for col_fields, group_name in zip(layout, names):
            col = QVGroupBox(group_name)
            for field in col_fields:
                self.labels[field] = QLabel()
                self.labels[field].mousePressEvent = partial(self.click_callback, field)
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

    def click_callback(self, label, event):
        self.callbacks.get(label, lambda x: None)(event)

    def register_click_callback(self, label, callback):
        self.callbacks[label] = callback

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
        # 'test': False,
        'test': True,
        'test_camera': False,
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

    if not config['test']:
        config['socket_ip'] = '169.254.116.60'

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