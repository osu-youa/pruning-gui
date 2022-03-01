from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy
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

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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
        self.last_click = None
        self.cutter_mask_clicks = []

        self.thread = QThread(self)
        self.thread.start()
        self.camera_and_network_handler = CameraAndNetworkHandler(config)
        self.camera_and_network_handler.moveToThread(self.thread)

        self.socket_thread = QThread(self)
        self.socket_thread.start()
        self.socket_handler = SocketHandler(self.config.get('socket_ip', 'localhost'), 10002, dummy=self.test)
        self.socket_handler.moveToThread(self.socket_thread)

        # Connect signals

        self.camera_and_network_handler.moving_start_signal.connect(partial(self.handle_move, 0))
        self.camera_and_network_handler.moving_end_signal.connect(partial(self.handle_move, 1))
        self.socket_handler.return_signal.connect(lambda x: self.action_panel_status.setText(x))

        # GUI setup
        self.image_window = ImageDisplay([['Main RGB', 'Alt RGB'], ['Depth', 'Flow'], ['Mask', 'Boxes']],
                                         names=['RGB', 'Depth/Flow', 'Processed'])
        self.image_window.register_click_callback('Mask', self.mask_click_callback)
        self.image_window.register_click_callback('Main RGB', self.cutter_mask_click_callback)
        self.image_window.setWindowTitle('Images')
        self.image_window.show()

        # Boilerplate setup
        self.setWindowTitle('Pruning GUI')
        left_side_layout = self.init_left_layout()
        middle_layout = self.init_middle_layout()
        right_side_layout = self.init_right_layout()
        action_panel_layout = self.init_action_panel()
        widget = QWidget()
        self.setCentralWidget(widget)
        vertical_layout = QVBoxLayout()
        top_level_layout = QHBoxLayout()
        widget.setLayout(vertical_layout)
        vertical_layout.addLayout(top_level_layout)
        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(middle_layout)
        top_level_layout.addLayout(right_side_layout)
        vertical_layout.addLayout(action_panel_layout)




        self.camera_and_network_handler.new_image_signal.connect(partial(self.image_window.update_image, width=256))
        self.camera_and_network_handler.status_signal.connect(self.update_status)


    def update_status(self, status):
        self.status.setText(status)

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
            self.image_window.update_image('Boxes', np.zeros((1,1,3), dtype=np.uint8))
            return

        if self.camera_and_network_handler.mask_img is None:
            raise Exception("Detections added but no mask has been generated!")

        colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        h, w = self.camera_and_network_handler.rgb_main.shape[:2]
        annotated_mask = cv2.resize(self.camera_and_network_handler.mask_img, (w,h))

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

    def handle_move(self, is_end):
        if not is_end:
            print('Started moving')
        else:
            print('Ended moving')
            self.actions_list.reset()

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


    @staticmethod
    def call_async(func, *args, **kwargs):
        QTimer.singleShot(0, partial(func, *args, **kwargs))


    def move_robot_waypoint(self):
        if not len(self.waypoint_list):
            return

        waypoint_index = self.waypoint_menu.currentData()
        target = self.waypoint_list[waypoint_index]

        self.call_async(self.camera_and_network_handler.move_robot, 1, target)
        self.reset_detections(clear=True)
        self.image_window.clear()


    def compute_flow(self):

        self.call_async(self.camera_and_network_handler.compute_flow)


    def mask_click_callback(self, event):

        coord = self.convert_click_coord(event, 'Mask')
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

            h, w = self.camera_and_network_handler.rgb_main.shape[:2]
            mask_upscaled = cv2.resize(self.camera_and_network_handler.mask_img, (w,h))
            branch_binary = mask_upscaled[:, :, 0] > 128

            valid_pix = branch_binary[y1:y2,x1:x2]
            if valid_pix.sum() < 10:
                print('There are not enough branch pixels here to target!')
            else:

                yvals, xvals = np.where(valid_pix)
                pix = np.array([xvals, yvals]).T + [x1, y1]
                pc = self.camera_and_network_handler.pc
                if pc is not None:

                    #DEBUGGING
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')

                    sel_pts = pc[pix[:,1], pix[:,0]]
                    sel_pts = sel_pts[(sel_pts[:,2] < 1.0) & (sel_pts[:,2] > 0.05)]
                    #
                    # temp_pc = self.pc.reshape(-1,3)
                    # temp_pc = temp_pc[temp_pc[:,2] < 2.0]

                    #
                    # ax.scatter3D(sel_pts[:,0], sel_pts[:,1], sel_pts[:,2])
                    # # ax.scatter3D(temp_pc[:,0], temp_pc[:,1], temp_pc[:,2])
                    # plt.show()

                    target = sel_pts.mean(axis=0)

                else:
                    print('No PC detected, filling with dummy detections')
                    target = np.random.uniform(-1, 1, 3)

                detection = (target, pix)
                print('New target at: {:.3f}, {:.3f}, {:.3f}'.format(*target))
                self.detections.append(detection)

                self.reset_detections()

    def cutter_mask_click_callback(self, event):
        coord = self.convert_click_coord(event, 'Main RGB')
        if not self.cutter_mask_clicks:
            self.cutter_mask_clicks.append(coord)
            print('Click at {} recorded!'.format(coord))
            return
        last_click = self.cutter_mask_clicks[-1]
        if np.linalg.norm(np.array(last_click) - coord) < 4:
            # Close the shape
            pts = np.array(self.cutter_mask_clicks)
            mask = np.zeros(self.camera_and_network_handler.rgb_main.shape[:2], dtype=np.uint8)
            mask = cv2.fillPoly(mask, pts.reshape((1,-1,2)), (255,255,255))

            Image.fromarray(mask).save('last_cutter_mask.png')
            print('New cutter mask defined!')

            self.cutter_mask_clicks = []
            self.camera_and_network_handler.load_cutter_gt()
        else:
            self.cutter_mask_clicks.append(coord)
            print('Click at {} recorded!'.format(coord))


    def convert_click_coord(self, event, window):
        click_x = event.pos().x()
        click_y = event.pos().y()

        pixmap_rect = self.image_window.labels[window].pixmap().rect()
        pixmap_x = pixmap_rect.width()
        pixmap_y = pixmap_rect.height()

        im_y, im_x = self.camera_and_network_handler.rgb_main.shape[:2]

        return int(click_x * im_x / pixmap_x), int(click_y * im_y / pixmap_y)

    def detect_prune_points(self):
        print('Automatic detection of pruning is currently not implemented')
        print('You can click on the mask box instead to define boxes!')


    def move_to_box(self):

        idx = self.detection_combobox.currentIndex()
        to_send = np.zeros(4)
        to_send[:3] = self.detections[idx][0]
        to_send[3] = -0.30

        print('Sending move to box command {}'.format(to_send))
        self.call_async(self.camera_and_network_handler.move_robot, 4, to_send)

    def execute_approach(self):
        self.call_async(self.camera_and_network_handler.execute_approach)


    def cut(self):
        if self.test:
            print('Cut!')
        else:
            print('No cutting will be done, will instead do retraction')
            self.call_async(self.camera_and_network_handler.move_robot, 0)


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
                                                 [self.camera_and_network_handler.acquire_image, self.compute_flow, self.detect_prune_points],
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

    def init_action_panel(self):

        self.action_panel_status = QLabel('')
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        abort_button = QPushButton('Abort')
        abort_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        abort_button.resize(250, 100)
        abort_button.setStyleSheet("background-color: red; color: white")
        abort_button.clicked.connect(partial(self.socket_handler.send_array, np.array([0])))

        buttons = [abort_button]
        for button in buttons:
            button_layout.addWidget(button)

        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.action_panel_status)

        return main_layout


class SocketHandler(QObject):

    return_signal = pyqtSignal(str)

    def __init__(self, address, socket=None, dummy=False):

        super().__init__()

        self.address = address
        self.socket = socket
        self.dummy = dummy

    def send_array(self, array):



        msg = array.astype(np.float64).tobytes()

        if not self.dummy:

            ADDRESS = self.config.get('socket_ip', 'localhost')
            PORT = self.config['move_server_port']
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (ADDRESS, PORT)

            sock.connect(address)
            sock.sendall(msg)
            response = sock.recv(1024)
            sock.close()

        else:
            response = 'Sent dummy array command {}'.format(array)

        self.return_signal.emit(response)


class CameraAndNetworkHandler(QObject):

    new_image_signal = pyqtSignal(str, np.ndarray)
    moving_start_signal = pyqtSignal()
    moving_end_signal = pyqtSignal()
    status_signal = pyqtSignal(str)

    def __init__(self, config):

        self.config = config
        self.test = config['test']

        super().__init__()

        self.cam = None
        if not self.config.get('test_camera', True):
            self.cam = Camera(640, 480)

        # Loaded images
        self.rgb_main = None
        self.rgb_alt = None
        self.depth_img = None
        self.pc = None
        self.plane = None
        self.flow_img = None
        self.mask_img = None
        self.mask_detections = None
        self.cutter_gt = None

        # Loaded utilites
        self.image_processor = None
        self.identifier = None
        self.rl_system = None

        self.load_cutter_gt()

    def load_cutter_gt(self):
        if os.path.exists('last_cutter_mask.png'):
            self.cutter_gt = np.array(Image.open('last_cutter_mask.png'))
            if len(self.cutter_gt.shape) > 2:
                self.cutter_gt = self.cutter_gt.mean(axis=2).astype(np.uint8)

    def acquire_image(self):
        if self.cam is None:
            path = self.config['dummy_image_path']
            file_format = self.config['dummy_image_format']
            start_frame = np.random.randint(5, 20)

            self.rgb_alt = np.array(Image.open(os.path.join(path, file_format.format(frame=start_frame)))).astype(
                np.uint8)[:, :, :3]
            self.rgb_main = np.array(Image.open(os.path.join(path, file_format.format(frame=start_frame + 3)))).astype(
                np.uint8)[:, :, :3]
        else:
            self.rgb_main, self.depth_img = self.cam.acquire_image()
            self.pc = self.cam.acquire_pc(return_rgb=False)
            self.process_pc()

            self.move_robot(3, np.array([0.01, 0.0, 0]))

            self.rgb_alt = self.cam.acquire_image()[0]
            self.move_robot(0)
            depth_display = (self.depth_img / self.depth_img.max() * 255).astype(np.uint8)
            self.new_image_signal.emit('Depth', depth_display)

        self.new_image_signal.emit('Main RGB', self.rgb_main)
        self.new_image_signal.emit('Alt RGB', self.rgb_alt)


    def process_pc(self):
        # Used to determine the plane corresponding to the PC
        FAR_DIST = 2.0
        NEAR_DIST = 0.0
        QUANT = 0.25
        PERC_CUTOFF = 0.01

        pc = self.pc.reshape(-1, 3)
        total_points = len(pc)
        idx = (pc[:,2] > NEAR_DIST) & (pc[:,2] < FAR_DIST)
        pc = pc[idx]

        prop = len(pc) / total_points
        print('Proportion of foreground pts: {:.4f}'.format(prop))

        if prop < PERC_CUTOFF:

            print('There were not enough points to do a SVD! Using assumption of 30 cm away')
            self.plane = (np.array([0, 0, 1], np.array([0, 0, 0.30])))
            return

        lower_z = np.quantile(pc[:,2], QUANT)
        upper_z = np.quantile(pc[:,2], 1 - QUANT)

        pc = pc[(pc[:,2] > lower_z) & (pc[:,2] < upper_z)]
        center = pc.mean(axis=0)

        pc = pc - center

        if len(pc) > 1000:
            pc = pc[np.random.choice(len(pc), 1000, replace=False)]

        u, s, v = np.linalg.svd(pc, full_matrices=True)
        least_sig = v[2]
        self.plane = (center, least_sig)


    def move_robot(self, code, pt=None):

        # See move_server.py for defined codes

        self.moving_start_signal.emit()
        to_send = [code]
        if pt is not None:
            to_send.extend(pt)
        to_send = np.array(to_send, dtype=np.float64)

        if not self.test or (self.test and self.config['use_dummy_socket']):
            msg = to_send.tobytes()
            ADDRESS = self.config.get('socket_ip', 'localhost')
            PORT = self.config['move_server_port']
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (ADDRESS, PORT)

            sock.connect(address)
            sock.sendall(msg)
            response = sock.recv(1024)
            sock.close()
            print('Move server response: {}'.format(response))
        print('Sent robot command: {}'.format(', '.join(['{:.3f}'.format(x) for x in to_send])))

        self.moving_end_signal.emit()

    def load_networks(self):
        if self.image_processor is not None:
            return
        self.status_signal.emit('Loading networks...')

        self.image_processor = ImageProcessor((424, 240), (128, 128), use_flow=True, gan_name=self.config['gan_name'],
                                              gan_output_channels=self.config.get('gan_output_channels', 3))
        self.rl_system = PPO.load(self.config['rl_model_path'])
        self.status_signal.emit('Done loading networks!')

    def compute_flow(self):
        self.load_networks()
        self.image_processor.reset()
        self.image_processor.process(self.rgb_alt)

        self.mask_img = self.image_processor.process(self.rgb_main)
        self.flow_img = self.image_processor.last_flow

        if self.cutter_gt is not None:
            self.mask_img[:,:,-1] = cv2.resize(self.cutter_gt, (self.mask_img.shape[1], self.mask_img.shape[0]))

        self.new_image_signal.emit('Flow', self.flow_img)
        self.new_image_signal.emit('Mask', self.mask_img)

    def execute_approach(self):

        self.moving_start_signal.emit()

        sock = None
        if not self.test or (self.test and self.config['use_dummy_socket']):
            ADDRESS = self.config.get('socket_ip', 'localhost')
            PORT = self.config['vel_server_port']
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (ADDRESS, PORT)
            sock.connect(address)

        self.image_processor.reset()
        try:
            if self.cam is None:
                path = self.config['dummy_image_path']
                file_format = os.path.join(path, self.config['dummy_image_format'])
                start_frame = np.random.randint(0, 40)

                for i in range(np.random.randint(10, 30)):
                    file_path = file_format.format(frame=start_frame + i)
                    img = np.array(Image.open(file_path), dtype=np.uint8)
                    seg = self.image_processor.process(img)

                    self.new_image_signal.emit('Main RGB', img)
                    self.new_image_signal.emit('Flow', self.image_processor.last_flow)
                    self.new_image_signal.emit('Mask', seg)

                    action = self.rl_system.predict(seg)[0]
                    if sock is not None:
                        sock.sendall(action.tobytes())
                        np.frombuffer(sock.recv(1024), dtype=np.uint8) # Synchronization
                    else:
                        print(action)

            else:
                start = time.time()
                action_freq = 2
                duration = self.config.get('approach_duration', 10.0)

                last_time = 0.0

                while time.time() - start < duration:

                    elapsed = time.time() - last_time
                    if elapsed < 1 / action_freq:
                        time.sleep(1 / action_freq - elapsed)

                    # Get image from camera
                    img, _ = self.cam.acquire_image()
                    seg = self.image_processor.process(img)
                    if self.cutter_gt is not None:
                        seg[:,:,-1] = cv2.resize(self.cutter_gt, (seg.shape[1], seg.shape[0]))

                    self.new_image_signal.emit('Main RGB', img)
                    self.new_image_signal.emit('Flow', self.image_processor.last_flow)
                    self.new_image_signal.emit('Mask', seg)

                    action = self.rl_system.predict(seg)[0]
                    speed = self.config.get('cutter_speed', 0.03)
                    final_action = np.array([action[0] * speed, action[1] * speed, speed], dtype=np.float64)
                    print(final_action)

                    if sock is not None:
                        sock.sendall(final_action.tobytes())
                        response = np.frombuffer(sock.recv(1024), dtype=np.uint8)  # Synchronization
                        if response == 0:
                            break

        except Exception as e:
            print('Ran into an Exception!')
            print(e)
        finally:
            if sock is not None:
                sock.close()
        self.moving_end_signal.emit()
        print('Approach finished!')


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

        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            img_array = np.dstack([img_array] * 3)
        elif img_array.shape[2] == 2:
            img_array = np.dstack([img_array, img_array[:,:,1]])

        qimg = self.np_to_qimage(img_array)
        pixmap = QPixmap(qimg).scaledToWidth(width)
        self.labels[label].setPixmap(pixmap)

    def clear(self, key=None):
        if key:
            keys = [key]
        else:
            keys = self.labels.keys()
        for label in keys:
            self.update_image(label, np.zeros((1,1,3), dtype=np.uint8))

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


def async_wrapper(func):
    def new_func(*args, **kwargs):
        QTimer.singleShot(0, partial(func, *args, **kwargs))
    return new_func

if __name__ == '__main__':

    config = {
        # 'test': False,
        'test': True,
        'test_camera': False,
        # 'dummy_image_path': r'C:\Users\davijose\Pictures\TrainingData\GanTrainingPairsWithCutters\train',
        # 'dummy_image_format': 'render_{}_randomized_{:05d}.png',
        # 'dummy_image_path': r'C:\Users\davijose\Pictures\TrainingData\RealData\MartinDataCollection\20220109-141856-Downsized',
        # 'dummy_image_format': '{frame:04d}-C.png',
        'dummy_image_path': r'C:\Users\davijose\Pictures\frames',
        'dummy_image_format': '{frame:03d}.jpg',
        
        
        'gan_name': 'synthetic_flow_cutter_pix2pix',
        'gan_output_channels': 2,
        'use_dummy_socket': False,
        'move_server_port': 10000,
        'vel_server_port': 10001,
        'rl_model_path': r'C:\Users\davijose\PycharmProjects\pybullet-test\best_model_1_0.zip',
        'approach_duration': 20.0,

    }

    if not config['test']:
        config['socket_ip'] = '169.254.174.52'

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