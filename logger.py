from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QComboBox
from PyQt5.QtCore import QRect, QTimer, QObject, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPainter, QPixmap, QImage

class Logger(QWidget):
    def __init__(self):
        super().__init__()



class LoggerItem(QWidget):
    def __init__(self, msg=None, choices=None, use_response=False):
        super().__init__()