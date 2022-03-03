from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QComboBox, QRadioButton
from PyQt5.QtCore import QRect, QTimer, QObject, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPainter, QPixmap, QImage
import json


class Logger(QWidget):
    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.items = []

    def add_item(self, item):
        assert isinstance(item, LoggerItem)
        self.items.append(item)
        self.main_layout.addWidget(item)

    @property
    def complete(self):
        for item in self.items:
            if not item.complete:
                return False
        return True

    def serialize(self):
        return json.dumps([item.as_dict() for item in self.items])


class LoggerItem(QWidget):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = kwargs

    @property
    def complete(self):
        raise NotImplementedError()

    def init_layout(self):
        raise NotImplementedError()

    def as_dict(self):
        raise NotImplementedError()


class MsgItem(LoggerItem):
    def __init__(self, msg, **kwargs):
        super().__init__(**kwargs)
        self.msg = msg
        self.init_layout()

    def init_layout(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel(self.msg))

    def as_dict(self):
        return {
            'msg': self.msg,
            **self.data
        }

    @property
    def complete(self):
        return True

class ChoiceItem(LoggerItem):
    def __init__(self, msg, choices, **kwargs):
        super().__init__(**kwargs)
        self.msg = msg
        self.choices = choices
        self.buttons = []
        self.init_layout()

    def init_layout(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel(self.msg))
        choice_layout = QHBoxLayout()
        layout.addLayout(choice_layout)
        for choice in self.choices:
            button = QRadioButton(choice)
            self.buttons.append(button)
            choice_layout.addWidget(button)

    def get_choice(self):
        for button in self.buttons:
            if button.isChecked():
                return button.text()
        return None

    def as_dict(self):
        return {
            'msg': self.msg,
            'choice': self.get_choice(),
            **self.data
        }

    @property
    def complete(self):
        return bool(self.get_choice())


class PromptItem(LoggerItem):
    def __init__(self, msg, required=False, **kwargs):
        super().__init__(**kwargs)
        self.msg = msg
        self.required = required
        self.text_field = QLineEdit()
        self.init_layout()


    def init_layout(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel(self.msg))
        layout.addWidget(self.text_field)

    def as_dict(self):
        return {
            'msg': self.msg,
            'text': self.text_field.text(),
            **self.data
        }

    @property
    def complete(self):
        if not self.required:
            return True
        if not self.text_field.text().strip():
            return False
        return True

class TestLogger(QMainWindow):
    def __init__(self):
        super().__init__()
        widget = QWidget()
        self.setCentralWidget(widget)

        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.logger = Logger()
        layout.addWidget(self.logger)

        self.logger.add_item(MsgItem('This is a standard message.', time=0.1))
        self.logger.add_item(ChoiceItem('This is a choice message', choices=['Ton', 'Pei', 'Nan', 'Sha'], time=0.2))
        self.logger.add_item(PromptItem('What is your date of birth?*', required=True, time=0.3))
        self.logger.add_item(PromptItem('(Optional) What is your zodiac sign?', required=False, time=0.4))

        button = QPushButton('Save Results')
        layout.addWidget(button)
        button.clicked.connect(self.serialize)

    def serialize(self):

        if not self.logger.complete:
            print('Logger has not been fully filled out!')
        else:
            print(self.logger.serialize())



if __name__ == '__main__':

    app = QApplication([])
    gui = TestLogger()
    gui.show()
    app.exec_()