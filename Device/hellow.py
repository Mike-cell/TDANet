import sys
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class CustomWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建控件及布局
        self.label1 = QtWidgets.QLabel('主标题')
        self.label2 = QtWidgets.QLabel('副标题')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.label2)
        self.setLayout(layout)

class MyWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建主窗口并设置标题
        self.setWindowTitle('PyQt窗口')

        # 创建绘图部件
        self.canvas = FigureCanvas(Figure())
        self.setCentralWidget(self.canvas)

        # 添加自定义控件作为子标题
        self.custom_widget = CustomWidget(self)
        self.canvas_layout = QtWidgets.QVBoxLayout(self.centralWidget())
        self.canvas_layout.insertWidget(0, self.custom_widget)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())