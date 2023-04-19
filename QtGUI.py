import sys
import time
import numpy as np
import pyaudio
import pyqtgraph as pg
from PyQt5.QtCore import QRect
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from datetime import datetime


class MyWidget(QWidget):

    def __init__(self):
        super().__init__()

        # 初始化UI界面，包括曲线图和按钮
        self.init_ui()
        # 初始化录音设参数
        self.init_recorder()

        # 定义定时器，用于每秒更新数据
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)

        # 初始化数据包括时间戳和计数器变量
        self.data_x, self.data_y = [], []
        self.timestamp = int(time.time())

        # 启动定时器
        # self.start_timer()

    def init_ui(self):
        # 设置窗口大小和标题
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Realtime Plot')

        # # 创建一个PlotWidget对象，用于绘制曲线图
        # self.plot_widget = pg.PlotWidget(self)
        # self.plot_widget.setGeometry(QRect(50, 50, 700, 500))

        # # 添加曲线图信息和坐标轴标签
        # self.plot = self.plot_widget.plot()
        # self.plot.setPen(pg.mkPen(color='r', width=2))
        # self.plot.setData([], [])
        # self.plot_widget.setLabel('left', 'Count')
        # self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # create the plot widgets
        self.plot_widgets = []
        self.plots = []
        for i in range(4):
            plot_widget = pg.PlotWidget(self)
            plot_widget.setGeometry(QRect(50 + 350 * (i % 2), 50 + 250 * (i // 2), 300, 200))
                    # 添加曲线图信息和坐标轴标签
            plot = plot_widget.plot()
            plot.setPen(pg.mkPen(color='r', width=2))
            plot.setData([], [])
            plot_widget.setLabel('left', 'Count')
            plot_widget.setLabel('bottom', 'Time (s)')
            self.plots.append(plot)
            self.plot_widgets.append(plot_widget)

        # 创建并添加两个按钮，分别用于启动和暂停计数器
        self.start_button = QPushButton('Start', self)
        self.start_button.setGeometry(QRect(150, 10, 100, 30))
        self.start_button.clicked.connect(self.start_timer)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.setGeometry(QRect(350, 10, 100, 30))
        self.stop_button.clicked.connect(self.stop_timer)
        
        # create a button to exit the program
        self.exit_btn = QPushButton('Exit', self)
        self.exit_btn.setGeometry(QRect(550, 10, 100, 30))
        self.exit_btn.clicked.connect(self.close)
        

    def init_recorder(self):
        # 设置录音参数
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 16000
        self.CHUNK = 2048
        self.DEVICE = 1
        self.SECORD = 10
        self.DATA = SECORD * RATE
        # 创建PyAudio对象
        self.p = pyaudio.PyAudio()
        data = np.array([[],], dtype=np.int16)
        show = 0
        def callback(in_data, frame_count, time_info, status):
            nonlocal show
            if True:
                # 将二进制音频数据转换为numpy数组
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                
                audio_data = audio_data.reshape(-1, self.CHANNELS).T
                data = np.concatenate((data, audio_data), axis=1)
                
                if len(data[0]) > self.DATA:
                    data = data[:, -self.DATA:]
                # print(data.shape)
                # 绘制音频数据的波形图
                show
                show += 1
                if show % 5 == 0:
                    for i in range(len(line)):
                        line[i].set_data(np.linspace(0, self.SECORD, len(data[i])), data[i])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    

                    # 将录制的音频数据写入 wav 文件
                    path = os.path.join(filepath, 'x_channel.wav') 
                    
                    if show % 20 == 0:
                        for i in range(self.CHANNELS):
                            
                            WAVE_OUTPUT_FILENAME = path.replace('x', str(i))
                            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                            wf.setnchannels(1)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            # data[i] = np.ascontiguousarray(data[i])
                            wf.writeframes(data[i].tostring())
                            wf.close()
                        print(recording)
                        print("音频已保存为 ", WAVE_OUTPUT_FILENAME)
                        plt.savefig(path.replace('x_channel.wav', 'wave.png'), dpi=300, bbox_inches='tight')
                        show = 0
                # 返回录音数据，继续录音
                return (in_data, pyaudio.paContinue)
            else:
                # 返回空的数据，停止录音
                return (in_data, pyaudio.paComplete)
        
        
        self.stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                        frames_per_buffer=self.CHUNK, stream_callback=callback,
                        input_device_index=self.DEVICE)
    
       
        
        
        

    def start_timer(self):
        # 启动定时器，每秒触发一次
        self.timer.start(1000)
        self.stream.start_stream()

    def stop_timer(self):
        # 停止定时器
        self.timer.stop()
        # 停止录音并关闭音频流
        self.stream.stop_stream()
        self.stream.close()
        
    def close(self):
        # 退出程序
        self.p.terminate()
        QApplication.instance().quit() 

    def update_data(self):
        # 计算时间差和计数器值
        current_timestamp = int(time.time())
        delta_time = current_timestamp - self.timestamp
        '''改变这里的数据,实现自定义数据更新'''
        count = np.random.randint(1, 10)

        # 更新数据列表
        self.data_x.append(delta_time)
        self.data_y.append(count)

        # 绘制曲线图，并更新坐标轴的范围
        self.plots[0].setData(self.data_x, self.data_y)
        self.plot_widgets[0].setXRange(max(0, delta_time - 10), delta_time + 1)
        self.plot_widgets[0].setYRange(0, max(self.data_y) + 1)

    def closeEvent(self, event):
        # 关闭窗口时停止定时器
        self.timer.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())
    
