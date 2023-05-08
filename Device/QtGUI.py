import sys
import time
import os
import wave
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

        # 定义定时器，用于10秒后记录数据
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.audio_save)
        self.timer2.setSingleShot(True)  # 设置定时器只触发一次

        # 初始化数据包括时间戳和计数器变量
        self.data_x, self.data_y = [], []
        self.timestamp = int(time.time())


    def init_ui(self):
        # 设置窗口大小和标题
        self.setGeometry(100, 100, 1600, 1200)
        self.setWindowTitle('Realtime Plot')
        
        # create the plot widgets
        self.plot_widgets = []
        self.plots = []
        for i in range(4):
            plot_widget = pg.PlotWidget(self)
            plot_widget.setGeometry(QRect(50 + 1000 * (i % 2), 50 + 500 * (i // 2), 800, 450))
            # 添加曲线图信息和坐标轴标签
            plot = plot_widget.plot()
            plot.setPen(pg.mkPen(color='r', width=2))
            plot.setData([], [])
            plot_widget.setLabel('left', 'Count')
            plot_widget.setLabel('bottom', 'Time (s)')
            plot.suptitle('')
            self.plots.append(plot)
            self.plot_widgets.append(plot_widget)
            self.plot_widgets[i].setYRange(-5000, 5000)

        # 创建并添加两个按钮，分别用于启动和暂停计数器
        self.start_button = QPushButton('Start', self)
        self.start_button.setGeometry(QRect(150, 10, 80, 30))
        self.start_button.clicked.connect(self.start_timer)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.setGeometry(QRect(250, 10, 80, 30))
        self.stop_button.clicked.connect(self.stop_timer)

        # 创建一个记录数据的按键，并绑定槽函数
        self.record_btn = QPushButton("Record", self)
        self.record_btn.setGeometry(QRect(350, 10, 80, 30))
        self.record_btn.clicked.connect(self.start_timer2)

        # create a button to exit the program
        self.exit_btn = QPushButton('Exit', self)
        self.exit_btn.setGeometry(QRect(450, 10, 80, 30))
        self.exit_btn.clicked.connect(self.close)
        

    def init_recorder(self):
        # 设置录音参数
        self.FORMAT = pyaudio.paInt16
        '''根据通道数修改'''
        self.CHANNELS = 4
        self.RATE = 16000
        self.CHUNK = 2048
        self.DEVICE = 1
        '''根据录音时长修改'''
        self.SECORD = 10
        self.DATA = self.SECORD * self.RATE
        # 创建PyAudio对象
        self.p = pyaudio.PyAudio()
        '''根据通道数修改'''
        self.data = np.array([[],[],[],[]], dtype=np.int16)
        self.show_t = 0
        self.recording = True

        def audio_callback(in_data, frame_count, time_info, status):

            if self.recording:
                # 将二进制音频数据转换为numpy数组
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                audio_data = audio_data.reshape(-1, self.CHANNELS).T
                self.data = np.concatenate((self.data, audio_data), axis=1)
                
                if len(self.data[0]) > self.DATA:
                    self.data = self.data[:, -self.DATA:]
                           
                # self.show_t += 1
              
                # # 将录制的音频数据写入 wav 文件
                # path = os.path.join('Recording\Record_files', 'x_channel.wav') 
                
                # if self.show_t % 20 == 0:
                #     # for i in range(self.CHANNELS):
                        
                #     self.show_t = 0
                # 返回录音数据，继续录音
                return (in_data, pyaudio.paContinue)
            else:
                # 返回空的数据，停止录音
                return (in_data, pyaudio.paComplete)

        
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                        frames_per_buffer=self.CHUNK, stream_callback=audio_callback,
                        input_device_index=self.DEVICE)
         
    def audio_save(self):
        for i in range(self.CHANNELS):   
            wf = wave.open(f"Recording\Record_files\{i}_channel.wav", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(self.data[i].tobytes())
            wf.close()
            screenshot = self.plot_widgets[i].grab()
            screenshot.save(f"Recording\Record_files\screenshot_{i}.png")
        
        print("音频已保存为:", r'Recording\Record_files\x_channel.wav')
        print("图片已保存为:", r'Recording\Record_files\x_channel.wav')

    def start_timer2(self):
        # 启动定时器，每秒触发一次
        self.timer2.start(10000)

    def start_timer(self):
        # 启动定时器，每秒触发一次
        self.timer.start(1000)
        self.stream.start_stream()

    def stop_timer(self):
        # 停止定时器
        self.timer.stop()
        # 停止录音并关闭音频流
        self.stream.stop_stream()
        
    def close(self):
        # 退出程序
        self.stream.stop_stream()
        self.stream.close()
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
        # self.plots[0].setData(self.data_x, self.data_y)
        # self.plot_widgets[0].setXRange(max(0, delta_time - 10), delta_time + 1)
        # self.plot_widgets[0].setYRange(0, max(self.data_y) + 1)
        x = np.linspace(0, self.SECORD, len(self.data[0]))
        for i, y in enumerate(self.data):
            self.plots[i].setData(x, y)
            self.plot_widgets[i].setXRange(0, 10.1)
            # self.plot_widgets[i].setYRange(min(y), max(y))
        

    def closeEvent(self, event):
        # 关闭窗口时停止定时器
        self.timer.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())
    
