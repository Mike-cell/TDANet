# import pyaudio
# import numpy as np
# import matplotlib.pyplot as plt
# import threading
# import keyboard


# # 检测计算机上的麦克风设备
# def search_device():

#     p = pyaudio.PyAudio()
#     info = p.get_host_api_info_by_index(0)
#     numdevices = info.get('deviceCount')
    
#     # 打印可用的输入设备列表
#     for i in range(0, numdevices):
#         if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#             print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


# search_device()



# # 设置录音参数
# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 16000
# CHUNK = 1024
# DEVICE = 2

# # 创建PyAudio对象
# p = pyaudio.PyAudio()

# # 创建绘图窗口
# fig, ax = plt.subplots(4)

# line = [[], [], [], []]
# for i in range(len(ax)):
#     line[i], = ax[i].plot([], [])

# # 定义全局变量，用于控制录音状态
# recording = False
# data = np.array([[], [], [], []], dtype=np.int16)

# # 定义回调函数
# def callback(in_data, frame_count, time_info, status):
#     global recording
#     global data
#     if recording:
#         # 将二进制音频数据转换为numpy数组
#         audio_data = np.frombuffer(in_data, dtype=np.int16)
#         print(audio_data.shape)
#         audio_data = audio_data.reshape(-1, CHANNELS * 2).T
#         data = np.concatenate((data, audio_data), axis=1)
#         data = data[:, -48000:]
#         # 绘制音频数据的波形图
     
#         for i in range(len(line)):
#             line[i].set_data(range(len(data[i])), data[i])
#             ax[i].relim()
#             ax[i].autoscale_view()
#         fig.canvas.draw()
#         fig.canvas.flush_events()

#         # 返回录音数据，继续录音
#         return (in_data, pyaudio.paContinue)
#     else:
#         # 返回空的数据，停止录音
#         return (in_data, pyaudio.paComplete)

# # 开始录音
# def start_recording():
#     global recording
#     # 打开录音设备，开始录音
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
#                     frames_per_buffer=CHUNK, stream_callback=callback,
#                     input_device_index=DEVICE)
#     stream.start_stream()
#     recording = True
#     while recording:
#         # 等待按键停止录音
#         keyboard.wait('esc')
#     # 停止录音并关闭音频流
#     stream.stop_stream()
#     stream.close()

# # 创建线程，用于开始录音
# record_thread = threading.Thread(target=start_recording)

# # 按下空格键开始录音，再次按下停止录音
# keyboard.add_hotkey('space', record_thread.start)
# keyboard.add_hotkey('esc', lambda: setattr(record_thread, "recording", False))

# # 显示音频波形图
# plt.show()

# # 关闭PyAudio对象
# p.terminate()


import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import keyboard
import wave


# 检测计算机上的麦克风设备
def search_device():

    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    # 打印可用的输入设备列表
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


search_device()

# 设置录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 4
RATE = 16000
CHUNK = 2048
DEVICE = 1

# 创建PyAudio对象
p = pyaudio.PyAudio()

# 创建绘图窗口
fig, ax = plt.subplots(4)

line = [[], [], [], []]
for i in range(len(ax)):
    line[i], = ax[i].plot([], [])
    ax[i].set_ylim(-10000, 10000)
    ax[i].set_xlim(0, 48000)

# 定义全局变量，用于控制录音状态
recording = False
data = np.array([[], [], [], []], dtype=np.int16)
show = 0

# 定义回调函数
import time 
def callback(in_data, frame_count, time_info, status):
    global recording
    global data
    
    if recording:
        # 将二进制音频数据转换为numpy数组
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        audio_data = audio_data.reshape(-1, CHANNELS).T
        data = np.concatenate((data, audio_data), axis=1)
        
        if len(data[0]) > 48000:
            data = data[:, -48000:]
        print(data.shape)
        # 绘制音频数据的波形图
     
        # for i in range(len(line)):
        #     line[i].set_data(range(len(data[i])), data[i])
            # ax[i].relim()
            # ax[i].autoscale_view()
        # time.sleep(0.01)
        global show 
        show += 1
        if show >= 20:
            for i in range(len(line)):
                line[i].set_data(range(len(data[i])), data[i])
            fig.canvas.draw()
            fig.canvas.flush_events()
            show = 0

            # 将录制的音频数据写入 wav 文件
            path = 'microphone\\4chanals\\171103\\x_channal.wav'
            WAVE_OUTPUT_FILENAME = [path.replace('x', str(i)) for i in range(CHANNELS)]
            for i, wavefile in enumerate(WAVE_OUTPUT_FILENAME):
                wf = wave.open(wavefile, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(data[i])
                wf.close()
            print("音频已保存为 ", WAVE_OUTPUT_FILENAME)
            # plt.savefig("microphone\\4chanals\\171103\\wav.png", dpi=300, bbox_inches='tight')
        # 返回录音数据，继续录音
        return (in_data, pyaudio.paContinue)
    else:
        # 返回空的数据，停止录音
        return (in_data, pyaudio.paComplete)

# 开始录音
def start_recording():
    
    global recording
    # 打开录音设备，开始录音
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    frames_per_buffer=CHUNK, stream_callback=callback,
                    input_device_index=DEVICE)
    
    stream.start_stream()
    recording = True
    
    while recording:
        # 等待按键停止录音
        keyboard.wait('esc')
    # 停止录音并关闭音频流
    stream.stop_stream()
    stream.close()

# 创建线程，用于开始录音
record_thread = threading.Thread(target=start_recording)


# 按下空格键开始录音，再次按下停止录音
keyboard.add_hotkey('space', record_thread.start)

keyboard.add_hotkey('esc', lambda: setattr(record_thread, "recording", False))


# 显示音频波形图
plt.show()


# 关闭PyAudio对象
p.terminate()

