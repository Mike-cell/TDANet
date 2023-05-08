import random
import pandas as pd
from utils import handle_scp


def split_test_and_train(frac=0.1, scr=r'..\data\path\cv_all.scp'):
    # 读取 cv_all.scp 文件
    df = pd.read_csv(scr, delimiter=' ', header=None)

    # 随机抽取 10% 数据保存在文件 ts.scp 中
    ts_df = df.sample(frac=frac)
    ts_df.to_csv(scr.replace('cv_all', 'ts'), sep=' ', index=False, header=False)

    # 剩余数据保存在文件 tr.scp 中
    tr_df = df.drop(ts_df.index)
    tr_df.to_csv(scr.replace('cv_all', 'tr'), sep=' ', index=False, header=False)

def replace_str(scr="..\data\path\cv_all.scp"):
    with open(scr, "r") as f:
    # 读取文件内容
        content = f.read()

    # 替换字符
    content = content.replace('Datasets/LRS2/Standard/Process/TrainSet/aac/', 'data\\audio\\').replace('/', '\\')

    with open("..\data\path\cv_all1.scp", "w") as f:
        # 将修改后的内容写回文件
        f.write(content)

def create():
    

    data = handle_scp(r'data\path\tr.scp')

    n = 200000

    scp1 = dict()
    scp2 = dict()
    scp_mix = dict()
    with open(r'data\path\scp_mix.scp', 'w') as f:
    # 循环n次，每次随机选择不同的两个人物对应的文件
        s1 = open(r'data\path\tr_scp1.scp', 'w')
        s2 = open(r'data\path\tr_scp2.scp', 'w')
        for i in range(n):
            # 随机选择两个不同的人物
            while True:
                person1, person2 = random.sample(set([x.split('\\')[0] for x in data.keys()]), 2)
                if person1 != person2:
                    break
            
            # 随机选择两个不同的文件，并新建key值并将其value分别写入两个不同的文件
            file1, file2 = None, None
            while True:
                file1 = random.choice([x for x in data.keys() if x.startswith(f"{person1}\\")])
                file2 = random.choice([x for x in data.keys() if x.startswith(f"{person2}\\")])
                v = sorted([file1, file2])
                key = f"{v[0]}_{v[1]}"
                if file1 != file2 and (key not in scp1.keys()):  # 检查选取的两个文件是否与上次完全相同
                    break
            
            
            value1 = data[file1]
            value2 = data[file2]
            scp1[key] = value1
            scp2[key] = value2
            value_mix = 'data\\audio\\mix\\' + str(i).zfill(5) + '.wav'
            scp_mix[key] = value_mix
            
            s1.write(f"{key} {value1}\n")
            s2.write(f"{key} {value2}\n")
            f.write(f"{key} {value_mix}\n")

            # last_files[key1] = value1
        s1.close()
        s2.close()

import os
import wave
import numpy as np
from scipy.io.wavfile import read, write
def mix():
    filename1 = r'data\path\tr_scp1.scp'
    filename2 = r'data\path\tr_scp2.scp'
    # filename3 = r'data\path\tr_mix.scp'
    rate = 16000
    with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
        for i, (line1, line2, line3) in enumerate(zip(f1, f2, f3)) :
            line1 = line1.split()[1].strip()  # 去除行末换行符
            line2 = line2.split()[1].strip()
            # line3 = line3.split()[1].strip()
            # 打开两个音频文件，读取它们的音频数据和采样率
            with wave.open(line1, 'rb') as w1, wave.open(line2, 'rb') as w2:
                data1 = w1.readframes(-1)
                data2 = w2.readframes(-1)
                rate1 = f1.getframerate()
                rate2 = f2.getframerate()   
                # 将音频数据转换为ndarray数组
                audio1 = np.frombuffer(data1, dtype=np.short)
                audio2 = np.frombuffer(data2, dtype=np.short)
                # 找出两个音频数据中的最大长度
                length1 = len(audio1)
                length2 = len(audio2)
                max_length = max(length1, length2)
                # 用0在末尾补全时长较短的音频数据
                if length1 < max_length:
                    delta = max_length - length1
                    s = np.random.uniform(0, delta)
                    audio1 = np.concatenate((np.zeros((0, s)), audio1, np.zeros(s, delta)))
                if length2 < max_length:
                    delta = max_length - length2
                    s = np.random.uniform(0, delta)
                    audio2 = np.concatenate((np.zeros((0, s)), audio2, np.zeros(s, delta)))
                # 将两个音频数据叠加
                audio_sum = audio1 + audio2
                

                n1 = "data\\Dataset\\2speakers\\tr\\mix\\{:06d}.wav".format(i)
                n2 = "data\\Dataset\\2speakers\\tr\\s1\\{:06d}.wav".format(i)
                n3 = "data\\Dataset\\2speakers\\tr\\s2\\{:06d}.wav".format(i)
                # 将叠加后的音频数据保存为新文件
                if os.path.exists(n1) is False:
                    write(n1, max(rate1, rate2), audio_sum.astype(np.short))
                # 将补全后的音频数据保存
                if len(audio1) != len(audio2):
                    raise "len(audio1) != len(audio2)"
                if os.path.exists(n2) is False:
                    write(n2, rate1, audio1.astype(np.short))
                if os.path.exists(n3) is False:
                    write(n3, rate2, audio2.astype(np.short))
            if (i+1) % 100 == 0:
                print(f"{i+1} files has been process")


if __name__ == "__main__":
    # # linux 转 windows 格式
    # replace_str()
    # # 划分训练集和数据集
    # split_test_and_train()
    # create()
    mix()

