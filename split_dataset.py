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


if __name__ == "__main__":
    # # linux 转 windows 格式
    # replace_str()
    # # 划分训练集和数据集
    # split_test_and_train()
    create()
    pass
