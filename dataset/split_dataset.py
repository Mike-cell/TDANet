import pandas as pd


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



if __name__ == "__main__":
    split_test_and_train()
    # replace_str()

