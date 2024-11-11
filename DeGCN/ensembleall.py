import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    # 定义工作目录路径
    r1 = "work_dir/test/DeGCN_joint/weights"
    r2 = "TeGCN/work_dir/test/tegcn_jb"
    r3 = "sttformer/work_dir/test/joint"
    r4 = "sttformer/work_dir/test/motion"


    sr1 = "work_dir/test/DeGCN_jbfB/weights"
    sr2 = "work_dir/test/DeGCN_angle/weights"
    sr3 = "SkateFormer-main/work_dir/test/ske_joint"
    sr4 = "SkateFormer-main/work_dir/test/ske_bone"
    jr1 = "work_dir/test/DeGCN_jmf/weights"


    # 加载标签数据
    with open('data/test_label.npy', 'rb') as f:
        label = np.load(f)

    # 加载模型分数文件
    with open(os.path.join(r1, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())
    with open(os.path.join(r2, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())
    with open(os.path.join(r3, 'epoch1_test_score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3).items())
    with open(os.path.join(r4, 'epoch1_test_score.pkl'), 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(os.path.join(sr1, 'epoch1_test_score.pkl'), 'rb') as sr1:
        sr1 = list(pickle.load(sr1).items())
    with open(os.path.join(sr2, 'epoch1_test_score.pkl'), 'rb') as sr2:
        sr2 = list(pickle.load(sr2).items())
    with open(os.path.join(sr3, 'epoch1_test_score.pkl'), 'rb') as sr3:
        sr3 = list(pickle.load(sr3).items())
    with open(os.path.join(sr4, 'epoch1_test_score.pkl'), 'rb') as sr4:
        sr4 = list(pickle.load(sr4).items())

    with open(os.path.join(jr1, 'epoch1_test_score.pkl'), 'rb') as jr1:
        jr1 = list(pickle.load(jr1).items())


    # 初始化准确率统计变量
    right_num = right_num_5 = total_num = 0
    best = 0.0

    optimal_weights = [0.04122027, 0.17165557, 0.07938541, 0.05975812, 0.03982683, 0.16690768, 0.13782612, 0.13259714, 0.17082286]
    alpha = optimal_weights[:4]
    alpha2 = optimal_weights[4:8]
    alpha3 = optimal_weights[8:9]

    # 存储融合结果
    fused_results = []

    # 循环计算每个标签的准确率
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]

        _, sr11 = sr1[i]
        _, sr22 = sr2[i]
        _, sr33 = sr3[i]
        _, sr44 = sr4[i]

        _, jr11 = jr1[i]

        # 使用权重融合各模型的结果
        result1 = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]
        result2 = sr11 * alpha2[0] + sr22 * alpha2[1] + sr33 * alpha2[2] + sr44 * alpha2[3]
        result3 = jr11 * alpha3[0]

        # 将各融合结果求和
        r = result1 + result2 + result3
        fused_results.append(r)

        # 计算Top-5和Top-1准确率
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r_max = np.argmax(r)
        right_num += int(r_max == int(l))
        total_num += 1

    # 转换融合结果为numpy数组并保存
    fused_results_array = np.array(fused_results)
    np.save('pred.npy', fused_results_array)

    # 计算并输出Top-1和Top-5准确率
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    print('Fusion results saved to pred.npy')

    # 如果当前准确率比最佳准确率高，更新最佳结果
    if acc > best:
        best = acc
        best_alpha = alpha

    print("Best Top1 Acc: {:.4f}%, with weights: {}".format(best * 100, best_alpha))
