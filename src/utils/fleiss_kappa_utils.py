"""
@Desc:
κ denotes Fleiss’
kappa (Fleiss and Joseph, 1971) to measure the interannotator agreement
@Reference:
- Fleiss Kappa百科
https://en.wikipedia.org/wiki/Fleiss%27_kappa#Equations
- FLeiss Kappa系数和Kappa系数的Python实现
https://blog.csdn.net/qq_31113079/article/details/76216611
@Note:
\kappa 	Interpretation
 Subjective example:
only for two annotators,
on two classes.
See Landis & Koch 1977	< 0	Poor agreement
0.01 – 0.20	Slight agreement
0.21 – 0.40	Fair agreement
0.41 – 0.60	Moderate agreement
0.61 – 0.80	Substantial agreement
0.81 – 1.00	Almost perfect agreement
"""

from typing import List
import numpy as np

def fleiss_kappa(test_data: List[List[int]], N, k, n):
    """
    @N: Subjects 对应 Evaluation里的samples
    @k: categories 对应 Evaluation里的choices
    @n: rators/annotators/evaluators
    @testData: a matrix with two dimensions; 行: samples 列: choices value: rator votes
    @desc:
    testData表示要计算的数据，（N,k）表示矩阵的形状，说明数据是N行j列的，一共有n个标注人员
    i 行索引: i-th subject， j 列索引: j-th category
    P_i 计算： （1/n(n-1))sum(n_ij(n_ij-1))
    P_j 计算： （1/Nn)sum(n_ij)
    bar_P: (1/N)sum(P_i)
    bar_P_e: sum(P_j^2)
    Kappa: (bar_P - bar_P_e)/(1 - bar_P_e)
    """
    # P_i 计算： （1/n(n-1))sum(n_ij(n_ij-1))
    # P_j 计算： （1/Nn)sum(n_ij)
    p_i_list = []
    p_j_list = []
    # P_i 计算： （1/n(n-1))sum_j(n_ij(n_ij-1))
    for i in range(N):
        sum_j_square = 0.0
        for j in range(k):
            sum_j_square += test_data[i][j] * test_data[i][j]
        p_i_list.append(1/(n * (n - 1)) * (sum_j_square - n))
    # P_j 计算： （1/Nn)sum_i(n_ij)
    for j in range(k):
        sum_i = 0.0
        for i in range(N):
            sum_i += test_data[i][j]
        p_j_list.append(1/(N * n) * sum_i)
    # kappa 计算
    bar_p = sum(p_i_list) / N
    bar_p_e = np.sum(np.square(p_j_list))

    f_kappa = (bar_p - bar_p_e) / (1 - bar_p_e)
    # 省略小数位 并转换为百分位
    f_kappa = round(f_kappa, 3) * 100

    return f_kappa


if __name__ == "__main__":
    data_arr = [[0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7]]
    f_kappa = fleiss_kappa(data_arr, 10, 5, 14)
    print(f_kappa)


