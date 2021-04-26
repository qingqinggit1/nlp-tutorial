# 代码来源：哔哩哔哩，小猿圈爬虫课程
# 发开人员：付金霞
# 开发时间：2021/4/25 11:59 上午

import torch

# a = torch.arange(8).reshape(2,4)
# print(a)
# a.transpose(0,1)    #只能2维的转换
# print(a)

# tensor([[0, 4],
#         [1, 5],
#         [2, 6],
#         [3, 7]])
# transpose 可以转换多维
# a = torch.arange(24).reshape(2,3,4)
# a.transpose(0,2).shape   #torch.Size([4, 3, 2])     #第0维和第2维置换

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
# 输入字符的个数
num_dic = {n: i for i, n in enumerate(char_arr)}


seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

batch_size = len(seq_data)
print(batch_size)