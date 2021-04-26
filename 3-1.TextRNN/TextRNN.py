# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:  #每一句话进行循环
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # 输入转化成向量【0，3】create (1~n-1) as input
        target = word_dict[word[-1]]  # 输出转成数字：create (n) as target, We usually call this 'casual language model'

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

class TextRNN(nn.Module):
    #初始化函数
    def __init__(self):
        super(TextRNN, self).__init__()
        #实例化rnn（7，5）
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        #权重，全连接线性函数
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        #
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model

if __name__ == '__main__':
    #输入2个单词，时间步
    n_step = 2 # number of cells(= number of Step)
    #隐藏层参数
    n_hidden = 5 # number of hidden units in one cell
    #输入的句子
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    #将输入的3个句子链接一起，用空格分开
    word_list = " ".join(sentences).split()
    #输入的所有单词去掉重复的，组成列表 set()函数 eg: ['dog', 'milk', 'love', 'hate', 'like', 'i', 'coffee']
    word_list = list(set(word_list))
    #将单词组成字典格式，enumerate()函数  [(i,w) for i, w in enumerate([1,2,3])]结果： [(0, 1), (1, 2), (2, 3)]
    word_dict = {w: i for i, w in enumerate(word_list)}
    #将索引对应单词
    number_dict = {i: w for i, w in enumerate(word_list)}
    #词库的长度，也就是类别的个数或输出的维度
    n_class = len(word_dict)
    #一个批次的数量
    batch_size = len(sentences)
    #初始化TextRNN模型
    model = TextRNN()
    #就算损失函数 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #优化器  梯度下降
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    #输入向量 转成Tensor格式的向量
    input_batch = torch.FloatTensor(input_batch)
    #输出向量
    target_batch = torch.LongTensor(target_batch)
    #以上是数据和模型的初始化
    #下面开始训练模型
    # Training
    for epoch in range(5000): #迭代5000次
        optimizer.zero_grad()  #梯度支持化为0

        # hidden : [num_layers * num_directions, batch, hidden_size] [1*7，
        hidden = torch.zeros(1, batch_size, n_hidden) #[1,3,5]
        # input_batch : [batch_size, n_step, n_class] [3,2,7]
        output = model(hidden, input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch) #计算输出与目标之间的损失值
        if (epoch + 1) % 1000 == 0: #每1000次，打印损失值
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    #开始预测
    #预测的输入
    input = [sen.split()[:2] for sen in sentences]

    # Predict 预测
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])