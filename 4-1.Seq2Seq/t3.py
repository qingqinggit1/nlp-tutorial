# %%
# code by Tae Hwan Jung @graykode
import argparse
import numpy as np
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch(input_data):
    #输入批次，输出批次，目标批次，把单词转成Tensor向量格式，onehot类型
    input_batch, output_batch, target_batch = [], [], []
    #时输入和输出的字符串长度相同
    for seq in input_data: #seq:['man', 'women'];i=0,['manPP', 'women']; i=1 ['manPP', 'women']
        for i in range(2): #i:0
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [num_dic[n] for n in seq[0]]  #input：[15, 3, 16, 2, 2]
        output = [num_dic[n] for n in ('S' + seq[1])]  #output：[0, 25, 17, 15, 7, 16]
        target = [num_dic[n] for n in (seq[1] + 'E')]  #target：[25, 17, 15, 7, 16, 1]

        input_batch.append(np.eye(n_class)[input])  #shape：(5,29)
        output_batch.append(np.eye(n_class)[output]) #shape:(6,29）
        target_batch.append(target) # not one-hot  [[25, 17, 15, 7, 16, 1]]

    # make tensor 把向量转成tensor向量
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)

# Model
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        #初始化编码器单元，解码器单元和全连接层
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5) #RNN(29, 128, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5) #RNN(29, 128, dropout=0.5)

        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        '''
        前向传播，输入参数为（编码器输入，编码器隐层，解码器输入）
        '''
        #transpose(0,1) :保持不变。transpose(1,0):转置，这里为什么保持不变？？（5，6，29） （6，6，29）
        enc_input = enc_input.transpose(0, 1) # enc_input: [max_len(=n_step, time step), batch_size, n_class]
        dec_input = dec_input.transpose(0, 1) # dec_input: [max_len(=n_step, time step), batch_size, n_class]

        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        #经过编码器单元，获取编码器的状态输出，
        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        # outputs : [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]
        #获得解码器的输出
        outputs, _ = self.dec_cell(dec_input, enc_states)
        #经过全连接层的，模型输出
        model = self.fc(outputs) # model : [max_len+1(=6), batch_size, n_class]
        return model

if __name__ == '__main__':
    #输入或者输出单词的最大长度
    n_step = 5
    #隐层128个单元
    n_hidden = 128

    #输入的字符
    #单个字符的列表 ['S', 'E', 'P', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    #词库 ：转成字典格式  {'S': 0, 'E': 1, 'P': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 'g': 9, 'h': 10, 'i': 11, 'j': 12, 'k': 13, 'l': 14, 'm': 15, 'n': 16, 'o': 17, 'p': 18, 'q': 19, 'r': 20, 's': 21, 't': 22, 'u': 23, 'v': 24, 'w': 25, 'x': 26, 'y': 27, 'z': 28}
    num_dic = {n: i for i, n in enumerate(char_arr)}
    #输入和输出的序列对
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

    #n_class：29
    n_class = len(num_dic)
    #batch_size:6 批次大小
    batch_size = len(seq_data)

    #初始化模型
    model = Seq2Seq()

    #损失函数
    criterion = nn.CrossEntropyLoss()
    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #对数据的处理
    input_batch, output_batch, target_batch = make_batch(input_data=seq_data)

    for epoch in range(5000): #训练5000次
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        #初始化隐层为0，shape:(1,6,128)
        hidden = torch.zeros(1, batch_size, n_hidden)

        #优化器清0
        optimizer.zero_grad()
        # input_batch : [batch_size, max_len(=n_step, time step), n_class]   shape:(6,5,
        # output_batch : [batch_size, max_len+1(=n_step, time step) (becase of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot
        #计算输出
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1, batch_size, n_class]
        output = output.transpose(0, 1) # [batch_size, max_len+1(=6), n_class]
        loss = 0
        for i in range(0, len(target_batch)):
            # output[i] : [max_len+1, n_class, target_batch[i] : max_len+1]
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:  #每1000次输出一次结果
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        #损失反馈
        loss.backward()
        #优化器更新
        optimizer.step()

    # Test
    def translate(word):
        #输入为word,输出为：PPPP
        test_data = [[word, 'P' * len(word)]]
        input_batch, output_batch, _ = make_batch(input_data=test_data)

        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, 1, n_hidden)
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1(=6), batch_size(=1), n_class]

        predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
        decoded = [char_arr[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P', '')

    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))