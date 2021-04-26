# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# S: 表示解码输入开始的符号
# E: 表示解码输出开始的符号
# P: 如果当前的批次数据量小于时间步数，将填补空白序列的符号 ，P表示Padding的位置

def make_batch(sentences):
    """
    准备数据，数据处理，变成Tensor格式的单词id
    :param sentences:
    :type sentences:
    :return:
    :rtype:
    """
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_sinusoid_encoding_table(n_position, d_model):
    """
    Position Encoding, 给序列生成对应的嵌入， 根据论文公式得出
    :param n_position: 序列的长度，eg: 6
    :type n_position:
    :param d_model:  生成的嵌入的维度 eg: 512
    :type d_model:
    :return:
    :rtype:
    """
    def cal_angle(position, hid_idx):
        """

        :param position: 根据位置, eg:0
        :type position:int
        :param hid_idx: eg: 0
        :type hid_idx: int
        :return: 0.0
        :rtype:
        """
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        """
        根据位置信息计算角度, 对d_model 即512个维度，都计算
        :param position: int， eg： 1
        :type position:
        :return: list， 返回512长度的列表， eg: [1.0, 1.0, 0.9646616199111991, 0.9646616199111991, 0.930572040929699, 0.930572040929699, 0.8976871324473142, 0.8976871324473142, 0.8659643233600653, 0.8659643233600653, 0.8353625469578262, 0.8353625469578262, 0.8058421877614819, 0.8058421877614819, 0.7773650302387758, 0.7773650302387758, 0.7498942093324559, 0.7498942093324559, 0.7233941627366748, 0.7233941627366748, 0.6978305848598664, 0.6978305848598664, 0.6731703824144982, 0.6731703824144982, 0.6493816315762113, 0.6493816315762113, 0.6264335366568855, 0.6264335366568855, 0.6042963902381328, 0.6042963902381328, 0.5829415347136074, 0.5829415347136074, 0.5623413251903491, 0.5623413251903491, 0.5424690937011326, 0.5424690937011326, 0.5232991146814947, 0.5232991146814947, 0.504806571666747, 0.504806571666747, ,......]
        :rtype:
        """
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    # 得到[6,512]的矩阵, #代表6个位置
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    # dim 2i  偶数位置做sin
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # # dim 2i+1 奇数位置做cos
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    #转换成Tensor后返回
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    """
    padding 部分的注意力,
    :param seq_q:
    :type seq_q:
    :param seq_k:
    :type seq_k:
    :return:
    :rtype:
    """
    # 通过形状，获取batch大小和序列长度
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(0)表示的是Padding的部分，因为我们字典中把0作为了Padding
    # # batch_size x 1 x len_k(=len_q), 1表示这个位置被Mask了,
    # pad_attn_mask: tensor([[[False, False, False, False,  True]]])， 这里False表示是有单词的地方，True表示是Padding的地方, 维度是 torch.Size([1, 1, 5])
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # batch_size x len_q x len_k， expand表示扩充形状到--> torch.Size([1, 5, 5]),
    # eg: tensor([[[False, False, False, False,  True],
    #          [False, False, False, False,  True],
    #          [False, False, False, False,  True],
    #          [False, False, False, False,  True],
    #          [False, False, False, False,  True]]])
    result = pad_attn_mask.expand(batch_size, len_q, len_k)
    return result

def get_attn_subsequent_mask(seq):
    """
    序列部分的注意力
    :param seq: tensor([[5, 1, 2, 3, 4]]), 形状 【1，5】， batch_size, seq_len
    :type seq:
    :return:  torch.Size([1, 5, 5])
    :rtype:
    """
    # 设定下注意力的形状
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # eg: subsequent_mask : [[[0. 1. 1. 1. 1.],  [0. 0. 1. 1. 1.],  [0. 0. 0. 1. 1.],  [0. 0. 0. 0. 1.],  [0. 0. 0. 0. 0.]]]
    # 初始化一个注意力形状的上三角矩阵，因为我们从头往后预测，所以模型只有预测出来后才能看到单词, subsequent_mask, shape: torch.Size([1, 5, 5])
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    # 形状不变，转出tensor格式, torch.Size([1, 5, 5])
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        """
        带缩放的点积注意力
        """
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """

        :param Q: torch.Size([1, 8, 5, 64]),  batch_size, n_heads, seq_len, q_dimension
        :type Q:
        :param K: torch.Size([1, 8, 5, 64])
        :type K:
        :param V: torch.Size([1, 8, 5, 64])
        :type V:
        :param attn_mask:
        :type attn_mask:
        :return: context 维度 torch.Size([1, 8, 5, 64])，  attn 维度 torch.Size([1, 8, 5, 5]),
        :rtype:
        """
        #  K.transpose(-1, -2)之后的维度为torch.Size([1, 8, 64, 5]), matmul是矩阵相乘
        # torch.matmul(Q, K.transpose(-1, -2)) --> torch.Size([1, 8, 5, 5])， [1, 8, 5, 64] * [1, 8, 64, 5] 点积后的维度为 -->[1, 8, 5, 5], 即最后2维矩阵相乘
        # np.sqrt(d_k) : 8.0， 缩放因子
        # scores最终的维度是 [1, 8, 5, 5]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # 这里用到了mask，即mask的地方分数，我们是不要的，这里设置了一个很小的值,-1e9表示 -1000000000.0
        # 维度不变, 还是[1, 8, 5, 5]，这里是padding的位置，即最后一个位置时很小的值
        scores.masked_fill_(attn_mask, -1e9)
        # 计算attention值，softmax，在最后一个维度计算, attn的维度torch.Size([1, 8, 5, 5]) ，即每个单词位置的注意力，也可以说成关注点
        attn = nn.Softmax(dim=-1)(scores)
        # 和V矩阵相乘 context 形状 torch.Size([1, 8, 5, 64]), 即对V来说，哪些位置是值得注意的，意思是对于输入句子来说，每个单词相对于整个句子来说的重要性
        context = torch.matmul(attn, V)
        # context 维度 torch.Size([1, 8, 5, 64])，  attn 维度 torch.Size([1, 8, 5, 5]),
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        """
        attention 结构包含多头注意力+FeedForward（前馈层）
        attention 结构中的 Attention 组件
        """
        super(MultiHeadAttention, self).__init__()
        # 初始化Q，K，V
        # （512，64*8=512）Q，K，V 维度相同， 这里采用了简便方法，不用写8次线性Linear了
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads) #Linear(in_features: 512, out_features: 512, bias = True)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        #  初始一个Linear
        self.linear = nn.Linear(n_heads * d_v, d_model)
        # 层归一化，对应论文的Norm, 层归一化使得训练更稳定， 查看层归一化: https://www.jianshu.com/p/67ca488fde9f
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """

        :param Q:  输入维度， torch.Size([1, 5, 512])
        :type Q:
        :param K:  torch.Size([1, 5, 512])
        :type K:
        :param V:  torch.Size([1, 5, 512])
        :type V:
        :param attn_mask:  torch.Size([1, 5, 5]), [batch_size, seq_len,seq_len]
        :type attn_mask:
        :return: new_output: [1, 5, 512] [batch_size, 序列长度, 模型维度], attn: torch.Size([1, 8, 5, 5]), [batch_size, head数量, 序列长度，序列长度]
        :rtype:
        """
        # residual 代表的是论文中的残差的意思，即residual+attention后的输出作为下一个阶段的输入
        # residual残差来源于ResNet架构，目的是可以训练更深的网络，防止模型退化
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # view是改变形状，self.W_Q(Q)是计算Q后得到一个结果,得到的形状是torch.Size([1, 5, 512]), 然后view后，变成形状torch.Size([1, 5, 8, 64]),相当于拆出来的头的数量
        #  然后交互1和2的维度，变成了torch.Size([1, 8, 5, 64]), 即 [batch_size, n_heads, seq_len, K的维度]
        # 现在计算出8个头的q,k,v
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        # torch.Size([1, 5, 5]) --> torch.Size([1, 8, 5, 5])， 因为每个头都是一样的Mask计算，就是某些部分不计算注意力
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        # 开始计算Q，K，V, 缩放点积注意力，论文上注意力的计算公式，即公式1
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        #  得到最后的注意力，和经过注意力计算的V的到的context，  context.transpose(1, 2)得到的维度, torch.Size([1, 5, 8, 64])
        # contiguous没啥太大作用，就是保住你进行view更改形状前，保住这个矩阵的内存是连续的，不会出错
        # context的形状变成从 torch.Size([1, 5, 8, 64])--> torch.Size([1, 5, 512]), 即，合并了nheads, 就是合并了8个头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        #output的形状 torch.Size([1, 5, 512]) --> torch.Size([1, 5, 512])
        output = self.linear(context)
        # norm操作和残差, shape，torch.Size([1, 5, 512])， 做完残差和层归一化后的形状不变
        new_output = self.layer_norm(output + residual)
        return new_output, attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        """
        attention 结构中的 FeedForward 组件, 由2个1x1的卷积形成
        """
        super(PoswiseFeedForwardNet, self).__init__()
        ##内核大小为1的两个卷积。输入和输出的维度为 d_{model} =512，内层的维度为 d_{ff} =2048。
        # conv1和conv2卷积模型初始化， 输入的channel变化, d_model-->d_ff -->d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 层归一化,模型更加稳定
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        输出的形状应该是不变的，和输入的形状相同
        :param inputs: torch.Size([1, 5, 512])
        :type inputs:
        :return:  返回维度是[1, 5, 512]，
        :rtype:
        """
        # 也是做了残差结构
        residual = inputs # inputs : [batch_size, len_q, d_model]
        # 做卷积后用Relu激活，激活函数不改变形状, inputs.transpose(1, 2): 调换维度1和2的位置，torch.Size([1, 512, 5])
        # self.conv1(inputs.transpose(1, 2)) 卷积后的维度变化 torch.Size([1, 512, 5]) --> torch.Size([1, 2048, 5]), 通道数从512变成2048，这里的意义是，512维度的特征变成2048个特征
        # output的维度是[1, 2048, 5]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        # output在做一次卷积, self.conv2(output)后的形状变化:  [1, 2048, 5] --> .Size([1, 512, 5]) 通道数由2048变回512
        # 然后调换1，2维度，output的形状为 torch.Size([1, 5, 512])
        output = self.conv2(output).transpose(1, 2)
        # 残差部分和层归一化，维度不变，new_output [1, 5, 512]
        new_output = self.layer_norm(output + residual)
        return new_output

class EncoderLayer(nn.Module):
    def __init__(self):
        """
        Encoder 结构的层
        """
        super(EncoderLayer, self).__init__()
        # 初始化多头注意力
        self.enc_self_attn = MultiHeadAttention()
        #  初始化 FeedForward
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """

        :param enc_inputs: 词嵌入后的向量, 维度torch.Size([1, 5, 512])， batch_size, seq_len, embedding_size
        :type enc_inputs:
        :param enc_self_attn_mask:  注意力的mask，形状: torch.Size([1, 5, 5]), batch_size, seq_len, seq_len
        :type enc_self_attn_mask:
        :return: enc_outputs: ([1, 5, 512]),  [batch_size, len_q , d_model],   attn: torch.Size([1, 8, 5, 5]), 注意力的维度[batch_size, n_heads, seq_len, seq_len]
        :rtype:
        """
        # 这里Q和K和V都是输入的序列嵌入后的向量，enc_inputs的维度torch.Size([1, 5, 512])
        enc_outputs, attn = self.enc_self_attn(Q=enc_inputs, K=enc_inputs, V=enc_inputs, attn_mask=enc_self_attn_mask) # enc_inputs to same Q,K,V
        # attention的第二部分计算，FFN， FeedForward部分, enc_outputs的维度 torch.Size([1, 5, 512])
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        # 经过一层的计算后，输入enc_inputs和输出的enc_outputs维度相同，enc_outputs还会作为下一层的输入
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        """
        Decoder 结构的层 Decoder有2个attention结构和一个FeedForward结构
        """
        super(DecoderLayer, self).__init__()
        # self-attention部分
        self.dec_self_attn = MultiHeadAttention()
        # decoder, encoder attention
        self.dec_enc_attn = MultiHeadAttention()
        # FFN 层， FeedForwardNet
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """

        :param dec_inputs: torch.Size([1, 5, 512])
        :type dec_inputs:
        :param enc_outputs:  torch.Size([1, 5, 512])
        :type enc_outputs:
        :param dec_self_attn_mask:   torch.Size([1, 5, 5])
        :type dec_self_attn_mask:
        :param dec_enc_attn_mask:   torch.Size([1, 5, 5])
        :type dec_enc_attn_mask:
        :return:
        :rtype:
        """
        # 解码的输入的self-attention   dec_outputs: torch.Size([1, 5, 512]),  dec_self_attn: torch.Size([1, 8, 5, 5])
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # 解码的输出和编码的输出的隐层向量之间的attention， 计算之后是真正的dec_outputs，解码输出
        # dec_outputs: torch.Size([1, 5, 512])   dec_enc_attn: torch.Size([1, 8, 5, 5])
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # 解码输出经过FFN
        dec_outputs = self.pos_ffn(dec_outputs)
        # 得到返回结果
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        """
        Transformer的Encoder结构
        """
        super(Encoder, self).__init__()
        #初始化Embedding， (5,512)， 注意src_vocab_size是输入单词表的维度，输出是我们embedding的维度，这里是随机的初始化向量，这个向量可以被训练
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        #一个序列的位置嵌入的 Embedding初始化，get_sinusoid_encoding_table函数是创建一个Embedding的向量, 得到一个(6,512)
        postion_embedding_init = get_sinusoid_encoding_table(src_len + 1, d_model)
        # from_pretrained表示的是加载创建好的Embedding，freeze只加载，不训练，这里的不被训练
        self.pos_emb = nn.Embedding.from_pretrained(postion_embedding_init,freeze=True)
        # 初始化n_layers个层，每层都是一个attention的结构, ModuleList是接收一个列表，列表里面是所有层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """

        :param enc_inputs: 输入的形状， [batch_size x source_len]， 【1，5】
        :type enc_inputs:
        :return:
        :rtype:
        """
        # 先做Embedding, 这里是单词的Embedding加上单词的位置的Embedding，就得到了这个序列的每个单词的Embedding
        # # （1，5，512） 一共5个单词，每个单词用512维表示  1：表示batchsize为1
        # 维度变化 [1,5]-->（1，5，512）, 512代表嵌入的维度
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        # 返回维度为 batch_size x len1 x len2, 这个叫做self-attention,就是自己和自己计算注意力
        # self-attention的目标是计算一句话中每2个字之间的注意力，加入句子enc_inputs是: I love you ,就会两两计算： I 和love的注意力，I和you的注意力， love和you的注意力
        # 这里enc_self_attn_mask表示的是我这个句子中哪些位置是不需要计算注意力的，不需要计算的地方就是Mask掉的地方
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # 存储每层的注意力结果， 绘图用, eg：一共6层，存储6次注意力，每个注意力的维度是torch.Size([1, 8, 5, 5])
        enc_self_attns = []
        for layer in self.layers:
            # 循环每层,enc_outputs的维度： torch.Size([1, 5, 512]), enc_self_attn_mask的维度
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # 每次的attention都取出来，我们以后绘图用, enc_outputs做为下一层的而输入
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        """
        Transformer的Decoder结构
        """
        super(Decoder, self).__init__()
        # 目标序列的Embedding， 和Encoder一样，注意用的是tgt_vocab_size，即目标序列的字典
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model) #（7，512）
        # 位置嵌入
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        # Decoder的层初始化
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        """

        :param dec_inputs:  tensor([[5, 1, 2, 3, 4]])
        :type dec_inputs:
        :param enc_inputs:  tensor([[1, 2, 3, 4, 0]])
        :type enc_inputs:
        :param enc_outputs:  形状 torch.Size([1, 5, 512])
        :type enc_outputs:
        :return:
        :rtype:
        """
        # 和encoder一样，做单词嵌入和位置嵌入，相加后作为总的嵌入
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        # 和encoder一样，计算哪些位置时padding的，计算注意力时，剔除掉这个padding的位置
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # 上三角矩阵的注意力mask, 返回形状 torch.Size([1, 5, 5])
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        #gt: greater than ,dec_self_attn_pad_mask + dec_self_attn_subsequent_mask 得到的形状是:
        # tensor([[[0, 1, 1, 1, 1],
        #          [0, 0, 1, 1, 1],
        #          [0, 0, 0, 1, 1],
        #          [0, 0, 0, 0, 1],
        #          [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        #  和0比较，大于0的为True，否则为False
        # dec_self_attn_mask的得到的结果是
        # tensor([[[False, True, True, True, True],
        #          [False, False, True, True, True],
        #          [False, False, False, True, True],
        #          [False, False, False, False, True],
        #          [False, False, False, False, False]]])
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        # 计算pad的mask的位置, 这里是输入序列和输出序列之间计算
        # dec_enc_attn_mask计算后得到的结果是
        # tensor([[[False, False, False, False, True],
        #          [False, False, False, False, True],
        #          [False, False, False, False, True],
        #          [False, False, False, False, True],
        #          [False, False, False, False, True]]])
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        # 保存decoder 的self-attention值和 decoder encoder attention值
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: torch.Size([1, 5, 512])  dec_self_attn: torch.Size([1, 8, 5, 5]) dec_enc_attn: torch.Size([1, 8, 5, 5])
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    """
    Transformer结构
    """
    def __init__(self):
        """
        包含3个部分encoder，decoder和最后的输出层projection
        """
        super(Transformer, self).__init__()
        # 初始化编码器
        self.encoder = Encoder()
        # 初始化解码器
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False) #全连接
    def forward(self, enc_inputs, dec_inputs):
        """
        Transformer的前向网络
        :param enc_inputs:  eg: tensor([[1, 2, 3, 4, 0]]), shape: [batch_size, seq_len]， 这里是【1，5】，1代表batch大小，5代表序列长度
        :type enc_inputs:
        :param dec_inputs:  eg: tensor([[5, 1, 2, 3, 4]])
        :type dec_inputs:
        :return:
        :rtype:
        """
        #先把源序列enc_inputs进行 encoder,
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # encoder结束后得到encoder的向量和对应的attention值, 开始decoder   dec_outputs: torch.Size([1, 5, 512])
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        #线性映射  dec_logits: torch.Size([1, 5, 7])
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        #dec_logits.size(-1)表示最后一个维度, flatten_logit 拉平了维度, flatten_logit: torch.Size([5, 7]), 输出序列长度为5，每个单词对应的字典中的概率，字典中7个字
        flatten_logit = dec_logits.view(-1, dec_logits.size(-1))
        return flatten_logit, enc_self_attns, dec_self_attns, dec_enc_attns

def showgraph(attn):
    """
    画注意力的图
    :param attn:
    :type attn:
    :return:
    :rtype:
    """
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()

if __name__ == '__main__':
    # 我们的目标是让机器学会 让这个阿拉伯语翻译成英语， ich mochte ein bier  --> S i want a beer
    # ich mochte ein bier P 代表输入，例如要翻译的句子， P是Padding的首字母，即我的句子不是5个长度的时间步，但是我要变成5个， 所以做了Padding
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # P 代表Padding， 我们Padding对应的ID是0，源序列单词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)
    # 目标序列单词表
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    #目标单词表的大小
    tgt_vocab_size = len(tgt_vocab) # 7
    # 输入序列的长度
    src_len = 5 # length of source 输入5个单词
    # 输出序列的长度
    tgt_len = 5 # length of target 输出5个单词
    # 输入序列中每个单词embedding后的维度
    d_model = 512  # Embedding Size
    # FeedForward attention中 前馈层的维度
    d_ff = 2048  # FeedForward dimension
    # attention中注意力的K和V的维度
    d_k = d_v = 64  # dimension of K(=Q), V
    # 编码器和解码器层数，对应论文中的Nx
    n_layers = 6
    # 多头的注意力的头数量
    n_heads = 8

    #初始化Transformer模型
    model = Transformer()
    # 损失函数，交叉熵损失
    criterion = nn.CrossEntropyLoss()
    #优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 把输入的字符转换成tensor格式的 id， 分别是sentences[0]   sentences[1]    sentences[2]
    # enc_inputs:tensor([[1, 2, 3, 4, 0]]) dec_inputs:tensor([[5, 1, 2, 3, 4]]) target_batch tensor([[1, 2, 3, 4, 6]])
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    # 训练多20个epoch，即20轮
    for epoch in range(20):
        #每个epoch的时候，梯度清零
        optimizer.zero_grad()
        # 输入enc_inputs和dec_inputs，经过Transformer，得到输出
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        #计算和我们目标序列的损失，
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        # 打印
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        # 梯度计算
        loss.backward()
        # 更新参数
        optimizer.step()

    # 测试模型训练的好坏，这里假设
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    # 预测
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])
    # 显示编码器和和解码器的attention值
    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)