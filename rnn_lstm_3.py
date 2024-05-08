# 训练网络
import torch
from torch import nn

from rnn_lstm_1 import tag_to_ix, word_to_ix, training_data
from rnn_lstm_2 import LSTMTagger, prepare_sequence

# 定义几个超参数、实例化模型，选择损失函数、优化器等
EMBEDDING_DIM = 10
HIDDEN_DIM = 3  # 这里等于词性个数
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

# 简单运行一次
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# tag_scores = model.forward(inputs)
print(training_data[0][0])
print(inputs)
print(tag_scores)
print(torch.max(tag_scores, 1))
