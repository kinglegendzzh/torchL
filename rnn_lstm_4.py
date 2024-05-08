# 训练模型
import torch

from rnn_lstm_1 import tag_to_ix, word_to_ix, training_data
from rnn_lstm_2 import LSTMTagger, prepare_sequence
from rnn_lstm_3 import model, loss_function, optimizer

for epoch in range(400):  # 我们要训练400次。
    for sentence, tags in training_data:
        # 清除网络先前的梯度值
        model.zero_grad()
        # 重新初始化隐藏层数据
        model.hidden = model.init_hidden()
        # 按网络要求的格式处理输入数据和真实标签数据
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        # 实例化模型
        tag_scores = model(sentence_in)
        # 计算损失，反向传递梯度及更新模型参数
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
# 查看模型训练的结果
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(training_data[0][0])
print(tag_scores)
print(torch.max(tag_scores, 1))

# 测试模型
import torch

from rnn_lstm_1 import testing_data, word_to_ix
from rnn_lstm_2 import prepare_sequence
from rnn_lstm_3 import model

test_inputs = prepare_sequence(testing_data[0], word_to_ix)
tag_scores01 = model(test_inputs)
print(testing_data[0])
print(test_inputs)
print(tag_scores01)
print(torch.max(tag_scores01, 1))
