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
print(torch.max(tag_scores01,1))