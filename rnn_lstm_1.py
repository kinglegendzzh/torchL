# 词性判别
# 1.数据预处理
# 定义训练数据
training_data = [
       ("The cat ate the fish".split(), ["DET", "NN", "V", "DET", "NN"]),
       ("They read that book".split(), ["NN", "V", "DET", "NN"])
   ]
# 定义测试数据
testing_data=[("They ate the fish".split())]

# 构建每个单词的索引字典
word_to_ix = {} # 单词的索引字典
for sent, tags in training_data:
    for word in sent:
       if word not in word_to_ix:
           word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
# 两句话，共有9个不同单词
# {'The': 0, 'cat': 1, 'ate': 2, 'the': 3, 'fish': 4, 'They': 5, 'read': 6, 'that': 7, 'book': 8}
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
