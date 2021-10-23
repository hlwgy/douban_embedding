# %% ================ 读入文件，组合数据 =======================
# 导入包
import csv
import jieba

# 读取csv文件
csv_reader = csv.reader(open("datasets/douban_comments.csv"))

# 存储句子和标签
sentences = []
labels = []

# 循环读出每一行进行处理
i = 1 
for row in csv_reader:
    
    # 评论内容用结巴分词器以空格分词
    comments = jieba.cut(row[2]) 
    comment = " ".join(comments)
    sentences.append(comment)
    # 存入标签，1好评，0差评
    labels.append(int(row[3]))

    i = i + 1

    if i > 20000: break

# 取出训练数据条数，分隔开测试数据条数
training_size = 16000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# %% ================ 进行训练 =======================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 定义分词器
tokenizer = Tokenizer(oov_token="<OOV>")
# 分词器处理训练文本
tokenizer.fit_on_texts(training_sentences)
# 训练文本转化为数字序列
training_sequences = tokenizer.texts_to_sequences(training_sentences)
# 数字序列填充
training_padded = pad_sequences(training_sequences, padding='post')

# 训练集最大长度
max_length = max(len(t) for t in training_sequences)
# 验证集处理为数字序列
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
# 验证集，不足的填充，超过的截取
testing_padded = pad_sequences(testing_sequences, maxlen = max_length, padding='post', truncating='post')

# 索引词组的数量，所有单词，外加填充的0
vocab_size = len(tokenizer.word_index) + 1
# 嵌入维度
embedding_dim = 256
# 把关键参数保存
np.save('params.npy',[vocab_size, embedding_dim, max_length])

# 构建模型
model = tf.keras.Sequential([ 
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% 进行训练
num_epochs = 10
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

history = model.fit(training_padded, training_labels, epochs=num_epochs,
    validation_data=(testing_padded, testing_labels), verbose=2)

# 保存训练集结果
model.save_weights('checkpoint/checkpoint')

# # %% ============== 验证数据 =============================
# sentences = [
#     "很好，演技不错",
#     "要是好就奇怪了",
#     "一星给字幕",
#     "演技好，演技好，很差",
#     "演技好，演技好，演技好，演技好，很差"
# ]

# # 分词处理
# v_len = len(sentences)
# for i in range(v_len):
#     sentences[i] = " ".join(jieba.cut(sentences[i]) )

# # 序列化
# sequences = tokenizer.texts_to_sequences(sentences)
# # 填充为标准长度
# padded = pad_sequences(sequences, maxlen= max_length, padding='post', truncating='post')
# # 预测
# predicts = model.predict(np.array(padded))
# # 打印结果
# for i in range(len(sentences)):
#     print(sentences[i],   predicts[i][0],'===>好评' if predicts[i][0] > 0.5 else '===>差评')

