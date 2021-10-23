# douban_embedding
豆瓣中文影评差评分析

### 1. NLP
NLP（Natural Language Processing）是指自然语言处理，他的目的是让计算机可以听懂人话。

下面是我将2万条豆瓣影评训练之后，随意输入一段新影评交给神经网络，最终AI推断出的结果。
```python
    "很好，演技不错", 0.91799414 ===>好评
    "要是好就奇怪了", 0.19483969 ===>差评
    "一星给字幕", 0.0028086603 ===>差评
    "演技好，演技好，很差", 0.17192301 ===>差评
    "演技好，演技好，演技好，演技好，很差" 0.8373259 ===>好评
```
看完本篇文章，即可获得上述技能。


### 2. 读取数据

首先我们要找到待训练的数据集，我这里是一个csv文件，里面有从豆瓣上获取的影视评论50000条。

他的格式是如下这样的：

| 名称 | 评分 | 评论 | 分类 | 
| --- | --- | --- | --- |
| 电影名 | 1分到5分 | 评论内容 | 1 好评，0 差评 |

部分数据是这样的：
![2021-10-22_063822.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7ea7615a0e8d439eb1c61f91c70ca510~tplv-k3u1fbpfcp-watermark.image?)

代码是这样的：

```python
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
    
    # 评论内容用结巴分词以空格分词
    comments = jieba.cut(row[2]) 
    comment = " ".join(comments)
    sentences.append(comment)
    # 存入标签，1好评，0差评
    labels.append(int(row[3]))

    i = i + 1

    if i > 20000: break # 先取前2万条试验，取全部就注释

# 取出训练数据条数，分隔开测试数据条数
training_size = 16000
# 0到16000是训练数据
training_sentences = sentences[0:training_size]
training_labels = labels[0:training_size]
# 16000以后是测试数据
testing_sentences = sentences[training_size:]
testing_labels = labels[training_size:]
```

这里面做了几项工作：
1. 文件逐行读入，选取评论和标签字段。
2. 评论内容进行分词后存储。
3. 将数据切分为训练和测试两组。

#### 2.1 中文分词
重点说一下分词。

分词是中文特有的，英文不存在。

下面是一个英文句子。
> This is an English sentence.

请问这个句子，有几个词？

有6个，因为每个词之间有空格，计算机可以轻易识别处理。

| This | is | an | English | sentence | . | 
| --- | --- | --- | --- |--- | --- | 
| 1 | 2 | 3 | 4 |5 | 6 |

下面是一个中文句子。

> 欢迎访问我的掘金博客。

请问这个句子，有几个词？

恐怕你得读几遍，然后结合生活阅历，才能分出来，而且还带着各类纠结。

今天研究的重点不是分词，所以我们一笔带过，采用第三方的结巴分词实现。

**安装方法**

*代码对 Python 2/3 均兼容*

-   全自动安装：`easy_install jieba` 或者 `pip install jieba` / `pip3 install jieba`
-   半自动安装：先下载 <http://pypi.python.org/pypi/jieba/> ，解压后运行 `python setup.py install`
-   手动安装：[下载代码文件](https://gitee.com/bigcool/stutter-participle)将 jieba 目录放置于当前目录或者 site-packages 目录
-   通过 `import jieba` 来引用

引入之后，调用`jieba.cut("欢迎访问我的掘金博客。")`就可以分词了。

```python
import jieba
words = jieba.cut("欢迎访问我的掘金博客。") 
sentence = " ".join(words)
print(sentence) # 欢迎 访问 我 的 掘金 博客 。
```

为什么要有分词？因为词语是语言的最小单位，理解了词语才能理解语言，才知道说了啥。

对于中文来说，同一个的词语在不同语境下，分词方法不一样。

关注下面的“北京大学”：
```python
import jieba
sentence = " ".join(jieba.cut("欢迎来北京大学餐厅")) 
print(sentence) # 欢迎 来 北京大学 餐厅
sentence2 = " ".join(jieba.cut("欢迎来北京大学生志愿者中心")) 
print(sentence2) # 欢迎 来 北京 大学生 志愿者 中心
```

所以，中文的自然语言处理难就难在分词。

*至此，我们的产物是如下格式：*
```
sentences = ['我 喜欢 你','我 不 喜欢 他',……]
labels = [0,1，……]
```

### 3. 文本序列化
文本，其实计算机是无法直接认识文本的，它只认识0和1。

你之所以能看到这些文字、图片，是因为经过了多次转化。

就拿字母A来说，我们用65表示，转为二进制是0100 0001。

|二进制      |十进制 |缩写/字符| 解释 |
| :-------: | :--: | :-: | :---- |
| 0100 0001 |   65 |  A  | 大写字母A |
| 0100 0010 |   66 |  B  | 大写字母B |
| 0100 0011 |   67 |  C  | 大写字母C |
| 0100 0100 |   68 |  D  | 大写字母D |
| 0100 0101 |   69 |  E  | 大写字母E |

当你看到A、B、C时，其实到了计算机那里是0100 0001、0100 0010、0100 0011，它喜欢数字。

*Tips：这就是为什么当你比较字母大小是发现 A<B ，其实本质上是65<66。*

那么，我们的准备好的文本也需要转换为数字，这样更便于计算。

#### 3.1 fit_on_texts 分类
有一个类叫Tokenizer，它是分词器，用于给文本分类和序列化。

这里的分词器和上面我们说的中文分词不同，因为编程语言是老外发明的，人家不用特意分词，他起名叫分词器，就是给词语分类。

```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['我 喜欢 你','我 不 喜欢 他']
# 定义分词器
tokenizer = Tokenizer()
# 分词器处理文本，
tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index) # {'我': 1, '喜欢': 2, '你': 3, '不': 4, '他': 5}
```
上面做的就是找文本里有几类词语，并编上号。

看输出结果知道：2句话最终抽出5种不同的词语，编号1~5。

#### 3.2 texts_to_sequences 文本变序列
文本里所有的词语都有了编号，那么就可以用数字表示文本了。

```python
# 文本转化为数字序列
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences) # [[1, 2, 3], [1, 4, 2, 5]]
```

这样，计算机渐渐露出了笑容。

#### 3.3 pad_sequences 填充序列
虽然给它提供了数字，但这不是标准的，有长有短，计算机就是流水线，只吃统一标准的数据。

pad_sequences 会把序列处理成统一的长度，默认选择里面最长的一条，不够的补0。
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# padding='post' 后边填充， padding='pre'前面填充
padded = pad_sequences(sequences, padding='post')
print(padded) # [[1 2 3] [1 4 2 5]] -> [[1 2 3 0] [1 4 2 5]]
```
这样，长度都是一样了，计算机露出了开心的笑容。

少了可以补充，但是如果太长怎么办呢？

太长可以裁剪。

```python
# truncating='post' 裁剪后边， truncating='pre'裁剪前面
padded = pad_sequences(sequences, maxlen = 3,truncating='pre')
print(padded) # [[1, 2, 3], [1, 4, 2, 5]] -> [[1 2 3] [4 2 5]]
```

*至此，我们的产物是这样的格式：*
```
sentences = [[1 2 3 0] [1 4 2 5]]
labels = [0,1，……]
```

### 4. 构建模型

所谓模型，就是流水线设备。我们先来看一下流水线是什么感觉。


![流水线.gif](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/babf0228c74d46b4abd4e5caa9ce002a~tplv-k3u1fbpfcp-watermark.image?)

![流水线2.gif](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/74e85ad6768941b78d04c83f38cab0a5~tplv-k3u1fbpfcp-watermark.image?)

![流水线3.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/03a08bfd9540459b9df181b93053f3d8~tplv-k3u1fbpfcp-watermark.image?)

看完了吧，流水线的作用就是进来固定格式的原料，经过一层一层的处理，最终出去固定格式的成品。

模型也是这样，定义一层层的“设备”，配置好流程中的各项“指标”，等待上线生产。

```python
# 构建模型，定义各个层
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# 配置训练方法 loss=损失函数 optimizer=优化器 metrics=["准确率”]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

#### 4.1 Sequential 序列
你可以理解为整条流水线，里面包含各类设备（层）。

#### 4.2 Embedding 嵌入层
嵌入层，从字面意思我们就可以感受到这个层的气势。

![嵌入.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b66c107ad32a49289ff7c801e5dfae2e~tplv-k3u1fbpfcp-watermark.image?)

嵌入，就是插了很多个维度。一个词语用多个维度来表示。

下面说维度。

二维的是这样的（长，宽）：
![坐标系.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b80e27bcf663495ba371e927e9158982~tplv-k3u1fbpfcp-watermark.image?)

三维是这样的（长，宽，高）：

![三维坐标系.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a677776939324244aee5eb865b3f4459~tplv-k3u1fbpfcp-watermark.image?)

100维是什么样的，你能想象出来吗？除非物理学家，否则三维以上很难用空间来描述。但是，数据是很好体现的。

性别，职位，年龄，身高，肤色，这一下就是5维了，1000维是不是也能找到。

对于一个词，也是可以嵌入很多维度的。有了维度上的数值，我们就可以理解词语的轻重程度，可以计算词语间的关系。

如果我们给颜色设置R、B、G 3个维度：
| 颜色 | R |G |B |
| --- | --- | --- | --- |
| 红色|255|0|0|
| 绿色|0|255|0|
| 蓝色|0|0|255|
| 黄色|255|255|0|
| 白色|255|255|255|
| 黑色|0|0|0|

下面见证一下奇迹，懂色彩学的都知道，红色和绿色掺在一起是什么颜色？

来，跟我一起读：红色+绿色=黄色。

到数字上就是：[255,0,0]+[0,255,0] = [255,255,0]

这样，颜色的明暗程度，颜色间的关系，计算机就可以通过计算得出了。

只要标记的合理，其实计算机能够算出：国王+女性=女王、精彩=-糟糕，开心>微笑。

那你说，计算机是不是理解词语意思了，它不像你是感性理解，它全是数值计算。

嵌入层就是给词语标记合理的维度。

我们看一下嵌入层的定义：**Embedding(vocab_size, embedding_dim, input_length)**

- vocab_size：字典大小。有多少类词语。 
- embedding_dim：本层的输出大小。一个词用多少维表示。
- input_length：输入数据的维数。一句话有多少个词语，一般是max_length（训练集的最大长度）。

#### 4.3 GlobalAveragePooling1D 全局平均池化为一维

主要就是降维。我们最终只要一维的一个结果，就是好评或者差评，但是现在维度太多，需要降维。


#### 4.4 Dense 

这个也是降维，`Dense(64, activation='relu')`降到`Dense(1, activation='sigmoid')`，最终输出一个结果，就像前面流水线输入面粉、水、肉、菜等多种原材料，最终出来的是包子。

![神经网络.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6a5f81eff1e04936a8f3c670670291bc~tplv-k3u1fbpfcp-watermark.image?)

#### 4.5 activation 激活函数

activation是激活函数，它的主要作用是提供网络的非线性建模能力。

所谓线性问题就是可以用一条线能解决的问题。
可以来[TensorFlow游乐场](http://playground.tensorflow.org/)来试验。

如果是采用线性的思维，神经网络很快就能区分开这两种样本。
![relu.gif](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8460e945df294a1ca1dd82a987c6ac6b~tplv-k3u1fbpfcp-watermark.image?)

但如果是下面的这种样本，画一条直线是解决不了的。

![QQ截图20211023164946.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/de243023a37b40a9830311c586344c50~tplv-k3u1fbpfcp-watermark.image?)

如果是用relu激活函数，就可以很轻易区分。

![relu4.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/89e8ad4c46ff4c6e82047881b0cc3dbb~tplv-k3u1fbpfcp-watermark.image?)

这就是激活函数的作用。

常用的有如下几个，下面有它的函数和图形。

![未标题-1.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/15294e389f1742f1b1494a469b0e173c~tplv-k3u1fbpfcp-watermark.image?)

我们用到了relu和sigmoid。

- relu：线性整流函数（Rectified Linear Unit），最常用的激活函数。
- sigmoid：也叫Logistic函数，它可以将一个实数映射到(0,1)的区间。

`Dense(1, activation='sigmoid')`最后一个Dense我们就采用了sigmoid，因为我们的数据集中0是差评，1是好评，我们期望模型的输出结果数值也在0到1之间，这样我们就可以判断是更接近好评还是差评了。

### 4. 训练模型

#### 4.1 fit 训练
训练模型就相当于启动了流水线机器，传入训练数据和验证数据，调用fit方法就可以训练了。

```python
model.fit(training_padded, training_labels, epochs=num_epochs,
    validation_data=(testing_padded, testing_labels), verbose=2)
# 保存训练集结果
model.save_weights('checkpoint/checkpoint')
```

启动后，日志打印是这样的：
```python
Epoch 1/10 500/500 - 61s - loss: 0.6088 - accuracy: 0.6648 - val_loss: 0.5582 - val_accuracy: 0.7275 
Epoch 2/10 500/500 - 60s - loss: 0.4156 - accuracy: 0.8130 - val_loss: 0.5656 - val_accuracy: 0.7222 
Epoch 3/10 500/500 - 60s - loss: 0.2820 - accuracy: 0.8823 - val_loss: 0.6518 - val_accuracy: 0.7057
```
经过训练，神经网络会根据输入和输出自动调节参数，包括确定词语的具体维度，以及维度的数值取多少。这个过程变为黑盒了，这也是人工智能和传统程序设计不同的地方。

最后，调用save_weights可以把结果保存下来。

### 5. 自动分析结果

#### 5.1 predict 预测 
```python
sentences = [
    "很好，演技不错",
    "要是好就奇怪了",
    "一星给字幕",
    "演技好，演技好，很差",
    "演技好，演技好，演技好，演技好，很差"
]

# 分词处理
v_len = len(sentences)
for i in range(v_len):
    sentences[i] = " ".join(jieba.cut(sentences[i]) )

# 序列化
sequences = tokenizer.texts_to_sequences(sentences)
# 填充为标准长度
padded = pad_sequences(sequences, maxlen= max_length, padding='post', truncating='post')
# 预测
predicts = model.predict(np.array(padded))
# 打印结果
for i in range(len(sentences)):
    print(sentences[i],   predicts[i][0],'===>好评' if predicts[i][0] > 0.5 else '===>差评')

```
`model.predict()`会返回预测值，这不是个分类值，是个回归值（也可以做到分类值，比如输出1或者0，但是我们更想观察0.51和0.49有啥区别）。我们假设0.5是分界值，以上是好评，以下是差评。

最终打印出结果：

```python
很好，演技不错 0.93863165 ===>好评 
要是好就奇怪了 0.32386222 ===>差评 
一星给字幕 0.0030411482 ===>差评 
演技好，演技好，很差 0.21595979 ===>差评 
演技好，演技好，演技好，演技好，很差 0.71479297 ===>好评
```

*本文阅读对象为初级人员，为了便于理解，特意省略了部分细节，展现的知识点较为浅薄，旨在介绍流程和原理，仅做入门用。完整代码已上传github，地址点击此处 [https://github.com/hlwgy/douban_embedding](https://github.com/hlwgy/douban_embedding)*
