import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Activation, SimpleRNN, Flatten, Input, LSTM, GRU, Reshape,Dropout
from keras.optimizers import Adam



poetry_file = 'poetry.txt'
weight_file = 'poetry_model.h5'
# 根据前六个字预测第七个字
max_len = 6
batch_size = 512
learning_rate = 0.001


puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']


def preprocess_file():
    # 语料文本内容
    files_content = ''
    with open(poetry_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 每行的末尾加上"]"符号代表一首诗结束
            for char in puncs:
                line = line.replace(char, "")
            files_content += line.strip() + "]"

    words = sorted(list(files_content))
    words.remove(']')
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    del counted_words[']']
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])
    print(wordPairs)
    words, _ = zip(*wordPairs)
    # word到id的映射
    word2num = dict((c, i + 1) for i, c in enumerate(words))
    print(word2num)
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, 0)
    print(word2numF('描'))
    return word2numF, num2word, words, files_content

word2numF, num2word, words, files_content = preprocess_file()

def data_generator():
    '''生成器生成数据'''
    i = 0
    while 1:
        x = files_content[i: i + max_len]
        y = files_content[i + max_len]

        puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》', ':']
        if len([i for i in puncs if i in x]) != 0:
            i += 1
            continue
        if len([i for i in puncs if i in y]) != 0:
            i += 1
            continue

        y_vec = np.zeros(
            shape=(1, len(words)),
            dtype=np.bool
        )
        y_vec[0, word2numF(y)] = 1.0

        x_vec = np.zeros(
            shape=(1, max_len,len(words)),
            dtype=np.bool
        )

        for t, char in enumerate(x):
            x_vec[0, t, word2numF(char)] = 1.0
        yield x_vec, y_vec
        i += 1

input_tensor = Input(shape=(max_len, len(words)))
print(input_tensor)
lstm = LSTM(512, return_sequences=True)(input_tensor)
print(lstm)
dropout = Dropout(0.6)(lstm)
lstm = LSTM(256)(dropout)
dropout = Dropout(0.6)(lstm)
dense = Dense(len(words), activation='softmax')(dropout)
model = Model(inputs=input_tensor, outputs=dense)
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



number_of_epoch = len(words)

model.fit_generator(
    generator=data_generator(),
    verbose=True,
    steps_per_epoch=batch_size,
    epochs=1,
    # callbacks=[
    #     keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
    #     LambdaCallback(on_epoch_end=self.generate_sample_result)
    # ]
)

sentence = model.predict()
print(sentence)