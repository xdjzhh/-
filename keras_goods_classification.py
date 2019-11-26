
from os import listdir
import json
import tensorflow.contrib.keras as kr
from keras.layers import Conv1D, Embedding, Dense, Input, Activation, \
    GlobalMaxPooling1D, TimeDistributed, \
    BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from numpy.random import shuffle
from numpy import arange
import numpy as np
import re

'''goods classification prediction using Keras'''


def pre_processing():
    # jd_root = '/home/shuangxi/Documents/jd_classification_batch/'
    # medicine_f = '/home/shuangxi/PycharmProjects/AnhuiTaxProject/resource/medicine_name_list.csv'
    # huagong_f = '/home/shuangxi/PycharmProjects/AnhuiTaxProject/resource/huagong_data/huagong.txt'
    # jixie_f = '/home/shuangxi/PycharmProjects/AnhuiTaxProject/resource/jixie_data/jixie.txt'
    #
    # jixie_set = set()
    # with open(jixie_f, 'rU') as handle:
    #     for line in handle:
    #         line = line.strip()
    #         line = line.split(':')[0]
    #         line = re.sub('厂?家?直销|现?货?供应|现货供?应?', '', line)
    #         line = line.strip()
    #         if '项目合作' in line:
    #             continue
    #         jixie_set.add(line)
    # huagong_set = list(jixie_set)
    # shuffle(huagong_set)
    # huagong_set = huagong_set[0:20000]
    # # for f in listdir(jd_root):
    # #     if f.endswith('.txt'):
    # #         with open(jd_root+f, 'rU') as handle:
    # # #             line = handle.readline()
    # # #             line = line.strip()
    # # #             second_line = handle.readline()
    # # #             second_line = second_line.strip()
    # # #             if second_line:
    # # #                 huagong_set.add(line)
    # # # jd_set = list(huagong_set)
    # # # jd_set = sorted(jd_set, key=lambda ele: len(ele), reverse=True)
    # # # jd_set = jd_set[0:20000]
    # # medicine_set = set()
    # # with open(medicine_f, 'rU') as handle:
    # #     for line in handle:
    # #         line = line.strip()
    # #         line = line.split(',')
    # #         if line:
    # #             line = line[0]
    # #             medicine_set.add(line)
    # #
    # with open('/home/shuangxi/PycharmProjects/AnhuiTaxProject/resource/jixie_name_list.txt', 'w') as handle:
    #     for item in huagong_set:
    #         handle.write(item+'\n')

    root = '/home/honghao/Documents/classification'
    chemical_f = root + 'huagong_name_list.txt'
    medicine_f = root + 'medicine_name_list.txt'
    machine_f = root + 'jixie_name_list.txt'
    jd_f = root + 'jd_name_list.txt'

    mapping = {'huagong_name_list.txt': '化工', 'jixie_name_list.txt': '机械',
               'jd_name_list.txt': '京东', 'medicine_name_list.txt': '药品'}

    train_data = []
    label_data = []
    for f in listdir(root):
        if f in mapping:
            label = mapping[f]
            with open(root + f, 'rU') as handle:
                print(root + f)
                for line in handle:
                    line = line.strip()
                    train_data.append(line)
                    label_data.append(label)

    data = {'train_data': train_data, 'label': label_data}
    with open('/home/honghao/Documents/classification/train_data.json', 'w') as handle:
        json.dump(data, handle)


class GoodsClassification(object):
    def __init__(self, vocabulary_f='', train_f=''):
        self.vocabulary_f = vocabulary_f
        self.train_f = train_f
        self.batch_size = 128
        self.max_sequence_length = 100
        self.words, self.word_to_id = self.read_vocab()
        self.classes, self.class_to_id = self.target_classes_to_id()

    def read_vocab(self):
        vocab_path = self.vocabulary_f
        with open(vocab_path, 'rU') as fp:
            words = [_.strip() for _ in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id

    @staticmethod
    def target_classes_to_id():
        classes = ['京东', '药品', '机械', '化工']
        class_to_id = dict(zip(classes, range(len(classes))))

        return classes, class_to_id

    @staticmethod
    def define_model():
        embedding_dim = 64
        num_classes = 4
        num_filters = 128
        kernel_size = 5
        vocab_size = 5000

        hidden_dim = 128

        input = Input(shape=(100,))
        x = Embedding(vocab_size, embedding_dim)(input)
        x = Conv1D(num_filters, kernel_size)(x)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling1D()(x)
        x = Activation('relu')(x)
        x = Dense(hidden_dim)(x)
        x = Activation('relu')(x)
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        model = Model(inputs=input, outputs=x)
        adam = Adam(lr=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

    def pre_process(self):
        '''准备训练集和验证集'''

        with open(self.train_f, 'rU') as handle:
            train_data = json.load(handle)
        train_label = train_data['label']
        train_data = train_data['train_data']
        data_id_list = []
        label_id_list = []
        for data, label in zip(train_data, train_label):
            data_id = [self.word_to_id[x] for x in data if x in self.word_to_id]
            data_id_list.append(data_id)
            label_id_list.append(self.class_to_id[label])

        x_pad = kr.preprocessing.sequence.pad_sequences(data_id_list, maxlen=self.max_sequence_length, padding='post')
        y_pad = kr.utils.to_categorical(label_id_list, num_classes=len(self.classes))

        index = arange(len(train_data))
        shuffle(index)
        x_pad = x_pad[index, :]
        y_pad = y_pad[index, :]

        return x_pad, y_pad

    def pre_process_for_test(self, goods_name):
        '''测试集中每一条数据预处理'''
        data_id = [self.word_to_id[x] for x in goods_name if x in self.word_to_id]
        data_id = [data_id]
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, maxlen=self.max_sequence_length, padding='post')
        return x_pad

    def train_model(self):
        model = self.define_model()
        x_pad, y_pad = self.pre_process()
        callbacks = [EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')]
        model.fit(x_pad, y_pad, validation_split=0.2, shuffle=True,
                  epochs=50, batch_size=128, verbose=1, callbacks=callbacks)
        model.save('goods_classification_cnn')
        model.save_weights('goods_classification_cnn_weights')

    def get_trained_model(self):
        model_weights_f = '/home/honghao/Documents/classification/goods_classification_cnn_weights'
        model = self.define_model()
        model.load_weights(model_weights_f)
        return model

    def predict_model(self):
        model_weights_f = '/home/honghao/Documents/classification/goods_classification_cnn_weights'
        model = self.define_model()
        model.load_weights(model_weights_f)
        x_pad, y_pad = self.pre_process()
        num_samples, num_dim = x_pad.shape
        for i in range(num_samples):
            individual_sample = x_pad[i, :]
            individual_target = y_pad[i, :]
            individual_sample = individual_sample.reshape(-1, 100)
            prediction = model.predict(individual_sample)
            prediction = np.argmax(prediction, axis=1).tolist()
            target = np.argmax(np.array(individual_target).reshape(1, -1), axis=1).tolist()
            print(prediction, target)

    def raw_predict_model(self):
        '''预测任何一个商品名称所属的分类'''
        pass


if __name__ == '__main__':
    goods_classification = GoodsClassification(train_f='train_data.json',
                                               vocabulary_f='/home/honghao/Documents/classification/vocab.txt')
    goods_classification.train_model()
    goods_classification.predict_model()

    # trained_model = goods_classification.get_trained_model()
    # count = 0
    # goods_f = '/home/shuangxi/Documents/Documents/AnhuiProject/unsolved_08_jua_2019.txt'
    # with open(goods_f, 'rU') as handle:
    #     for line in handle:
    #         line = line.strip()
    #         line = line.replace('(', '（')
    #         line = line.replace(')', '）')
    #         line = re.sub('【[^（]*?】|以下', '', line)
    #         line = re.sub('【[^】]*?$', '', line)
    #         line = re.sub('详见.*?$', '', line)
    #         line = re.sub('[\\[\\]"×\\-－\\*\\,\\da-zA-Z\\/\\s\\+&\\.°，_\\-Φ]{8,}', '', line)
    #         line = re.sub('（[\\.\\sa-zA-Z\\d或-]*?）', '', line)
    #         line = re.sub('（[^（]*?）', '', line)
    #         line = re.sub('（[^）]*?$', '', line)
    #
    #         hit = re.search('^.+\\d{6,}.+公司$|^.+\\d{6,}.+厂$|^.+\\d{6,}.+制药|^.+\\d{6,}.+药业', line)
    #         if hit:
    #             count += 1
    #             print(line)
    #
    #         # line_xpad = goods_classification.pre_process_for_test(line)
    #         # _prediction_prob = trained_model.predict(line_xpad)
    #         # _prediction = np.argmax(_prediction_prob, axis=1).tolist()[0]
    #         # _prediction_prob = _prediction_prob.tolist()[0]
    #         # if _prediction == 3:
    #         #     print(_prediction, _prediction_prob[_prediction], line)
    #         #     count += 1
    #         #     print(count)
    # print(count)
