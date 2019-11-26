import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torch.nn.functional as F
import sklearn.preprocessing as preprocessing
from sklearn import linear_model

# with open("processed.cleveland.data", "r") as data:
#     lines = data.readlines()
#     data = []
#     for line in lines:
#         line_list = line.rstrip("\n").split(",")
#         data.append(line_list)
# data = np.asarray(data)
# pd.set_option('display.max_columns', None)
# pd.set_option("display.max_row",None)
# pd.set_option('max_colwidth',100)
# form = pd.DataFrame(data,columns=["age", "sex", "pain_type", "blood_pressure", "cholestora", "blood_sugar",\
#                                   "electrocardiograph", "rate", "angina", "oldpeak", "slope",\
#                                   "number_of_major", "Thalassemia", "target"])
# form = form.apply(pd.to_numeric, errors='coerce')
# form = form.dropna()
# print(form.shape)
# # print(form)
# data = form.iloc[:,:-1]
# target = form.iloc[:,-1]

file = pd.read_csv("2.csv")
# print(file)
file = file.iloc[:,1:]
# data = file.iloc[:,1:-1]
data = file.loc[:,["pain_type_4.0",'slope_1.0','number_of_major_0.0','Thalassemia_7.0','sex','angina']]
# data = file.loc[:,['num_major_vessels','thalassemia_reversable defect','st_depression','max_heart_rate_achieved']]
# data.drop(['age','pain_type_2.0','electrocardiograph_1.0','electrocardiograph_2.0','slope_3.0','number_of_major_1.0','number_of_major_3.0'\
#               ,'Thalassemia_6.0'],axis=1,inplace=True)
target = file.iloc[:,-1]
data = np.asarray(data)
target = np.asarray(target)
# print(data)

epoch_size = 500
num_classes = 2
input_size = 6  # one-hot size
hidden_size = 3
output_size = 2# output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
num_layers = 1  # one-layer rnn
sequence_length = 297

class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers,batch_size,output_size):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,batch_first=True)
        self.rnn1 = nn.RNN(input_size=hidden_size, hidden_size=output_size,batch_first=True)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.output_size))
        # print(h_0.shape)
        # Reshape input
        x.view(x.size(0), self.batch_size, self.input_size)
        # print(x)
        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, a = self.rnn(x, h_0)
        # print("aaaaaaa",a)
        # out = Variable(torch.Tensor([out.view(-1, hidden_size)]))
        # print(out.shape)
        out, _ = self.rnn1(out,h_1)
        # print(out.shape)
        out = out.view(-1, num_classes)
        # print(out.shape)
        return out


# Instantiate RNN model
rnn = RNN(num_classes, input_size, hidden_size, num_layers,batch_size,output_size)
# print(rnn)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)

def preprocess(data):
    pd.set_option('display.max_columns', None)
    pd.set_option("display.max_row", None)
    pd.set_option('max_colwidth', 100)
    pass


def predict():
    for epoch in range(1,epoch_size+1):
        i = 1
        right = 0
        for data1,each in zip(data,target):
            if i<=198:
                data1 = Variable(torch.Tensor([[data1]]))
                each = Variable(torch.LongTensor([each]))
                outputs = rnn(data1)
                rnn.zero_grad()
                loss = criterion(outputs, each)
                loss.backward()
                optimizer.step()
                i += 1
                # print("epoch: %d, loss: %1.3f" % (i, loss.data))
            else:
                data1 = Variable(torch.Tensor([[data1]]))
                outputs = rnn(data1)
                pred = outputs.data.max(1, keepdim=True)
                if pred[1].data[0][0] == each:
                    # print(pred[1].data[0][0],each)
                    right +=1
                else:
                    # print(pred[1].data[0][0], each)
                    pass
                i+=1
        acc = right/(i-198)
        print("epoch: %d,right number: %d,test number: %d,acc: %1.3f"%(epoch,right,i-198,acc))
        # print(rnn.state_dict()) 查看权重
        # break
    # print("/n/n/n",processed_data)
    # out = rnn(processed_data)
    # pred = out.data.max(1, keepdim=True)
    # print(out)
    # print(pred)
    # if pred[1].data[0][0] == 1:
    #     result = "yes"
    # else:
    #     result = "no"
    # print("the prediction is : {}".format(result))



predict()
