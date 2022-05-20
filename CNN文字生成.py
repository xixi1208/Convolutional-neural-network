import time
import unidecode
import string
import random
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

text = unidecode.unidecode(open('shakespeare.txt').read())
text_len = len(text)
text[0:100]

all_characters = string.printable
n_characters = len(all_characters)
print(all_characters)

def random_chunk(chunk_len=200):
    idx_start = random.randint(0,text_len-chunk_len)
    idx_end = idx_start + chunk_len
    return text[idx_start:idx_end]
random_chunk()

def char2tensor(string):
    n = len(string)
    res = torch.LongTensor(n)
    for i in range(n):
        res[i] = all_characters.index(string[i])
    return res

class CharRNN(nn.Module):
    def __init__(self, dict_size, embed_size, hidden_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(dict_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, dict_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
def forward(self, x, h0, c0):
    # seq_len x batch_size
    o = self.embed(x)
    # seq_len x batch_sizex dict_size
    o, (h, c) = self.lstm(o, (h0, c0))
    # seq_len x batch_size x hidden_size
    o = o.view(-1, self.hidden_size)
    o = self.fc(o)
    # seq_len*batch_size x dict_size
    return o, h, c
def init_hidden(self, batch_size):
    return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size))
# 因为LSTM的输入，batch size 在第二个维度
def gen_data(batch_size,seq_len=100):
    inputs = torch.LongTensor(seq_len,batch_size)
    outputs = torch.LongTensor(seq_len,batch_size)
    for i in range(batch_size):
        strs = random_chunk(seq_len+1)
        inputs[:,i] = char2tensor(strs[0:seq_len])
        outputs[:,i] = char2tensor(strs[1:seq_len+1])
    return inputs,outputs

inputs,outputs = gen_data(2,22)
print(inputs[:,0])
print(outputs[:,0])

net = CharRNN(n_characters,120,100,1)
inputs,outputs = gen_data(2,9)
h0,c0 = net.init_hidden(2)
o,h,c = net(inputs,h0,c0)
print(o.shape)

criterion = nn.CrossEntropyLoss()
loss = criterion(o,outputs.view(-1))
print(loss)

def train_batch(net, optimizer, criterion, batch_size, seq_len, device):
    h0, c0 = net.init_hidden(batch_size)
    inputs, outputs = gen_data(batch_size, seq_len)
    h0, c0, inputs, outputs = h0.to(device), c0.to(device), inputs.to(device), outputs.to(device)

    optimizer.zero_grad()
    logit, h, c = net(inputs, h0, c0)
    loss = criterion(logit, outputs.view(-1))
    loss.backward()
    optimizer.step()

    return loss.item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nsteps = 5000
lr = 1e-3batch_size = 10
seq_len = 100

hidden_size = 200
embed_size = 200
num_layers = 2

net = CharRNN(n_characters,embed_size,hidden_size,num_layers).to(device)
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[int(nsteps*0.8)],gamma=0.1)

criterion = nn.CrossEntropyLoss()

loss_his = []
time_start = time.time()
for step in range(nsteps):
    loss = train_batch(net,optimizer,criterion,batch_size,seq_len,device)
    if step%100==0:
        time_end = time.time()
        print('%d/%d,%20took%20%.0f%20seconds\t%20%.2e'%(step+1,nsteps,time_end-time_start,loss))
        loss_his.append(loss)

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(loss_his)

def evaluate(net, char_start='I', len_pred=100, device=torch.device('cpu'), temperature=1):
    h, c = net.init_hidden(1)
    x = char2tensor(char_start).view(1, 1)
    h, c, x = h.to(device), c.to(device), x.to(device)

    res = [char_start]
    for i in range(len_pred):
        logit, h, c = net(x, h, c)
        prob = logit.div(temperature).exp()

        x = torch.multinomial(prob, 1)
        idx = x.squeeze().item()
        res.append(all_characters[idx])
    return res

device = torch.device('cpu')
net = net.to(device)
res = evaluate(net,'H',len_pred=20,device=device,temperature=2.0)
print(''.join(res))

res = evaluate(net,'H',len_pred=20,device=device,temperature=0.2)
print(''.join(res))

res = evaluate(net,'H',len_pred=20,device=device,temperature=0.2)
print(''.join(res))

res = evaluate(net,'H',len_pred=20,device=device,temperature=0.2)
print(''.join(res))

res = evaluate(net,'H',len_pred=20,device=device,temperature=0.2)
print(''.join(res))