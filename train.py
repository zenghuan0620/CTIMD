import json
from Configuration import Config
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from model.textcnn import textcnn
from model.bilstm import BiLSTMWithAttention_api_ioc_rule
from model.transformer import TransformerClassifier_CEMDS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

path_black=r'../dataset/black.json'
path_white=r'../dataset/white.json'

def load_api_ioc_rule(type, config):
    if type == 'black':
        path = r'../dataset/black_IOC_match.json'
        path_rule=r'../dataset/black_rule_match.json'
    else:
        path = r'../dataset/white_IOC_match.json'
        path_rule = r'../dataset/white_rule_match.json'
    jsonFile = open(path, encoding='utf-8')
    json_file = json.load(jsonFile)
    rule=json.load(open(path_rule, encoding='utf-8'))
    t = []
    del jsonFile
    cluster=100/config.num_cluster
    for sha, apiList in json_file.items():
        for i in range(len(apiList)):
            apiList[i][2] = int(apiList[i][2] / cluster)
            apiList[i][3] = int(apiList[i][3] / cluster)
            apiList[i][5] = int(apiList[i][5] / cluster)
            apiList[i] += rule[sha][i]          #添加规则特征
            apiList[i] = torch.tensor(apiList[i]).unsqueeze(0)
        apiList = torch.cat(tuple(apiList), 0).unsqueeze(0)
        t.append(apiList)
    X = torch.cat(tuple(t), 0)
    if type == 'black':
        Y = [1] * len(json_file)
    else:
        Y = [0] * len(json_file)
    Y = torch.LongTensor(Y)
    del json_file
    return X, Y

def load_data_api_ioc_rule(config):
    jsonFile = open(r'../dataset/api2index.json')
    word_index = json.load(jsonFile)
    vocab_size = len(word_index) + 1
    config.wordNum = vocab_size

    X_black, Y_black = load_api_ioc_rule("black", config)
    X_white, Y_white = load_api_ioc_rule("white", config)

    X = torch.cat((X_black, X_white), 0)
    Y = torch.cat((Y_black, Y_white), 0)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=config.train_size, random_state=42, stratify=Y)

    return x_train, x_test, y_train, y_test, config

def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, x_test, y_train, y_test, config = load_data_api_ioc_rule(config)
    dataset_train = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)

    model = textcnn(config).to(device)
    # model = BiLSTMWithAttention_api_ioc_rule(config).to(device)
    # model = TransformerClassifier_CEMDS(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    loss_fun = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    for epoch in range(100):
        losses = []
        for index, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fun(output, y)
            losses.append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
        print('第{}轮，训练损失：{}'.format(epoch, np.mean(losses)))

    mudelName="../model/textcnn.mdl"
    torch.save(model, mudelName)

def test(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, x_test, y_train, y_test, config = load_data_api_ioc_rule(config)
    model = torch.load("../model/textcnn.mdl").to(device)
    dataset_test = TensorDataset(x_test, y_test)
    dataloader = DataLoader(dataset=dataset_test, batch_size=config.batch_size, shuffle=True)

    torch.cuda.empty_cache()
    predict = []
    yyy = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            output = model(x).cpu()
            pred = torch.max(output.data, 1)[1]
            predict += pred.tolist()
            yyy += y.cpu().tolist()
    print('accuracy:{},precision:{},recall:{},f1:{}'.format(
        accuracy_score(yyy, predict), precision_score(yyy, predict),
        recall_score(yyy, predict), f1_score(yyy, predict)))

if __name__ == '__main__':
    config = Config()
    train(config)
    print("train done")

    test(config)
    print("test done")