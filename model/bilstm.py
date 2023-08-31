import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMWithAttention_api_ioc_rule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.wordNum, config.embed)
        self.embedding_ip = nn.Embedding(3, config.ipEmbedLen, padding_idx=3 - 1)
        self.embedding_path = nn.Embedding(config.num_cluster + 1, config.pathEmbedLen, padding_idx=config.num_cluster)
        self.embedding_url = nn.Embedding(config.num_cluster + 1, config.urlEmbedLen, padding_idx=config.num_cluster)
        self.embedding_domain = nn.Embedding(3, config.domainEmbedLen, padding_idx=3 - 1)
        self.embedding_filename = nn.Embedding(config.num_cluster + 1, config.filenameEmbedLen,
                                               padding_idx=config.num_cluster)
        self.totalLen = config.embed + config.ipEmbedLen + config.urlEmbedLen + config.pathEmbedLen + config.domainEmbedLen + config.filenameEmbedLen + config.ruleNum
        self.lstm = nn.LSTM(self.totalLen, config.hidden_size, num_layers=config.num_layers,
                            bidirectional=True, dropout=config.dropout, batch_first=True)

        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes_new)
        self.dropout = nn.Dropout(config.dropout)

        self.attention = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, x):
        index = x[:, :, 0]
        ip = x[:, :, 1]
        url = x[:, :, 2]
        path = x[:, :, 3]
        domain = x[:, :, 4]
        filename = x[:, :, 5]
        rule = x[:, :, 6:]

        emb = self.embedding(index)
        ip = self.embedding_ip(ip)
        url = self.embedding_url(url)
        path = self.embedding_path(path)
        domain = self.embedding_domain(domain)
        filename = self.embedding_filename(filename)

        embedded = torch.cat((emb, ip, url, path, domain, filename, rule), 2)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        attention_weights = self.attention(output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = torch.bmm(output.transpose(1, 2), attention_weights).squeeze()
        logits = self.fc(weighted_output)
        return logits