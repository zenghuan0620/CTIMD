import torch
import torch.nn as nn
import torch.nn.functional as F

class textcnn(nn.Module):
    def __init__(self, config):
        super(textcnn, self).__init__()
        self.embedding = nn.Embedding(config.wordNum, config.embed, padding_idx=config.wordNum - 1)
        self.embedding_ip = nn.Embedding(3, config.ipEmbedLen, padding_idx=3 - 1)
        self.embedding_path = nn.Embedding(config.num_cluster+1, config.pathEmbedLen, padding_idx=config.num_cluster)
        self.embedding_url = nn.Embedding(config.num_cluster+1, config.urlEmbedLen, padding_idx=config.num_cluster)
        self.embedding_domain = nn.Embedding(3, config.domainEmbedLen, padding_idx=3 - 1)
        self.embedding_filename = nn.Embedding(config.num_cluster+1, config.filenameEmbedLen, padding_idx=config.num_cluster)
        self.totalLen = config.embed + config.ipEmbedLen + config.urlEmbedLen + config.pathEmbedLen + config.domainEmbedLen + config.filenameEmbedLen + config.ruleNum

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, self.totalLen)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size,config.num_classes_new)

        # self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes_new)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        index = x[:, :, 0]
        ip = x[:, :, 1]
        url = x[:, :, 2]
        path = x[:, :, 3]
        domain = x[:, :, 4]
        filename = x[:, :, 5]
        rule =x[:, :, 6:]

        emb = self.embedding(index)
        ip = self.embedding_ip(ip)
        url = self.embedding_url(url)
        path = self.embedding_path(path)
        domain = self.embedding_domain(domain)
        filename = self.embedding_filename(filename)
        out = torch.cat((emb, ip, url, path, domain, filename, rule), 2)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.fc1(out)
        return out