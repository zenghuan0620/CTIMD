import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerClassifier_CEMDS(nn.Module):
    def __init__(self, config):
        super(TransformerClassifier_CEMDS, self).__init__()
        self.batchsize=config.batch_size
        self.maxlen=config.seq_lenth
        self.embedding = nn.Embedding(config.wordNum, config.embed)
        self.embedding_ip = nn.Embedding(3, config.ipEmbedLen, padding_idx=3 - 1)
        self.embedding_path = nn.Embedding(config.num_cluster + 1, config.pathEmbedLen, padding_idx=config.num_cluster)
        self.embedding_url = nn.Embedding(config.num_cluster + 1, config.urlEmbedLen, padding_idx=config.num_cluster)
        self.embedding_domain = nn.Embedding(3, config.domainEmbedLen, padding_idx=3 - 1)
        self.embedding_filename = nn.Embedding(config.num_cluster + 1, config.filenameEmbedLen,
                                               padding_idx=config.num_cluster)
        self.totalLen = config.embed + config.ipEmbedLen + config.urlEmbedLen + config.pathEmbedLen + config.domainEmbedLen + config.filenameEmbedLen + config.ruleNum + 3

        self.positional_encoding = PositionalEncoding(self.totalLen, config.transformer_dropout)
        encoder_layer = nn.TransformerEncoderLayer(self.totalLen, config.transformer_nhead, config.hidden_size, config.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.transformer_num_layers)
        self.fc = nn.Linear(self.totalLen, config.num_classes_new)

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
        fill=torch.zeros(self.batchsize,self.maxlen,3,dtype=torch.int32).to("cuda")
        x = torch.cat((emb, ip, url, path, domain, filename, rule, fill), 2)

        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # average the output of all positions
        x = self.fc(x)
        return x