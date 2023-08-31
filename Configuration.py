class Config(object):

    """配置参数"""
    def __init__(self):
        self.wordNum=0              #单词数量
        self.embed=200              #词向量长度 原200
        self.hidden_size=256        #隐藏层1大小
        self.num_layers=2           #隐藏层层数   原2
        self.dropout=0.5            #dropout    原0.5
        self.hidden_size2=256       #隐藏层2大小 原256
        self.hidden_size3 = 64      #隐藏层3大小
        self.num_classes=8          #label数量
        self.num_epochs = 100       #训练几次
        self.batch_size = 256        #batch大小   原256
        self.data_path="data/security_data.csv"
        self.dict_path="data/api2index.json"
        self.csv_out_path="data/security_data_number.csv"
        self.pkl_out_path="data/security_data_number.pkl"
        self.seq_lenth=1200         #每个api序列的最大个数  原1200
        self.train_size=0.8         #训练样本占总样本的比重
        self.lr=0.001               #优化器的学习率 0.001

        self.ipEmbedLen=10          #ip嵌入长度
        self.urlEmbedLen = 10       #url嵌入长度
        self.pathEmbedLen = 10      #路径嵌入长度
        self.domainEmbedLen = 10    #域名嵌入长度
        self.filenameEmbedLen = 10  #文件名嵌入长度
        self.num_classes_new=2

        self.filter_sizes = (3, 4, 5, 6)  # 卷积核尺寸
        self.num_filters = 256      # 卷积核数量(channels数)

        self.num_cluster=20         #ioc簇的数量

        self.filter_sizes_lstmcnn = (3, 4, 5)  # lstmcnn卷积核尺寸

        self.ruleNum=3              #规则数量

        self.transformer_d_model=128            #原 512
        self.transformer_nhead=4                #原 8
        self.transformer_num_layers=2           #原 6
        self.transformer_dim_feedforward=512   #原 2048
        self.transformer_dropout=0.2            #原 0.1