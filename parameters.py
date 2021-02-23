import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
#device = torch.device("cpu")

embed_size_encoder = 128
embed_size_decoder = 128
hidden_size = 128
lstm_layers_encoder = 1
lstm_layers_decoder = 1
dropout  = 0.2

uniform_init = 0.1
learning_rate = 0.001
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 32

maxEpochs = 8
log_every = 10
test_every = 2000

max_patience = 5
max_trials = 5
