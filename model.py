#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch

class Encoder(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2ind, lstm_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.hid_size = hidden_size
        self.lstmLayers = lstm_layers

        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers=lstm_layers, bidirectional = True)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, source, source_lengths):
        m = source.shape[0]
        E = self.embed(source)
        output_packed, (h, c) = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E,source_lengths, enforce_sorted=False))
        output = torch.nn.utils.rnn.pad_packed_sequence(output_packed)
        output = output[0].view(m, source.shape[1], 2, self.hid_size)
        t = torch.cat((output[:,:,0,:], output[:,:,1,:]),2)
        output = self.dropout(t)
        return output, h[0], c[0]


class Decoder(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2ind, lstm_layers, dropout_rate):
        super(Decoder, self).__init__()
        self.hid_size = hidden_size
        self.lstmLayers = lstm_layers

        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers=lstm_layers)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, source, h, c):
        source = source.unsqueeze(0)
        E = self.embed(source)
        output, (h,c) = self.lstm(E, (h,c))
        output = self.dropout(output[0])
        
        return output, h, c


class Attention(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.W = torch.nn.Parameter(torch.rand(decoder_dim, encoder_dim)-0.5)


    def forward(self, h_e, h_d):
        sentence_length = h_e.shape[0]
        att = torch.matmul(h_d, self.W)  # transpose(h_e): n_batch x seq_len x 2*hidden_size
                                                                                     # att: n_batches x seq_len x hidden_size
        e = torch.bmm(att, torch.transpose(torch.transpose(h_e,0,1),1,2)) #transpose(h_e): n_batch x seq_len x 2*hidden_size
        alpha = torch.nn.functional.softmax(e, dim=2)
        a = torch.sum(alpha.permute(2,0,1)*h_e, dim = 0)
        output = torch.cat((a,h_d.squeeze(1)), 1)


        return output


class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        unkTokenIdx = word2ind[self.unkToken]
        padTokenIdx = word2ind[self.padToken]
        sents = [[word2ind.get(w, unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))
    
    def __init__(self, embed_size_encoder, embed_size_decoder, hidden_size, sourceWord2ind, targetWord2ind, unkToken, padToken, startToken, endToken, lstm_layers_encoder, lstm_layers_decoder, dropout):
        super(NMTmodel, self).__init__()
        self.startToken = startToken
        self.endToken = endToken
        self.unkToken = unkToken
        self.padToken = padToken

        self.sourceWord2ind = sourceWord2ind
        self.targetWord2ind = targetWord2ind

        self.encoder = Encoder(embed_size_encoder, hidden_size, sourceWord2ind, lstm_layers_encoder, dropout)
        self.decoder = Decoder(embed_size_decoder, hidden_size, targetWord2ind, lstm_layers_decoder, dropout)
        self.attention = Attention(hidden_size*2, hidden_size)
        self.projection = torch.nn.Linear(3*hidden_size, len(targetWord2ind))

    def forward(self, source, target):
        X = self.preparePaddedBatch(source, self.sourceWord2ind)
        Y = self.preparePaddedBatch(target, self.targetWord2ind)        
        device = next(self.parameters()).device

#        output_size = len(self.targetWord2ind)
#        res = torch.zeros(Y.shape[0], Y.shape[1]).to(device)
        source_lengths = [len(s) for s in source]
        h_e, h, c = self.encoder(X, source_lengths)

        h = h.unsqueeze(0)
        c = c.unsqueeze(0)

        sents_len = Y.shape[0]
        prev = Y[0]
#        Hc=0
        outputs = torch.zeros(sents_len, Y.shape[1], len(self.targetWord2ind)).to(device)
        for i in range(1, sents_len):
            h_d, h,c = self.decoder(prev, h,c) #h_d: n_batches x hid_size
            att = self.attention(h_e, h_d.unsqueeze(1))
            output = self.projection(att)
            prev = torch.argmax(output, 1)
            outputs[i] = output
        outputs = outputs[1:].reshape(-1, outputs.shape[2])
        Y = Y[1:].reshape(-1)

        H = torch.nn.functional.cross_entropy(outputs,Y,ignore_index=self.targetWord2ind[self.padToken])
        return H

    def translateSentence(self, sentence, limit=1000):
        wordset = list(self.targetWord2ind)
        X = self.preparePaddedBatch([sentence], self.sourceWord2ind)

        with torch.no_grad():
            source_lengths = [len(sentence)]
            h_e, h,c = self.encoder(X, source_lengths) #h_e: seq_len x n_batches, 2xhid_size 
                                                       # h, c: n_batches, hid_size -> only forward vectors 
            prev = self.preparePaddedBatch([[self.startToken]], self.targetWord2ind)[0]
            result = []
            h = h.unsqueeze(0)
            c = c.unsqueeze(0)
            for _ in range(limit):
                h_d, h,c = self.decoder(prev, h, c)
                #h_d - shape: n_batches x hid_size
                #h/c - shape: num_layers x num_batches x hid_size
                att = self.attention(h_e, h_d.unsqueeze(1))
                proj = self.projection(att)
                output = torch.nn.functional.softmax(proj, dim=1)
                prev = torch.argmax(output,1)
                if prev == self.targetWord2ind[self.endToken]: break
                result.append(wordset[prev])

        return result
