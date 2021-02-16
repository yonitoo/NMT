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

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind, unkTokenIdx, padTokenIdx):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w, unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))
    
    def __init__(self, embed_size, hidden_size, word2ind_bg, word2ind_en, startToken, unkToken, padToken, 
                endToken, encoder_layers, decoder_layers, dropout):
        super(NMTmodel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word2ind_bg = word2ind_bg
        self.startTokenBGIdx = word2ind_bg[startToken]
        self.unkTokenBGIdx = word2ind_bg[unkToken]
        self.padTokenBGIdx = word2ind_bg[padToken]
        self.endTokenBGIdx = word2ind_bg[endToken]
        self.word2ind_en = word2ind_en
        self.startTokenENIdx = word2ind_en[startToken]
        self.unkTokenENIdx = word2ind_en[unkToken]
        self.padTokenENIdx = word2ind_en[padToken]
        self.endTokenENIdx = word2ind_en[endToken]

        self.embed_bg = torch.nn.Embedding(len(word2ind_bg), embed_size)
        self.embed_en = torch.nn.Embedding(len(word2ind_en), embed_size)
        self.encoder = torch.nn.LSTM(embed_size, hidden_size, num_layers = encoder_layers)
        self.decoder = torch.nn.LSTM(embed_size, hidden_size, num_layers = decoder_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(hidden_size, len(word2ind_bg))
        self.attention = torch.nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, source, target):
        X1 = self.preparePaddedBatch(source, self.word2ind_en, self.unkTokenENIdx, self.padTokenENIdx)
        X1_E = self.embed_en(X1)
        X2 = self.preparePaddedBatch(target, self.word2ind_bg, self.unkTokenBGIdx, self.padTokenBGIdx)
        X2_E = self.embed_bg(X2[:-1])

        ###Encoder
        source_lengths = [len(s) for s in source]
        outputPackedSource, (hidden_source, state_source) = self.encoder(
            torch.nn.utils.rnn.pack_padded_sequence(X1_E, source_lengths, enforce_sorted=False))
        outputSource, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPackedSource)
        #outputSource = outputSource.flatten(0, 1)

        ###Decoder
        target_lengths = [len(t) - 1 for t in target]
        outputPackedTarget, (_, _) = self.decoder(torch.nn.utils.rnn.pack_padded_sequence(X2_E, 
                                target_lengths, enforce_sorted=False), (hidden_source, state_source))
        outputTarget, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPackedTarget)

        ###Attention
        #outputSource -> l1,batch,hidSize
        #outputTarget -> l2,batch,hidSize
        #torch.bmm -> batch, l1, hidSize | batch, hidSize, l2 -> batch, l1, l2
        #attentionWeights -> batch, l1, l2
        #torch.bmm outputSource and attentionWeights -> batch, hidSize, l1 | batch, l1, l2 -> batch, hidSize, l2
        #contextVector -> batch, hidSize, l2
        #outputTarget -> l2, batch, 2 * hidSize 

        attentionWeights = torch.nn.functional.softmax((torch.bmm(outputSource.permute(1, 0, 2),
                                                                outputTarget.permute(1, 2, 0))), dim = 1)
        contextVector = torch.bmm(outputSource.permute(1, 2, 0), attentionWeights).permute(2, 0, 1)
        outputTarget = self.attention(torch.cat((contextVector, outputTarget), dim = -1))
        ###
        
        Z1 = self.projection(self.dropout(outputTarget.flatten(0,1)))
        Y1_bar = X2[1:].flatten(0,1)
        H = torch.nn.functional.cross_entropy(Z1, Y1_bar, ignore_index=self.padTokenBGIdx)
        return H

    def translateSentence(self, sentence, limit=1000):

        ind2word = dict(enumerate(self.word2ind_bg))

        X = self.preparePaddedBatch([sentence], self.word2ind_en, self.unkTokenENIdx, self.padTokenENIdx)
        X_E = self.embed_en(X)
        outputSource, (hidden_source, state_source) = self.encoder(X_E)
        result = []
        inputSource = torch.tensor([[self.startTokenBGIdx]], device = next(self.parameters()).device)
        hidden_target = hidden_source
        state_target = state_source
        for _ in range(limit):
            
            outputTarget = self.embed_bg(inputSource)
            outputTarget, (hidden_target, state_target) = self.decoder(outputTarget, 
                                                                      (hidden_target, state_target))

            attentionWeights = torch.nn.functional.softmax((torch.bmm(outputSource.permute(1, 0, 2), 
                                                            outputTarget.permute(1, 2, 0))), dim = 1)
            contextVector = torch.bmm(outputSource.permute(1, 2, 0), attentionWeights).permute(2, 0, 1)
            outputTarget = self.attention(torch.cat((contextVector, outputTarget), dim = -1))

            Z = self.projection(self.dropout(outputTarget.flatten(0,1)))
            _, topIdx = torch.topk(Z.data, 1)
            currentWordIdx = topIdx[0].item()

            if currentWordIdx == self.endTokenBGIdx:
                break
            else:
                result.append(ind2word[currentWordIdx])
                inputSource = torch.tensor([[currentWordIdx]], device = next(self.parameters()).device)

        return result