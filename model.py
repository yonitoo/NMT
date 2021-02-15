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
    
    def __init__(self, embed_size, hidden_size, word2ind_bg, word2ind_en, unkToken, padToken, endToken, 
                encoder_layers, decoder_layers, dropout):
        super(NMTmodel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word2ind_bg = word2ind_bg
        self.unkTokenBGIdx = word2ind_bg[unkToken]
        self.padTokenBGIdx = word2ind_bg[padToken]
        self.endTokenBGIdx = word2ind_bg[endToken]
        self.word2ind_en = word2ind_en
        self.unkTokenENIdx = word2ind_en[unkToken]
        self.padTokenENIdx = word2ind_en[padToken]
        self.endTokenENIdx = word2ind_en[endToken]

        self.embed_bg = torch.nn.Embedding(len(word2ind_bg), embed_size)
        self.embed_en = torch.nn.Embedding(len(word2ind_en), embed_size)
        self.encoder = torch.nn.LSTM(embed_size, hidden_size, num_layers = encoder_layers)
        self.decoder = torch.nn.LSTM(embed_size, hidden_size, num_layers = decoder_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(hidden_size, len(word2ind_bg))
    
    def forward(self, source, target):
        X1 = self.preparePaddedBatch(source, self.word2ind_en, self.unkTokenENIdx, self.padTokenENIdx)
        X1_E = self.embed_en(X1)
        X2 = self.preparePaddedBatch(target, self.word2ind_bg, self.unkTokenBGIdx, self.padTokenBGIdx)
        X2_E = self.embed_bg(X2[:-1])
        source_lengths = [len(s) for s in source]
        outputPackedSource, (hidden_source, _) = self.encoder(torch.nn.utils.rnn.pack_padded_sequence(X1_E, source_lengths, enforce_sorted=False))
        outputSource, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPackedSource)
        outputSource = outputSource.flatten(0, 1)

        target_lengths = [len(t) - 1 for t in target]
        outputPackedTarget, (hidden_target, _) = self.decoder(torch.nn.utils.rnn.pack_padded_sequence(X2_E, target_lengths, enforce_sorted=False), 
                                (hidden_source, torch.zeros(hidden_source.size()).to(next(self.parameters()).device)))
        outputTarget, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPackedTarget)
        
        Z1 = self.projection(self.dropout(outputTarget.flatten(0,1)))
        Y1_bar = X2[1:].flatten(0,1)
        H = torch.nn.functional.cross_entropy(Z1, Y1_bar, ignore_index=self.padTokenBGIdx)
        return H

    def translateSentence(self, sentence, limit=1000):

        device = next(self.parameters()).device
        def getWordFromIdx(dictionary, idx):
            if idx in dictionary.keys():
                return dictionary[idx]
            return 2

        tokens = [getWordFromIdx(self.word2ind_en, word) for word in sentence]
        tokens.insert(0, self.word2ind_en["<S>"])
        tokens.append(self.word2ind_en["</S>"])

        sentence_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)

        with torch.no_grad():
            enc_res, (h, c) = self.encoder(sentence_tensor, sentence_tensor.size())
        outputs = [self.word2ind_bg[startToken]]

        for _ in range(limit):
            previous_word = torch.LongTensor([outputs[-1]]).to(device)

            with torch.no_grad():
                output, h, c = self.decoder(previous_word, enc_res, h, c)
                best_guess = output.argmax(1).item()

            outputs.append(best_guess)

            if output.argmax(1).item() == self.word2ind_en["</S>"]:
                break
        revBulgarian ={v:k for k, v in self.word2ind_bg.items()}
        translated_sentence = [revBulgarian[idx] for idx in outputs]
        result = translated_sentence[1:]

        return result

        #ind2word_bg = dict(enumerate(self.word2ind_bg))
        #device = next(self.parameters()).device
        #with torch.no_grad():
        #X = self.preparePaddedBatch([[sentence[0]]], self.word2ind_en, self.unkTokenENIdx, self.padTokenENIdx)
        #Y = self.embed_en(X)
        #outputSource, (hidden_source, c_source) = self.encoder(Y)
        #size = list(hidden_source.size())
        #size[0] = limit
        #outputSources = torch.zeros(size, device=device)
        #outputSources[0] = outputSource.flatten(0, 1)
