import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from string import punctuation
import pickle
from math import log
import sys

# torch
import torch
from torch import nn

from model import Model




def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    #return RandomBaseline()
    return StudentModel(device)


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device, method="SE"):

        self.device = device
        self.EMBEDDING_SIZE = 300
        
        if method == "IDF-AVG" or method == "SIF":
            self.model = WiCClassifier(input_features=4*self.EMBEDDING_SIZE, hidden_size=600)
            if method == "IDF-AVG":
                self.model.load_state_dict(torch.load("model/model_IDF_avg_GloVe.pth"))
            else:
                self.model.load_state_dict(torch.load("model/model_SIF_GloVe.pth"))
                self.principal_component = torch.load("model/principal_component.pt")
            self.model.to(self.device)
            self.model.eval()

        self.method = method
        self.word_embs = dict()

    def create_dicts(self) -> None:
        embs_file = "model/glove.840B.300d.txt"

        with open(embs_file) as emb_file:
        
            for line in emb_file:

                word, *word_emb = line.strip().split(' ')
                word_emb = torch.tensor([float(component_string) for component_string in word_emb])

                if len(word_emb) == self.EMBEDDING_SIZE:
                    self.word_embs[word] = word_emb
        
        if self.method != "SIF":
            with open("model/IDF_vocab.pickle", "rb") as input_file:
                self.IDF_vocab = pickle.load(input_file)
            N = 16000
            self.IDF_vocab = defaultdict(lambda: log(N + 1), self.IDF_vocab)

        else:
            with open("model/freq_vocab.pickle", "rb") as input_file:
                self.freq_vocab = pickle.load(input_file)
            self.freq_vocab = defaultdict(lambda: 0, self.freq_vocab) 
            self.Z = sum(self.freq_vocab.values()) # normalization constant

    def create_vectors_store(self) -> torch.Tensor:
        word_index = dict()
        vectors_store = []

        # pad token, index = 0
        vectors_store.append(torch.rand(self.EMBEDDING_SIZE))

        for word, vector in self.word_embs.items():
            word_index[word] = len(vectors_store)
            vectors_store.append(vector)

        self.word_index = word_index    
        vectors_store = torch.stack(vectors_store)
        return vectors_store

    def preprocess_sentence(self, sentence: str) -> str: # I remove punctuation, digits and lowerize the sentence
        cleaned_sentence = ''
        for char in sentence:
            if (char not in punctuation) and (char not in '“”') and (not char.isdigit()):
                cleaned_sentence = cleaned_sentence + char

        return cleaned_sentence.lower()

    def sentence2vector(self, sentence: str, alpha: float=None) -> Optional[torch.Tensor]:
        if alpha is None:
            sentence_word_embs = [self.IDF_vocab[word] * self.word_embs[word] for word in sentence.split(' ') if word in self.word_embs]
        else:
            sentence_word_embs = [(alpha / (alpha + ((self.freq_vocab[word] + 1) / (self.Z + 1)))) * self.word_embs[word] for word in sentence.split(' ') if word in self.word_embs]

        if len(sentence_word_embs) == 0:
            return None

        sentence_word_embs = torch.stack(sentence_word_embs) # tensor shape: (#words X #features)
        return torch.mean(sentence_word_embs, dim=0) # weighted average

    def sentence2indices(self, sentence: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        words = sentence.split(' ')
        vector = []
        for word in words:
            if word in self.word_embs:
                vector.append(self.word_index[word])

        return (None, None) if len(vector) == 0 else (torch.tensor(vector, dtype=torch.long), self.sentence2vector(sentence))

    def rnn_collate_fn(self, data_elements: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        X1, X2 = [de[0] for de in data_elements], [de[2] for de in data_elements]  # lists of index tensors
        X1_avg = [de[1] for de in data_elements]
        X1_avg = torch.stack(X1_avg)
        X2_avg = [de[3] for de in data_elements]
        X2_avg = torch.stack(X2_avg)

        # to implement the many-to-one strategy
        X1_lengths = torch.tensor([x.size(0) for x in X1], dtype=torch.long)
        X2_lengths = torch.tensor([x.size(0) for x in X2], dtype=torch.long)


        X1, X2 = torch.nn.utils.rnn.pad_sequence(X1, batch_first=True, padding_value=0), torch.nn.utils.rnn.pad_sequence(X2, batch_first=True, padding_value=0)  #  shape (batch_size x max_seq_len)


        return X1, X2, X1_lengths, X2_lengths, X1_avg, X2_avg

    @torch.no_grad()
    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of sentences!

        if len(self.word_embs) == 0:
            self.create_dicts()
            if self.method == "SE":
                vectors_store = self.create_vectors_store()
                self.model = WiCRecurrentClassifier(vectors_store, n_hidden=100, dropout=0.5, embedding_size=self.EMBEDDING_SIZE)
                self.model.load_state_dict(torch.load("model/model_sequence_encoding_GloVe.pth"))
                self.model.to(self.device)
                self.model.eval() 

        if self.method == "IDF-AVG":

            samples = None

            for sentence_pair in sentence_pairs:
                vector1 = self.sentence2vector(self.preprocess_sentence(sentence_pair['sentence1']))
                vector2 = self.sentence2vector(self.preprocess_sentence(sentence_pair['sentence2']))
                input_features = torch.cat((vector1, vector2, torch.abs(vector1 - vector2), vector1 * vector2), 0) # (v1, v2, |v1 - v2|, v1 * v2)
                samples = torch.cat((samples, input_features.unsqueeze(0)), 0) if samples is not None else input_features.unsqueeze(0)

            outputs = self.model(samples.to(self.device))

        elif self.method == "SE":

            samples = []

            for sentence_pair in sentence_pairs:
                preprocessed_sentence = self.preprocess_sentence(sentence_pair['sentence1'])
                vector1, avg_vector1 = self.sentence2indices(preprocessed_sentence)
                
                preprocessed_sentence = self.preprocess_sentence(sentence_pair['sentence2'])
                vector2, avg_vector2 = self.sentence2indices(preprocessed_sentence)

                sample = (vector1, avg_vector1, vector2, avg_vector2)
                samples.append(sample)

            X1, X2, X1_lengths, X2_lengths, X1_avg, X2_avg = self.rnn_collate_fn(samples)
            outputs = self.model(X1.to(self.device), X2.to(self.device), X1_lengths.to(self.device), X2_lengths.to(self.device), X1_avg.to(self.device), X2_avg.to(self.device))

        elif self.method == "SIF":

            samples = None

            vvt = torch.mm(self.principal_component, torch.transpose(self.principal_component, 0, 1)) # It is the matrix v * v^T, where v is the first singular vector (=self.principal_component)
            for sentence_pair in sentence_pairs:
                vector1 = self.sentence2vector(self.preprocess_sentence(sentence_pair['sentence1']), 1e-3)
                vector2 = self.sentence2vector(self.preprocess_sentence(sentence_pair['sentence2']), 1e-3)
                vector1 -= torch.mm(vvt, vector1.unsqueeze(1)).squeeze(1) # I remove from vector1 its projection to self.principal_component
                vector2 -= torch.mm(vvt, vector2.unsqueeze(1)).squeeze(1)

                input_features = torch.cat((vector1, vector2, torch.abs(vector1 - vector2), vector1 * vector2), 0) # (v1, v2, |v1 - v2|, v1 * v2)
                samples = torch.cat((samples, input_features.unsqueeze(0)), 0) if samples is not None else input_features.unsqueeze(0)
            
            outputs = self.model(samples.to(self.device))
        
        
        #round predictions to the closest integer
        predictions = torch.round(outputs['probabilities'])
        return ['True' if pred == 1.0 else 'False' for pred in predictions]

        

class WiCClassifier(nn.Module):

    def __init__(
        self,
        input_features: int,
        hidden_size: int
    ):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(input_features, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, 1)
        self.loss_fn = torch.nn.BCELoss()
        self.global_epoch = 0

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hidden_representation = self.hidden_layer(x)
        hidden_representation = torch.relu(hidden_representation)
        # extract the value from the tensor
        logits = self.output_layer(hidden_representation).squeeze(1)
        # we need to apply a sigmoid activation function
        probabilities = torch.sigmoid(logits)
        result = {'logits': logits, 'probabilities': probabilities}

        # compute loss
        if y is not None:
            loss = self.loss(probabilities, y)
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)

class WiCRecurrentClassifier(nn.Module):

    def __init__(
        self,
        vectors_store: torch.Tensor,
        n_hidden: int,
        dropout: float,
        embedding_size: int
    ):
        super().__init__()

        # embedding layer
        self.embedding = torch.nn.Embedding.from_pretrained(vectors_store)

        # recurrent layer
        self.rnn = torch.nn.LSTM(input_size=vectors_store.size(1), hidden_size=n_hidden, num_layers=1, batch_first=True, bidirectional=True)

        lstm_output_dim = 4*(2*n_hidden + embedding_size)
        hidden_size = lstm_output_dim//8

        # classification head
        self.lin1 = torch.nn.Linear(lstm_output_dim, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.lin2 = torch.nn.Linear(hidden_size, 1)

        # criterion
        self.loss_fn = torch.nn.BCELoss()
        
        self.global_epoch = 0

    def forward(
        self, 
        X1: torch.Tensor,
        X2: torch.Tensor,
        X1_length: torch.Tensor,
        X2_length: torch.Tensor,
        X1_avg: torch.Tensor,
        X2_avg: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        # embedding words from indices
        embedding_out1 = self.embedding(X1)
        embedding_out2 = self.embedding(X2)

        packed_input1 =  nn.utils.rnn.pack_padded_sequence(embedding_out1, X1_length, batch_first=True, enforce_sorted=False)
        packed_input2 =  nn.utils.rnn.pack_padded_sequence(embedding_out2, X2_length, batch_first=True, enforce_sorted=False)

        recurrent_out1 = self.rnn(packed_input1)[0]
        recurrent_out2 = self.rnn(packed_input2)[0]

        recurrent_out1 = nn.utils.rnn.pad_packed_sequence(recurrent_out1, batch_first=True)[0]
        recurrent_out2 = nn.utils.rnn.pad_packed_sequence(recurrent_out2, batch_first=True)[0]

        summary_vectors1 = torch.sum(recurrent_out1, 1)
        X1_length = X1_length.unsqueeze(1).expand_as(summary_vectors1)
        summary_vectors1 /= X1_length

        summary_vectors2 = torch.sum(recurrent_out2, 1)
        X2_length = X2_length.unsqueeze(1).expand_as(summary_vectors2)
        summary_vectors2 /= X2_length

        summary_vectors1 = torch.cat((summary_vectors1, X1_avg), 1)
        summary_vectors2 = torch.cat((summary_vectors2, X2_avg), 1)
        summary_vectors = torch.cat((summary_vectors1, summary_vectors2, torch.abs(summary_vectors1 - summary_vectors2), summary_vectors1 * summary_vectors2), 1)

        # now we can classify the reviews with a feedforward pass on the summary
        # vectors
        out = self.lin1(summary_vectors)
        out = self.dropout(torch.relu(out))
        out = self.lin2(out).squeeze(1)

        # compute logits (which are simply the out variable) and the actual probability distribution (pred, as it is the predicted distribution)
        logits = out
        # we need to apply a sigmoid activation function
        probabilities = torch.sigmoid(logits)

        result = {'logits': logits, 'probabilities': probabilities}

        # compute loss
        if y is not None:
            loss = self.loss(probabilities, y)
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)