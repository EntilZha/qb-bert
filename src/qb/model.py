from typing import Dict, Text

import torch
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.nn import util


class Guesser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 dropout: float,
                 label_namespace: str = "page_labels"):
        super().__init__(vocab)
        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._bert.get_output_dim(), self._bert.get_output_dim()),
            nn.GELU(),
            nn.LayerNorm(self._bert.get_output_dim()),
            nn.Dropout(dropout),
            nn.Linear(self._bert.get_output_dim(), self._num_labels),
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def _hidden_to_output(self,
                hidden_state: torch.LongTensor,
                page: torch.IntTensor = None):
        logits = self._classifier(hidden_state)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {
            'logits': logits,
            'probs': probs,
            'preds': torch.argmax(logits, 1)
        }

        if page is not None:
            loss = self._loss(logits, page.long().view(-1))
            output_dict['loss'] = loss
            self._accuracy(logits, page)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics


@Model.register('bert_guesser')
class BertGuesser(Guesser):
    def __init__(self,
                 vocab: Vocabulary,
                 dropout: float,
                 pool: Text = "cls",
                 label_namespace: str = "page_labels"):
        super().__init__(vocab, dropout=dropout, label_namespace=label_namespace)
        self._pool = pool
        self._bert = PretrainedBertEmbedder('bert-base-uncased', requires_grad=True)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                metadata = None,
                page: torch.IntTensor = None):
        input_ids: torch.LongTensor = text['text']
        # Grab the representation of CLS token, which is always first
        if self._pool == 'cls':
            bert_emb = self._bert(input_ids)[:, 0, :]
        elif self._pool == 'mean':
            mask = (input_ids != 0).long()[:, :, None]
            bert_seq_emb = self._bert(input_ids)
            bert_emb = util.masked_mean(bert_seq_emb, mask, dim=1)
        else:
            raise ValueError('Invalid config')
        return self._hidden_to_output(bert_emb, page)


@Model.register('classic_guesser')
class ClassicGuesser(Guesser):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder, contextualizer,
                 dropout: float,
                 label_namespace: str = "page_labels"):
        super().__init__(vocab, dropout=dropout, label_namespace=label_namespace)
        self._encoder = encoder
        self._contextualizer = contextualizer

    def forward(self,
                text: Dict[str, torch.LongTensor],
                metadata = None,
                page: torch.IntTensor = None):
        tokens: torch.LongTensor = text['tokens']
        mask: torch.LongTensor = text['mask']
        embeddings = self._encoder(tokens, mask)
        hidden_state = self._hidden_to_output(embeddings, mask)
        return self._hidden_to_output(hidden_state, page)
