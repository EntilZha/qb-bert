from typing import Dict, Text

import torch
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders import PretrainedBertEmbedder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, PytorchSeq2VecWrapper
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask


class Guesser(Model):
    def __init__(
        self,
        *,
        vocab: Vocabulary,
        dropout: float,
        hidden_dim: int,
        label_namespace: str = "page_labels"
    ):
        super().__init__(vocab)
        self.top_k = None
        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._hidden_dim = hidden_dim
        self._classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self._num_labels),
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def _hidden_to_output(
        self,
        # (batch_size, hidden_size)
        hidden_state: torch.LongTensor,
        page: torch.IntTensor = None,
    ):
        # (batch_size, n_classes)
        logits = self._classifier(hidden_state)
        # (batch_size, n_classes)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs, "preds": torch.argmax(logits, 1)}

        if page is not None:
            loss = self._loss(logits, page.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, page)
        if self.top_k is not None:
            output_dict["top_k_scores"], output_dict["top_k_indices"] = torch.topk(
                probs, self.top_k, dim=-1
            )

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics


@Model.register("bert_guesser")
class BertGuesser(Guesser):
    def __init__(
        self,
        vocab: Vocabulary,
        dropout: float,
        pool: Text = "cls",
        label_namespace: str = "page_labels",
    ):
        bert = PretrainedBertEmbedder("bert-base-uncased", requires_grad=True)
        super().__init__(
            vocab=vocab,
            dropout=dropout,
            label_namespace=label_namespace,
            hidden_dim=bert.get_output_dim(),
        )
        self._bert = bert
        self._pool = pool

    def forward(
        self, text: Dict[str, torch.LongTensor], metadata=None, page: torch.IntTensor = None
    ):
        input_ids: torch.LongTensor = text["text"]
        # Grab the representation of CLS token, which is always first
        if self._pool == "cls":
            bert_emb = self._bert(input_ids)[:, 0, :]
        elif self._pool == "mean":
            mask = (input_ids != 0).long()[:, :, None]
            bert_seq_emb = self._bert(input_ids)
            bert_emb = util.masked_mean(bert_seq_emb, mask, dim=1)
        else:
            raise ValueError("Invalid config")
        return self._hidden_to_output(bert_emb, page)


@Model.register("rnn_guesser")
class RnnGuesser(Guesser):
    def __init__(
        self,
        *,
        vocab: Vocabulary,
        dropout: float = 0.25,
        emb_dim: int = 300,
        hidden_dim: int = 1500,
        bidirectional: bool = True,
        n_hidden_layers: int = 1,
        label_namespace: str = "page_labels"
    ):
        contextualizer = PytorchSeq2VecWrapper(
            nn.GRU(
                emb_dim,
                hidden_dim,
                n_hidden_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=bidirectional,
            )
        )
        super().__init__(
            vocab=vocab,
            dropout=dropout,
            label_namespace=label_namespace,
            hidden_dim=contextualizer.get_output_dim(),
        )
        self._emb_dim = emb_dim
        self._dropout = dropout
        self._hidden_dim = hidden_dim
        self._bidirectional = bidirectional
        self._n_hidden_layers = n_hidden_layers

        self._embedder = BasicTextFieldEmbedder(
            {
                "text": Embedding(
                    num_embeddings=vocab.get_vocab_size(),
                    embedding_dim=emb_dim,
                    trainable=True,
                    pretrained_file="https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                )
            }
        )
        self._contextualizer = contextualizer

    def forward(
        self, text: Dict[str, torch.LongTensor], metadata=None, page: torch.IntTensor = None
    ):
        mask = get_text_field_mask(text)
        embeddings = self._embedder(text)
        hidden_state = self._contextualizer(embeddings, mask)
        return self._hidden_to_output(hidden_state, page)


@Model.register("dan_guesser")
class DanGuesser(Guesser):
    def __init__(
        self,
        *,
        vocab: Vocabulary,
        hidden_dim: int = 1000,
        n_hidden_layers: int = 1,
        emb_dim: int = 300,
        dropout: float = 0.5,
        pool: str = "avg",
        label_namespace: str = "page_labels"
    ):
        super().__init__(
            vocab=vocab, dropout=dropout, label_namespace=label_namespace, hidden_dim=hidden_dim
        )
        self._embedder = BasicTextFieldEmbedder(
            {
                "text": Embedding(
                    num_embeddings=vocab.get_vocab_size(),
                    embedding_dim=emb_dim,
                    trainable=True,
                    pretrained_file="https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                )
            }
        )
        self._pool = pool
        if pool == "avg":
            averaged = True
        elif pool == "sum":
            averaged = False
        else:
            raise ValueError("Invalid value for pool type")
        self._boe = BagOfEmbeddingsEncoder(emb_dim, averaged=averaged)
        encoder_layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                input_dim = emb_dim
            else:
                input_dim = hidden_dim

            encoder_layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self._encoder = nn.Sequential(*encoder_layers)

    def forward(
        self, text: Dict[str, torch.LongTensor], metadata=None, page: torch.IntTensor = None
    ):
        mask = get_text_field_mask(text)
        embeddings = self._embedder(text)
        pooled_emb = self._boe(embeddings, mask)
        hidden_state = self._encoder(pooled_emb)
        return self._hidden_to_output(hidden_state, page)
