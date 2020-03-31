from typing import Dict, Any, List
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import TokenIndexer, Tokenizer, Instance, Vocabulary

from qb.model import Guesser
from qb.data import QantaReader


log = logging.getLogger(__name__)

Predictor.register("qb_predictor")


class QbPredictor(Predictor):
    def __init__(self, model: Guesser, dataset_reader: QantaReader, top_k: int = 10):
        super().__init__(model, dataset_reader)
        self._top_k = top_k
        model.top_k = top_k

    def _json_to_instance(self, json_dict: JsonDict):
        return self._dataset_reader.text_to_instance(text=json_dict["text"])


def generate_guesses(
    *,
    model: Guesser,
    tokenizer: Tokenizer,
    token_indexers: Dict[str, TokenIndexer],
    max_n_guesses: int,
    fold: str,
    char_skip: int = 25,
    partial_question: bool = False,
    full_question: bool = False,
    first_sentence: bool = False,
    batch_size: int = 128
) -> pd.DataFrame:
    """
    Generates guesses for this guesser for all questions in specified folds and returns it as a DataFrame.

    :param max_n_guesses: generate at most this many guesses per question, sentence, and token
    :param folds: which folds to generate guesses for
    :param char_skip: generate guesses every 10 characters
    :return: dataframe of guesses
    """
    if full_question and first_sentence:
        raise ValueError("Invalid option combination")

    if full_question:
        dataset = QantaReader(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            full_question_only=True,
            first_sentence_only=False,
            char_skip=None,
            include_label=False,
        )
    elif first_sentence:
        dataset = QantaReader(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            full_question_only=False,
            first_sentence_only=True,
            char_skip=None,
            include_label=False,
        )
    elif partial_question:
        dataset = QantaReader(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            full_question_only=False,
            first_sentence_only=False,
            char_skip=char_skip,
            include_label=False,
        )
    else:
        raise ValueError("Invalid combination of arguments")

    predictor = QbPredictor(model, dataset, top_k=max_n_guesses)
    rows = []
    guesser_name = type(model).__name__
    questions = dataset.read(fold)
    idx = 0
    log.info("Making guess predictions")
    bar = tqdm(total=len(questions))
    while True:
        batch_questions = questions[idx : idx + batch_size]
        if len(batch_questions) == 0:
            break
        batch_preds = predictor.predict_batch_instance(batch_questions)
        for q, pred in zip(batch_questions, batch_preds):
            rows.extend(
                prediction_to_rows(
                    fold=fold,
                    guesser_name=guesser_name,
                    vocab=model.vocab,
                    question=q,
                    prediction=pred,
                )
            )

        idx += batch_size
        bar.update(len(batch_questions))
    bar.close()

    return pd.DataFrame(rows)


def prediction_to_rows(
    *, fold: str, guesser_name: str, vocab: Vocabulary, question: Instance, prediction
) -> List[Dict[str, Any]]:
    top_scores = prediction["top_k_scores"]
    top_indices = prediction["top_k_indices"]
    meta = question["metadata"]
    rows = []
    for score, guess_idx in zip(top_scores, top_indices):
        guess = vocab.get_token_from_index(guess_idx, namespace="page_labels")
        rows.append(
            {
                "qanta_id": meta["qanta_id"],
                "proto_id": meta["proto_id"],
                "char_index": meta["char_idx"],
                "guess": guess,
                "score": score,
                "fold": fold,
                "guesser": guesser_name,
            }
        )
    return rows
