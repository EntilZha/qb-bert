from typing import Text, List

import pandas as pd
from tqdm import tqdm

from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict

from qb.model import Guesser
from qb.data import QantaReader
from qb import util


log = util.get_logger(__name__)


Predictor.register('qb_predictor')
class QbPredictor(Predictor):
    def __init__(self,
                 model: Guesser,
                 dataset_reader: QantaReader,
                 top_k: int = 10):
        super().__init__(model, dataset_reader)
        self._top_k = top_k
        model.top_k = top_k
    
    def _json_to_instance(self, json_dict: JsonDict):
        return self._dataset_reader.text_to_instance(
            text=json_dict['text']
        )


def generate_guesses(
    *,
    model: Guesser,
    max_n_guesses: int,
    fold: Text,
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
        raise ValueError('Invalid option combination')

    if full_question:
        dataset = QantaReader(
            tokenizer=None, token_indexers=None,
            full_question_only=True,
            first_sentence_only=False,
            char_skip=None,
        )
    elif first_sentence:
        dataset = QantaReader(
            tokenizer=None, token_indexers=None,
            full_question_only=False,
            first_sentence_only=True,
            char_skip=None,
    )
    elif partial_question:
        dataset = QantaReader(
            tokenizer=None, token_indexers=None,
            full_question_only=False,
            first_sentence_only=False,
            char_skip=char_skip,
        )
    else:
        raise ValueError('Invalid combination of arguments')
    
    predictor = QbPredictor(
        model,
        dataset,
        top_k=max_n_guesses
    )
    rows = []
    guesser_name = type(model)
    questions = dataset.read(fold)
    predictions = []
    idx = 0
    while True:
        batch = questions[idx:idx + batch_size]
        if len(batch) == 0:
            break
        predictions.extend(predictor.predict_batch_instance(batch))
        idx += batch_size
    for q, pred in tqdm(zip(questions, predictions)):
        top_scores = pred['top_k_scores']
        top_indices = pred['top_k_indices']
        meta = q['metadata']
        for score, guess_idx in zip(top_scores, top_indices):
            guess = model.vocab.get_token_from_index(
                guess_idx,
                namespace='page_labels'
            )
            rows.append({
                'qanta_id': meta['qanta_id'],
                'proto_id': meta['proto_id'],
                'char_index': meta['char_idx'],
                'guess': guess,
                'score': score,
                'fold': fold,
                'guesser': guesser_name,
            })

    return pd.DataFrame(rows)
