from typing import Text, List

import pandas as pd

from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict

from qb.model import Guesser
from qb.data import QantaReader
from qb import util


log = util.get_logger(__name__)


Predictor.register('qb_predictor')
class QbPredictor(Predictor):
    def __init__(self, model: Guesser, dataset_reader: QantaReader):
        super().__init__(model, dataset_reader)
    
    def _json_to_instance(self, json_dict: JsonDict):
        return self._dataset_reader.text_to_instance(
            text=json_dict['text']
        )


def generate_guesses(
    *,
    model: Guesser,
    max_n_guesses: int,
    folds: List[Text],
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

    q_folds = []
    q_qnums = []
    q_char_indices = []
    q_proto_ids = []
    question_texts = []

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
    
    predictor = QbPredictor(model, dataset)
    rows = []
    for fold in folds:
        questions = dataset.read(fold)
        guesses = []
        idx = 0
        while True:
            batch = questions[idx:idx + batch_size]
            if len(batch) == 0:
                break
            batch_guesses = predictor.predict_batch_instance(batch)
            guesses.extend(batch_guesses)
            idx += batch_size
        for q, g in zip(questions, guesses):
            rows.append({
                'qanta_id': 0,
                'proto_id': 0,
                'char_index': 0,
                'guess': 0,
                'score': 0,
                'fold': 0,
                'guesser': 0,
            })

    log.info('Creating guess dataframe from guesses...')
    df_qnums = []
    df_proto_id = []
    df_char_indices = []
    df_guesses = []
    df_scores = []
    df_folds = []
    df_guessers = []
    guesser_name = self.display_name()

    for i in range(len(question_texts)):
        guesses_with_scores = guesses_per_question[i]
        fold = q_folds[i]
        qnum = q_qnums[i]
        proto_id = q_proto_ids[i]
        char_ix = q_char_indices[i]
        for guess, score in guesses_with_scores:
            df_qnums.append(qnum)
            df_proto_id.append(proto_id)
            df_char_indices.append(char_ix)
            df_guesses.append(guess)
            df_scores.append(score)
            df_folds.append(fold)
            df_guessers.append(guesser_name)

    return pd.DataFrame({
        'qanta_id': df_qnums,
        'proto_id': df_proto_id,
        'char_index': df_char_indices,
        'guess': df_guesses,
        'score': df_scores,
        'fold': df_folds,
        'guesser': df_guessers
    })
