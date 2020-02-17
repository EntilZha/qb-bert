from typing import Dict, List
import json

from overrides import overrides
from tqdm import tqdm

from allennlp.data import DatasetReader, TokenIndexer, Instance
from allennlp.data.fields import TextField, LabelField, Field, MetadataField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer

from qb.util import get_logger


log = get_logger(__name__)

QANTA_TRAIN = 'data/qanta.train.2018.04.18.json'
QANTA_DEV = 'data/qanta.dev.2018.04.18.json'
QANTA_TEST = 'data/qanta.test.2018.04.18.json'
QANTA_MAPPED = 'data/qanta.mapped.2018.04.18.json'

QANTA_GUESSTRAIN = 'data/qanta.guesstrain-90.2018.04.18.json'
QANTA_GUESSVAL = 'data/qanta.guesstrain-10.2018.04.18.json'
QANTA_GUESSDEV = 'data/qanta.guessdev.2018.04.18.json'
# Based on 99% of Bert token lengths
MAX_SENT_LENGTH = 64


@DatasetReader.register('qanta')
class QantaReader(DatasetReader):
    def __init__(self,
                 fold: str,
                 break_questions: bool,
                 first_sentence_only: bool = False,
                 debug: bool = False,
                 lazy: bool = False):
        super().__init__(lazy)
        self._fold = fold
        self._debug = debug
        self._break_questions = break_questions
        self._first_sentence_only = first_sentence_only
        self._tokenizer = PretrainedTransformerTokenizer(
            'bert-base-uncased', do_lowercase=True,
            start_tokens=[], end_tokens=[]
        )
        self._token_indexers = {'text': PretrainedBertIndexer('bert-base-uncased')}

    @overrides
    def _read(self, file_path):
        log.info(f"Reading instances from: {file_path}")
        return [inst for inst in self._read_serial(file_path)]

    def _read_serial(self, file_path):
        if self._debug:
            max_examples = 256
        else:
            max_examples = None
        with open(file_path) as f:
            for q in tqdm(json.load(f)['questions'][:max_examples]):
                if q['page'] is not None and q['fold'] == self._fold:
                    if self._break_questions:
                        for start, end in q['tokenizations']:
                            sentence = q['text'][start:end]
                            yield self.text_to_instance(sentence, q['page'], q['qanta_id'])
                    else:
                        yield self.text_to_instance(q['text'], q['page'], q['qanta_id'])

    @overrides
    def text_to_instance(self,
                         text: str,
                         page: str = None,
                         qanta_id: int = None):
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        if page is not None:
            fields['page'] = LabelField(page, label_namespace='page_labels')
        fields['metadata'] = MetadataField({
            'qanta_id': qanta_id,
            'tokens': tokenized_text
        })
        return Instance(fields)
