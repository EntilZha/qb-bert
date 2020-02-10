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
                 parallel: bool = False,
                 lazy: bool = False):
        super().__init__(lazy)
        self._fold = fold
        self._parallel = parallel
        self._break_questions = break_questions
        self._tokenizer = PretrainedTransformerTokenizer(
            'bert-base-uncased', do_lowercase=True
        )
        self._token_indexers = {'text': PretrainedBertIndexer('bert-base-uncased')}

    @overrides
    def _read(self, file_path):
        log.info(f"Reading instances from: {file_path}")
        if self._parallel:
            return [inst for inst in self._read_parallel(file_path)]
        else:
            return [inst for inst in self._read_serial(file_path)]

    def _read_serial(self, file_path):
        with open(file_path) as f:
            for q in tqdm(json.load(f)['questions']):
                if q['page'] is not None and q['fold'] == self._fold:
                    if self._break_questions:
                        for start, end in q['tokenizations']:
                            sentence = q['text'][start:end]
                            yield self.text_to_instance(sentence, q['page'], q['qanta_id'])
                    else:
                        yield self.text_to_instance(q['text'], q['page'], q['qanta_id'])

    def _read_parallel(self, file_path):
        texts = []
        pages = []
        qids = []
        with open(file_path) as f:
            for q in tqdm(json.load(f)['questions']):
                if q['page'] is not None and q['fold'] == self._fold:
                    if self._break_questions:
                        for start, end in q['tokenizations']:
                            texts.append(q['text'][start:end])
                            pages.append(q['page'])
                            qids.append(q['qanta_id'])
                    else:
                        texts.append(q['text'])
                        pages.append(q['page'])
                        qids.append(q['qanta_id'])

        log.info('Starting parallel tokenization')
        tokenized_texts = self._tokenizer.batch_tokenize(texts)
        log.info('Tokenization complete')
        for i in range(len(tokenized_texts)):
            text = tokenized_texts[i]
            fields: Dict[str, Field] = {}
            fields['text'] = TextField(text, token_indexers=self._token_indexers)
            fields['page'] = LabelField(pages[i], label_namespace='page_labels')
            fields['metadata'] = MetadataField({'qanta_id': qids[i], 'domain': 'qb'})
            yield Instance(fields)

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
        fields['metadata'] = MetadataField({'qanta_id': qanta_id})
        return Instance(fields)

