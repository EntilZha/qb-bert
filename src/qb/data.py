from typing import Dict, List, Optional, Text, Tuple
import logging
import re

from unidecode import unidecode
import nltk
from overrides import overrides
from tqdm import tqdm

from allennlp.data import DatasetReader, TokenIndexer, Instance
from allennlp.data.fields import TextField, LabelField, Field, MetadataField
from allennlp.data.tokenizers import Tokenizer

from qb import util

log = logging.getLogger(__name__)


qb_patterns = {
    "\n",
    ", for 10 points,",
    ", for ten points,",
    "--for 10 points--",
    "for 10 points, ",
    "for 10 points--",
    "for ten points, ",
    "for 10 points ",
    "for ten points ",
    ", ftp,",
    "ftp,",
    "ftp",
    "(*)",
}
re_pattern = "|".join([re.escape(p) for p in qb_patterns])
re_pattern += r"|\[.*?\]|\(.*?\)"


QANTA_TRAIN = "data/qanta.train.2018.04.18.json"
QANTA_DEV = "data/qanta.dev.2018.04.18.json"
QANTA_TEST = "data/qanta.test.2018.04.18.json"
QANTA_GUESSDEV = "data/qanta.guessdev.2018.04.18.json"
QANTA_MAPPED = "data/qanta.mapped.2018.04.18.json"


def extract_wiki_sentences(*, title: str, text: str, n_sentences: int, replace_title_mentions=""):
    """
    Extracts the first n_paragraphs from the text of a wikipedia page corresponding to the title.
    strip_title_mentions and replace_title_mentions control handling of references to the title in text.
    Oftentimes QA models learn *not* to answer entities mentioned in the question so this helps deal with this
    in the domain adaptation case.

    :param title: title of page
    :param text: text of page
    :param n_paragraphs: number of paragraphs to use
    :param replace_title_mentions: Replace mentions with the provided string token, by default removing them
    :return:
    """
    # Get simplest representation of title and text
    title = unidecode(title).replace("_", " ")
    text = unidecode(text)

    # Split on non-alphanumeric
    title_words = re.split("[^a-zA-Z0-9]", title)
    title_word_pattern = "|".join(re.escape(w.lower()) for w in title_words)

    # Breaking by newline yields paragraphs. Ignore the first since its always just the title
    paragraphs = [p for p in text.split("\n") if len(p) != 0][1:]
    sentences = []
    for p in paragraphs:
        formatted_text = re.sub(title_word_pattern, replace_title_mentions, p, flags=re.IGNORECASE)
        # Cleanup whitespace
        formatted_text = re.sub(r"\s+", " ", formatted_text).strip()

        sentences.extend(nltk.sent_tokenize(formatted_text))

    return sentences[:n_sentences]


def create_char_runs(text: str, char_skip: int) -> List[Tuple[str, int]]:
    """
    Returns runs of the question based on skipping char_skip characters at a time. Also returns the indices used

    q: name this first united states president.
    runs with char_skip=10:
    ['name this ',
        'name this first unit',
        'name this first united state p',
        'name this first united state president.']

    :param char_skip: Number of characters to skip each time
    """
    char_indices = list(range(char_skip, len(text) + char_skip, char_skip))
    return [(text[:idx], idx) for idx in char_indices]


@DatasetReader.register("qanta")
class QantaReader(DatasetReader):
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        full_question_only: bool,
        first_sentence_only: bool,
        char_skip: Optional[int] = None,
        qanta_path: str = QANTA_MAPPED,
        wiki_path: Optional[str] = None,
        n_wiki_sentences: int = 0,
        include_label: bool = True,
        debug: bool = False,
        lazy: bool = False
    ):
        super().__init__(lazy)
        self.full_question_only = full_question_only
        self.first_sentence_only = first_sentence_only
        self.char_skip = char_skip
        self.n_wiki_sentences = n_wiki_sentences
        self.include_label = include_label
        self.qanta_path = qanta_path
        self.wiki_path = wiki_path

        self._debug = debug
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

    @overrides
    def _read(self, fold):  # pylint: disable=arguments-differ
        log.info("Reading instances from fold=%s, file_path=%s", fold, self.qanta_path)
        questions = util.read_json(self.qanta_path)["questions"]
        max_examples = 256 if self._debug else None
        questions = questions[:max_examples]

        answer_set = set()
        for q in tqdm(questions):
            if q["page"] is not None and q["fold"] == fold:
                proto_id = q.get("proto_id")
                answer_set.add(q["page"])
                if self.full_question_only:
                    yield self.text_to_instance(
                        q["text"],
                        page=q["page"],
                        qanta_id=q["qanta_id"],
                        proto_id=proto_id,
                        source="qb",
                    )
                elif self.first_sentence_only:
                    start, end = q["tokenizations"][0]
                    sentence = q["text"][start:end]
                    yield self.text_to_instance(
                        sentence,
                        page=q["page"],
                        qanta_id=q["qanta_id"],
                        proto_id=proto_id,
                        source="qb",
                    )
                elif self.char_skip is not None:
                    text = q["text"]
                    for text_run, char_idx in create_char_runs(text, self.char_skip):
                        yield self.text_to_instance(
                            text_run,
                            page=q["page"],
                            qanta_id=q["qanta_id"],
                            proto_id=proto_id,
                            source="qb",
                            char_idx=char_idx,
                        )
                else:
                    for start, end in q["tokenizations"]:
                        sentence = q["text"][start:end]
                        yield self.text_to_instance(
                            sentence,
                            page=q["page"],
                            qanta_id=q["qanta_id"],
                            proto_id=proto_id,
                            source="qb",
                        )
        if fold == "guesstrain" and self.n_wiki_sentences > 0:
            wiki_lookup = util.read_json(self.wiki_path)
            pages_with_text = [(p, wiki_lookup[p]["text"]) for p in answer_set if p in wiki_lookup]
            for page, text in pages_with_text.items():
                sentences = extract_wiki_sentences(
                    title=page, text=text, n_sentences=self.n_wiki_sentences,
                )
                for sent in sentences:
                    yield self.text_to_instance(
                        sent, page=page, source="wiki",
                    )

    @overrides
    def text_to_instance(
        self,
        text: str,
        *,
        page: Optional[str] = None,
        char_idx: Optional[int] = None,
        source: Optional[Text] = None,
        proto_id: Optional[Text] = None,
        qanta_id: Optional[int] = None
    ):  # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        fields["text"] = TextField(tokenized_text, token_indexers=self._token_indexers)
        if page is not None and self.include_label:
            fields["page"] = LabelField(page, label_namespace="page_labels")
        fields["metadata"] = MetadataField(
            {
                "qanta_id": qanta_id,
                "proto_id": proto_id,
                "tokens": tokenized_text,
                "page": page,
                "source": source,
                "char_idx": char_idx,
            }
        )
        return Instance(fields)
