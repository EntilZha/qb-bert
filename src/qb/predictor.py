from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict
from qb.model import Guesser
from qb.data import QantaReader


Predictor.register('qb_predictor')
class QbPredictor(Predictor):
    def __init__(self, model: Guesser, dataset_reader: QantaReader):
        super().__init__(model, dataset_reader)
    
    def _json_to_instance(self, json_dict: JsonDict):
        return self._dataset_reader.text_to_instance(
            text=json_dict['text']
        )
