import logging
import argparse
import os
from typing import Optional, List

import typer
import toml
import comet_ml  # pylint: disable=unused-import
from allennlp.commands import train
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from pedroai.io import read_json

from qb.util import shell
from qb.evaluate import score_model, guess_df_path, create_guesser_report
from qb.predictor import generate_guesses
from qb import constants

# imports for allennlp register
from qb import model  # pylint: disable=unused-import
from qb import callbacks  # pylint: disable=unused-import
from qb import data  # pylint: disable=unused-import


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

app = typer.Typer()




@app.command("train")
def cli_train(config_path: str, log_to_comet: bool = False):
    log.info("log_to_comet: %s", log_to_comet)
    log.info("config_path: %s", config_path)
    log.info("Training model")
    with open(config_path) as f:
        conf = toml.load(f)

    log.info("Configuration\n%s", conf)
    train.train_model_from_file(
        parameter_filename=conf["allennlp_conf"],
        serialization_dir=conf["serialization_dir"],
        file_friendly_logging=True,
        force=True,
    )
    score_model(conf["serialization_dir"], log_to_comet=log_to_comet)
    shell(f'touch {conf["serialization_dir"]}/COMPLETE')


@app.command("evaluate")
def cli_evaluate(config_path: str, log_to_comet: bool = False, comet_experiment_id: Optional[str] = None):
    with open(config_path) as f:
        conf = toml.load(f)
    serialization_dir = conf["serialization_dir"]
    score_model(
        serialization_dir, log_to_comet=log_to_comet, comet_experiment_id=comet_experiment_id
    )


@app.command("generate_guesses")
def cli_generate_guesses(
    config_path: str,
    granularity: List[str] = [],
    char_skip: int = 25,
    max_n_guesses: int = 10,
    trickme_path: str = None,
    generation_fold: List[str] = constants.GENERATION_FOLDS,
):
    with open(config_path) as f:
        conf = toml.load(f)
    serialization_dir = conf["serialization_dir"]
    log.info("Loading model from: %s", serialization_dir)
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"), cuda_device=0)
    predictor = Predictor.from_archive(archive, "qb.predictor.QbPredictor")
    # pylint: disable=protected-access
    dataset_reader = predictor._dataset_reader
    tokenizer = dataset_reader._tokenizer
    token_indexers = dataset_reader._token_indexers
    for g in granularity:
        if g == "first":
            first_sentence = True
            full_question = False
            partial_question = False
        elif g == "full":
            first_sentence = False
            full_question = True
            partial_question = False
        elif g == "char":
            first_sentence = False
            full_question = False
            partial_question = True
        else:
            raise ValueError("Invalid granularity")

        log.info("Generating guesses for: %s", generation_fold)
        for fold in generation_fold:
            log.info("Guesses for fold %s", fold)
            df = generate_guesses(
                model=archive.model,
                tokenizer=tokenizer,
                token_indexers=token_indexers,
                max_n_guesses=max_n_guesses,
                fold=fold,
                first_sentence=first_sentence,
                full_question=full_question,
                partial_question=partial_question,
                char_skip=char_skip,
                trickme_path=trickme_path,
            )
            path = os.path.join(serialization_dir, guess_df_path(g, fold))
            df.to_pickle(path)


@app.command("predict")
def cli_predict(granularity: str, fold: str, config_path: str):
    with open(config_path) as f:
        conf = toml.load(f)
    serialization_dir = conf["serialization_dir"]
    log.info("Loading model from: %s", serialization_dir)
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"), cuda_device=0)
    predictor = Predictor.from_archive(archive, "qb.predictor.QbPredictor")
    # pylint: disable=protected-access
    dataset_reader = predictor._dataset_reader
    tokenizer = dataset_reader._tokenizer
    token_indexers = dataset_reader._token_indexers
    


@app.command("generate_report")
def cli_generate_report(mapped_qanta_path: str, config_path: str):
    create_guesser_report(mapped_qanta_path, config_path)


@app.command("to_fasttext")
def cli_to_fasttext(mapped_qanta_path: str, output_dir: str):
    questions = [q for q in read_json(mapped_qanta_path)["questions"] if q["page"] is not None]
    train = [q for q in questions if q["fold"] == "guesstrain"]
    dev = [q for q in questions if q["fold"] == "guessdev"]
    test = [q for q in questions if q["fold"] == "guesstest"]
    questions_by_fold = [("train", train), ("dev", dev), ("test", test)]
    for fold, examples in questions_by_fold:
        start_file = open(os.path.join(output_dir, f"qanta.{fold}.start.txt"), "w")
        end_file = open(os.path.join(output_dir, f"qanta.{fold}.end.txt"), "w")
        sent_file = open(os.path.join(output_dir, f"qanta.{fold}.sent.txt"), "w")
        for q in examples:
            page = q["page"]
            for sent_start, sent_end in q["tokenizations"]:
                sent_text = q["text"][sent_start:sent_end].lower()
                if len(sent_text) > 4:
                    sent_file.write(f"__label__{page} {sent_text}\n")

            start_text = q["first_sentence"].lower()
            end_text = q["text"].lower()
            start_file.write(f"__label__{page} {start_text}\n")
            end_file.write(f"__label__{page} {end_text}\n")

        start_file.close()
        end_file.close()
        sent_file.close()


if __name__ == "__main__":
    app()
