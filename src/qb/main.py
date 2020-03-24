import logging
import os

import click
import toml
import comet_ml  # pylint: disable=unused-import
from allennlp.commands import train
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from qb.util import shell
from qb.evaluate import score_model
from qb.predictor import generate_guesses

# imports for allennlp register
from qb import model  # pylint: disable=unused-import
from qb import callbacks  # pylint: disable=unused-import
from qb import data  # pylint: disable=unused-import


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option("--log_to_comet", default=True, type=bool)
@click.argument("config_path")
def cli_train(log_to_comet: bool, config_path: str):
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


@cli.command(name="evaluate")
@click.argument("config_path")
def cli_evaluate(config_path: str):
    with open(config_path) as f:
        conf = toml.load(f)
    serialization_dir = conf["serialization_dir"]
    score_model(serialization_dir)


@cli.command(name="generate_guesses")
@click.option("--char-skip", type=int, default=25)
@click.option("--max-n-guesses", type=int, default=10)
@click.argument("config_path")
@click.argument("granularity")
@click.argument("output_dir")
def cli_generate_guesses(
    char_skip: int, max_n_guesses: int, config_path: str, granularity: str, output_dir: str
):
    if granularity == "first":
        first_sentence = True
        full_question = False
        partial_question = False
    elif granularity == "full":
        first_sentence = False
        full_question = True
        partial_question = False
    elif granularity == "char":
        first_sentence = False
        full_question = False
        partial_question = True
    else:
        raise ValueError("Invalid granularity")

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
    generation_folds = ["guessdev", "guesstest", "buzztrain", "buzzdev", "buzztest"]
    log.info("Generating guesses")
    for fold in generation_folds:
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
        )
        path = os.path.join(output_dir, f"guesses_{granularity}_{fold}.pickle")
        df.to_pickle(path)


if __name__ == "__main__":
    cli()
