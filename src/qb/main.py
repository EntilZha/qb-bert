import os

import click
from tqdm import tqdm
import toml
import comet_ml  # pylint: disable=unused-import
from allennlp.commands import train
from allennlp.models.archival import load_archive

from qb.util import get_logger, shell
from qb.evaluate import score_model
from qb.predictor import generate_guesses

# imports for allennlp register
from qb import model  # pylint: disable=unused-import
from qb import callbacks  # pylint: disable=unused-import
from qb import data  # pylint: disable=unused-import


log = get_logger(__name__)


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
def cli_evaluate(config_path):
    with open(config_path) as f:
        conf = toml.load(f)
    serialization_dir = conf["serialization_dir"]
    score_model(serialization_dir)


@cli.command(name="generate_guesses")
@click.option("--max-n-guesses", type=int, default=10)
@click.argument("config_path")
@click.argument("granularity")
@click.argument("output_dir")
def cli_generate_guesses(max_n_guesses, config_path, granularity, output_dir):
    if granularity not in {"char", "full", "first"}:
        raise ValueError("Invalid granularity")
    with open(config_path) as f:
        conf = toml.load(f)
    serialization_dir = conf["serialization_dir"]
    log.info("Loading model from: %s", serialization_dir)
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"), cuda_device=0)
    generation_folds = ["guessdev", "guesstest", "buzztrain", "buzzdev", "buzztest"]
    log.info("Generating guesses")
    for fold in tqdm(generation_folds):
        df = generate_guesses(model=archive.model, max_n_guesses=max_n_guesses, fold=fold)
        path = os.path.join(output_dir, f"guesses_{granularity}_{fold}.pickle")
        df.to_pickle(path)


if __name__ == "__main__":
    cli()
