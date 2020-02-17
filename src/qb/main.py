import os
import subprocess
import glob

import click
import toml
import comet_ml
from allennlp.commands import train

from qb.util import get_logger
from qb.config import Config
from qb.hyper import hyper_cli
from qb.predictor import QbPredictor
from qb.model import Guesser
from qb.data import QantaReader, QANTA_GUESSDEV
# imports for allennlp register
from qb import model
from qb import callbacks
from qb import data


log = get_logger(__name__)


@click.group()
def cli():
    pass


cli.add_command(hyper_cli)


@cli.command(name='train')
@click.argument('config_path')
def cli_train(config_path):
    log.info('Training model')
    with open(config_path) as f:
        conf = toml.load(f)
    log.info(f'Configuration\n{conf}')
    generated_id = conf['generated_id']
    subprocess.run(f'fusermount -u /tmp/{generated_id}', shell=True, check=False)
    subprocess.run(f'rm -rf /tmp/{generated_id}', shell=True, check=True)
    subprocess.run(f'mkdir -p /tmp/{generated_id}', shell=True, check=True)
    subprocess.run(f's3fs entilzha-us-east-1 /tmp/{generated_id}', shell=True, check=True)

    train.train_model_from_file(
        parameter_filename=conf['allennlp_conf'],
        serialization_dir=conf['serialization_dir'],
        file_friendly_logging=True,
        force=True,
    )
    subprocess.run(f'touch {conf["serialization_dir"]}/COMPLETE', shell=True, check=True)
    subprocess.run(f'fusermount -u /tmp/{generated_id}', shell=True, check=True)
    subprocess.run(f'rm -rf /tmp/{generated_id}')


@cli.command(name='evaluate')
@click.argument('config_path')
def cli_evaluate(config_path):
    with open(config_path) as f:
        conf = toml.load(f)
    serialization_dir = conf['serialization_dir']
    predictor = Guesser.from_archive(serialization_dir)
    first_sentece_data = QantaReader(
        'guessdev',
        break_questions=False,
        first_sentence_only=True,
    ).read(QANTA_GUESSDEV)
    full_question_data = QantaReader(
        'guessdev',
        break_questions=False,
    ).read(QANTA_GUESSDEV)


if __name__ == '__main__':
    cli()
