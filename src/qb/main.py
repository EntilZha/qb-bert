import os
import subprocess
import glob

import numpy as np
import click
import toml
import comet_ml
import tqdm
from allennlp.commands import train
from allennlp.models.archival import load_archive

from qb.util import get_logger
from qb.config import Config
from qb.hyper import hyper_cli
from qb.predictor import QbPredictor
from qb.model import Guesser
from qb.data import QantaReader, QANTA_GUESSDEV, QANTA_TEST
from qb.evaluate import compute_accuracy
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
    archive = load_archive(
        os.path.join(serialization_dir, 'model.tar.gz'),
        cuda_device=0
    )
    predictor = QbPredictor.from_archive(
        archive,
        predictor_name='qb.predictor.QbPredictor',
    )
    dev_first_sentence = QantaReader(
        'guessdev',
        break_questions=False,
        first_sentence_only=True,
        include_label=False,
    ).read(QANTA_GUESSDEV)
    dev_full_question = QantaReader(
        'guessdev',
        break_questions=False,
        include_label=False,
    ).read(QANTA_GUESSDEV)
    test_first_sentence = QantaReader(
        'guesstest',
        break_questions=False,
        first_sentence_only=True,
        include_label=False,
    ).read(QANTA_TEST)
    test_full_question = QantaReader(
        'guesstest',
        break_questions=False,
        include_label=False,
    ).read(QANTA_TEST)
    print('first', 'dev', compute_accuracy(predictor, dev_first_sentence))
    print('full', 'dev', compute_accuracy(predictor, dev_full_question))
    print('first', 'test', compute_accuracy(predictor, test_first_sentence))
    print('full', 'test', compute_accuracy(predictor, test_full_question))


if __name__ == '__main__':
    cli()
