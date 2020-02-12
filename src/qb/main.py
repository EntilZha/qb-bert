import os
import glob

import click
import toml
import comet_ml
from allennlp.commands import train

from qb.util import get_logger
from qb.config import Config
from qb.hyper import hyper_cli
# imports for allennlp register
from qb import model
from qb import callbacks
from qb import data
# end register hook imports


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
    
    train.train_model_from_file(
        parameter_filename=conf['allennlp_conf'],
        serialization_dir=conf['serialization_dir'],
        file_friendly_logging=True,
        force=True,
    )


if __name__ == '__main__':
    cli()
