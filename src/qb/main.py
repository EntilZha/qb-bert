import os
import glob

import click
import mlflow

from qb.util import get_logger
from qb.config import Config
from qb.hyper import hyper_cli


log = get_logger(__name__)


@click.group()
def cli():
    pass


cli.add_command(hyper_cli)


@cli.command(name='train')
@click.option('--trial', default=0, type=int)
@click.argument('config_path')
def cli_train(trial, config_path):
    config = Config(config_path, trial=trial)
    log.info('Training model')
    with mlflow.start_run():
        config.train()
        config.complete()


@cli.command(name='check')
@click.option('--trials', default=3, type=int)
def cli_check(trials):
    log.info('Checking for missing models')
    for conf_paths in glob.glob('config/generated/**/*.toml'):
        name = os.path.basename(conf_paths).split('.toml')[0]
        for n in range(trials):
            model_complete = os.path.join('models', name, str(n), 'COMPLETE')
            if not os.path.exists(model_complete):
                log.info(f'Missing: {model_complete}')
            report = os.path.join('models', name, str(n), 'report.json')
            if not os.path.exists(report):
                log.info(f'Missing: {report}')


if __name__ == '__main__':
    cli()
