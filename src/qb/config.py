import os
import pathlib
import logging
import pprint
import subprocess

import toml
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

from qb.util import get_logger


LOG_CONFIGURED = [False]


log = get_logger(__name__)


def train_allennlp(serialization_dir, config_path):
    subprocess.run(
        f'allennlp train --include-package qb -f -s {serialization_dir} {config_path}',
        shell=True, check=True
    )


class Config:
    def __init__(self, config_path, trial=None):
        if trial is None:
            trial = 0
        self._config_path = config_path
        with open(config_path) as f:
            self._config = toml.load(f)
        self._validate()

        if 'name' in self._config and 'trial' in self._config and 'path' in self._config:
            self._config_name = self._config['name']
            self.trial = self._config['trial']
            path = self._config['path']
            self._path = self._config['path']
            dir_existed = True
            config_existed = True
        else:
            self.trial = trial
            self._config_name = os.path.splitext(os.path.basename(config_path))[0]

            path = f'models/{self._config_name}/{self.trial}/'
            self._path = path
            dir_existed = os.path.isdir(path)
            config_existed = False
            if not dir_existed:
                os.makedirs(path, exist_ok=True)

        if not LOG_CONFIGURED[0]:
            LOG_CONFIGURED[0] = True
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler = logging.FileHandler(f'{path}qb-bert.log')
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            for log_name in logging.Logger.manager.loggerDict:
                if '.' not in log_name:
                    logger = logging.getLogger(log_name)
                    logger.addHandler(handler)

        if dir_existed:
            log.info(f'Directory for config {config_path} exists at "{path}"')
        else:
            log.info(f'Creating directory for config {config_path} at "{path}"')

        if not config_existed:
            self._config['name'] = self._config_name
            self._config['trial'] = self.trial
            self._config['path'] = self._path
            with open(f'{path}config.toml', 'w') as f:
                toml.dump(self._config, f)

    def _validate(self):
        if 'model_type' not in self._config or 'predictor' not in self._config:
            raise ValueError('Model type or predictor not specified')
        else:
            if self._config['model_type'] not in ('allennlp', 'ir'):
                raise ValueError('Invalid model type')

    def train(self):
        if self.model_type == 'allennlp':
            train_allennlp(self.path('model/'), self.model_config)
        else:
            raise ValueError('Unexpected model type')

    def load_predictor(self):
        if self.model_type == 'allennlp':
            archive = load_archive(self.path('model.tar.gz'))
            return Predictor.from_archive(archive, self._config['predictor'])
        elif self.model_type == 'ir':
            raise NotImplementedError()
        else:
            raise ValueError('Unexpected model type')

    @property
    def name(self):
        return self._config_name

    def __repr__(self):
        return ("Config\n"
                f"model_name={self.model_name}\n"
                f"model_type={self.model_type}\n"
                f"model_config={self.model_config}\n"
                f"path={self._path}\n"
                f"trial={self.trial}\n")

    def path(self, filename=None):
        if filename is None:
            return self._path
        else:
            return f'{self._path}/{filename}'

    def open(self, filename, mode='r'):
        return open(self.path(filename=filename), mode=mode)

    @property
    def model_name(self):
        return self._config['model_name']

    @property
    def model_config(self):
        return self._config['model_config']

    @property
    def model_type(self):
        return self._config['model_type']

    @property
    def predictor_name(self):
        return self._config['predictor_name']

    @property
    def mlflow_id(self):
        if 'mlflow_id' in self._config:
            return self._config['mlflow_id']
        else:
            return None

    def complete(self):
        pathlib.Path(self.path('COMPLETE')).touch()

    def get_config_dict(self):
        return self._config
