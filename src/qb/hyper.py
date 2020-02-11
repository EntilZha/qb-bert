import os
import uuid
import copy
import subprocess
import toml
import click
import toml
from sklearn.model_selection import ParameterGrid


def run_jsonnet(base_model, args, out_path):
    subprocess.run(
        f'jsonnet {base_model} {args} > {out_path}',
        shell=True, check=True
    )


def hyper_to_configs(path):
    with open(path) as f:
        hyper_conf = toml.load(f)
    configs = []
    if 'hyper' in hyper_conf:
        grid = ParameterGrid(hyper_conf['hyper'])
        del hyper_conf['hyper']
        for params in grid:
            conf = copy.deepcopy(hyper_conf)
            for name, val in params.items():
                splits = name.split('.')
                access = conf
                for part in splits[:-1]:
                    access = access[part]
                access[splits[-1]] = val
            configs.append(conf)
        return configs
    else:
        del hyper_conf['hyper']
        return [hyper_conf]


@click.command(name='hyper')
@click.option('--dry-run', is_flag=True, default=False)
@click.option('--n-trials', type=int, default=3)
@click.option('--slurm-job/--no-slurm-job', is_flag=True, default=True)
@click.argument('hyper_conf_path')
@click.argument('base_json_conf')
@click.argument('name')
def hyper_cli(dry_run, n_trials, slurm_job, hyper_conf_path, base_json_conf, name):
    if dry_run:
        print('Running in dry run mode')
    configs = hyper_to_configs(hyper_conf_path)
    conf_paths = []
    model_paths = []
    for c in configs:
        conf_name = f'generated-{name}-{uuid.uuid4()}'
        file_path = os.path.join('config', 'generated', name, f'{conf_name}.json')
        target_path = os.path.join('model', 'generated', name, conf_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        args = []
        for key, val in c['params'].items():
            args.append(f'--tla-code {key}={val}')
        args = ' '.join(args)
        run_jsonnet(base_json_conf, args, file_path)

        conf_paths.append(file_path)
        model_paths.append(target_path)

    with open(f'{name}-jobs.sh', 'w') as f:
        for m_path, c_path in zip(model_paths, conf_paths):
            if slurm_job:
                f.write(f'sbatch slurm-allennlp.sh -s {m_path} {c_path}\n')
            else:
                f.write(f'allennlp train --include-package qb -f -s {m_path} {c_path}\n')
