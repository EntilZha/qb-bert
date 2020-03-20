import os
import random
import glob
import copy
import subprocess
import toml
import click
import toml
from sklearn.model_selection import ParameterGrid


def run_jsonnet(base_model: str, args: str, out_path: str):
    subprocess.run(
        f'jsonnet {base_model} {args} > {out_path}',
        shell=True, check=True
    )


def clone_src(target_dir: str):
    subprocess.run(
        f'python setup.py build', 
        shell=True, check=True
    )
    subprocess.run(f'cp -r build/lib/qb {target_dir}', shell=True, check=True)


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
        if 'hyper' in hyper_conf:
            del hyper_conf['hyper']
        return [hyper_conf]


def random_experiment_id():
    return str(random.randint(1_000_000, 2_000_000))


@click.group()
def cli():
    pass


@cli.command(name='generate')
@click.option('--n-trials', type=int, default=1)
@click.option('--slurm-job/--no-slurm-job', is_flag=True, default=True)
@click.argument('hyper_conf_path')
@click.argument('base_json_conf')
@click.argument('name')
def hyper_cli(n_trials, slurm_job, hyper_conf_path, base_json_conf, name):
    configs = hyper_to_configs(hyper_conf_path)
    conf_paths = []
    directories = []
    for c in configs:
        conf_name = random_experiment_id()
        conf_dir = os.path.abspath(
            os.path.join('config', 'generated', name, conf_name)
        )
        allennlp_conf_path = os.path.join(
            conf_dir, f'{conf_name}.json'
        )
        conf_path = os.path.join(
            conf_dir, f'{conf_name}.toml'
        )
        serialization_dir = os.path.abspath(os.path.join('model', 'generated', name, conf_name))
        c['generated_id'] = conf_name
        c['name'] = name
        c['allennlp_conf'] = allennlp_conf_path
        c['serialization_dir'] = serialization_dir
        c['conf_dir'] = conf_dir
        os.makedirs(os.path.dirname(conf_path), exist_ok=True)
        os.makedirs(serialization_dir, exist_ok=True)
        with open(conf_path, 'w') as f:
            toml.dump(c, f)
        args = []
        for key, val in c['params'].items():
            if isinstance(val, str):
                args.append(f'--tla-str {key}={val}')
            else:
                args.append(f'--tla-code {key}={val}')
        args = ' '.join(args)
        run_jsonnet(base_json_conf, args, allennlp_conf_path)
        clone_src(conf_dir)
        directories.append(conf_dir)
        conf_paths.append(conf_path)

    with open(f'{name}-jobs.sh', 'w') as f:
        for conf_dir, c_path in zip(directories, conf_paths):
            if slurm_job:
                f.write(f'sbatch slurm-allennlp.sh {conf_dir} {c_path}\n')
            else:
                f.write(f'python qb/main.py train {c_path}\n')

    with open(f'{name}-scav-jobs.sh', 'w') as f:
        for c_path in conf_paths:
            f.write(f'sbatch slurm-scav-allennlp.sh {conf_dir} {c_path}\n')


@cli.command(name='train')
def train_cli():
    pass


if __name__ == '__main__':
    hyper_cli()