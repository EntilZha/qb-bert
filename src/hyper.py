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
    subprocess.run(f"jsonnet {base_model} {args} > {out_path}", shell=True, check=True)


def clone_src(target_dir: str):
    subprocess.run(f"python setup.py build", shell=True, check=True)
    subprocess.run(f"cp -r build/lib/qb {target_dir}", shell=True, check=True)


def random_experiment_id():
    return str(random.randint(1_000_000, 2_000_000))


def hyper_to_configs(path: str):
    with open(path) as f:
        hyper_conf = toml.load(f)
    configs = []
    n_trials = hyper_conf.get("n_trials", 1)
    if "hyper" in hyper_conf:
        grid = ParameterGrid(hyper_conf["hyper"])
        del hyper_conf["hyper"]
        for params in grid:
            for trial in range(n_trials):
                conf = copy.deepcopy(hyper_conf)
                for name, val in params.items():
                    splits = name.split(".")
                    access = conf
                    for part in splits[:-1]:
                        access = access[part]
                    access[splits[-1]] = val
                conf["trial"] = trial
                configs.append(conf)
        return configs
    else:
        if "hyper" in hyper_conf:
            del hyper_conf["hyper"]
        return [hyper_conf]


@click.command()
@click.option("--slurm-job/--no-slurm-job", is_flag=True, default=True)
@click.argument("hyper_conf_path")
@click.argument("base_json_conf")
@click.argument("name")
def hyper_cli(slurm_job: bool, hyper_conf_path: str, base_json_conf: str, name: str):
    configs = hyper_to_configs(hyper_conf_path)
    for c in configs:
        conf_name = random_experiment_id()
        trial = c["trial"]
        conf_dir = os.path.abspath(os.path.join("config", "generated", name, conf_name, trial))
        allennlp_conf_path = os.path.join(conf_dir, f"{conf_name}.json")
        conf_path = os.path.join(conf_dir, f"{conf_name}.toml")
        serialization_dir = os.path.abspath(
            os.path.join("model", "generated", name, conf_name, trial)
        )
        c["generated_id"] = conf_name
        c["name"] = name
        c["allennlp_conf"] = allennlp_conf_path
        c["serialization_dir"] = serialization_dir
        c["conf_dir"] = conf_dir
        c["conf_path"] = conf_path
        c["trial"] = trial
        os.makedirs(os.path.dirname(conf_path), exist_ok=True)
        os.makedirs(serialization_dir, exist_ok=True)
        with open(conf_path, "w") as f:
            toml.dump(c, f)
        args = []
        for key, val in c["params"].items():
            if isinstance(val, str):
                args.append(f"--tla-str {key}={val}")
            else:
                args.append(f"--tla-code {key}={val}")
        args = " ".join(args)
        run_jsonnet(base_json_conf, args, allennlp_conf_path)
        clone_src(conf_dir)

    with open(f"{name}-jobs.sh", "w") as f:
        for c in configs:
            conf_dir = c["conf_dir"]
            conf_path = c["conf_path"]
            if "slurm" in c:
                slurm_time = c["slurm"].get("time", "4-00:00:00")
                slurm_qos = c["slurm"].get("qos", "gpu-long")
            else:
                slurm_time = "4-00:00:00"
                slurm_qos = "gpu-long"

            if slurm_job:
                args = [
                    "sbatch",
                    "--qos",
                    slurm_qos,
                    "--time",
                    slurm_time,
                    "slurm-allennlp.sh",
                    conf_dir,
                    conf_path,
                ]
                f.write(" ".join(args) + "\n")
            else:
                f.write(f"bash train.sh {conf_dir} {conf_path}\n")

    with open(f"{name}-scav-jobs.sh", "w") as f:
        for c in configs:
            conf_path = c["conf_path"]
            args = [
                "sbatch",
                "slurm-scav-allennlp.sh",
                conf_dir,
                conf_path,
            ]
            f.write(" ".join(args) + "\n")


if __name__ == "__main__":
    hyper_cli()
