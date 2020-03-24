import subprocess
import json
import os


def shell(command, check=True):
    return subprocess.run(command, check=check, shell=True)


def safe_path(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def read_json(path):
    with open(path) as f:
        return json.load(f)
