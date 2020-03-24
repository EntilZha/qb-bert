import glob
import re
import click
import comet_ml
import tqdm


FIRST_RE = re.compile(r"first dev ([0-9\.]+)")
FULL_RE = re.compile(r"full dev ([0-9\.]+)")
EXP_RE = re.compile(r".+url: https.+qb\-bert\/([a-zA-Z0-9]+)")


@click.command()
@click.argument("log_dir")
def main(log_dir: str):
    for path in tqdm.tqdm(glob.glob(log_dir)):
        experiment_id = None
        first_dev = None
        full_dev = None
        with open(path) as f:
            for line in f:
                maybe_first = re.match(FIRST_RE, line)
                if maybe_first is not None:
                    first_dev = float(maybe_first.group(1))

                maybe_full = re.match(FULL_RE, line)
                if maybe_full is not None:
                    full_dev = float(maybe_full.group(1))

                maybe_id = re.match(EXP_RE, line)
                if maybe_id is not None:
                    experiment_id = maybe_id.group(1)

        if experiment_id is not None and first_dev is not None and full_dev is not None:
            print(experiment_id, first_dev, full_dev)
            experiment = comet_ml.ExistingExperiment(previous_experiment=experiment_id)
            experiment.log_metric("dev_first_accuracy", first_dev)
            experiment.log_metric("dev_full_accuracy", full_dev)
            experiment.end()


if __name__ == "__main__":
    main()
