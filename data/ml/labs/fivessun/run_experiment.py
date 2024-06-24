from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import pathlib

import dotenv
from loguru import logger

from data.etl.python.core import args
from data.pylib import command_utils
from data.pylib import strenum

_PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent


def _parse_comma_separated_str(input_str: str) -> list[str]:
    return [s.strip() for s in input_str.split(",")]


def _get_user():
    dotenv.load_dotenv(_PROJECT_DIR / ".env")
    return os.getenv("USER")


class RunTarget(strenum.StrEnum):
    DATA = "data"
    MODEL = "model"

    @classmethod
    def parse(cls, value: str) -> RunTarget:
        for member in cls:
            if value == member.value:
                return member
        raise ValueError

    @property
    def tag(self):
        return f"{self.value}_tag"

    @property
    def run_script(self):
        return f"_run_{self.value}.py"


@dataclasses.dataclass
class ExperimentOptions(args.TaskArguments):
    target: RunTarget
    model: str
    genres: list[str]
    train: bool = dataclasses.field(default=False)
    predict: bool = dataclasses.field(default=False)
    deploy_image: bool = dataclasses.field(default=False)
    deploy_code: bool = dataclasses.field(default=False)
    exp_name: str = dataclasses.field(default="")
    extra_path: str = dataclasses.field(default="")
    config_file_name: str = dataclasses.field(default="configs.json")

    @classmethod
    def define_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("-t", "--target", type=RunTarget.parse, required=True)
        parser.add_argument("-m", "--model", type=str, required=True)
        parser.add_argument("-g", "--genres", type=_parse_comma_separated_str, required=True)
        parser.add_argument("--train", type=bool, default=False)
        parser.add_argument("--predict", type=bool, default=False)
        parser.add_argument("--deploy-image", type=bool, default=False)
        parser.add_argument("--deploy-code", type=bool, default=False)
        parser.add_argument("--exp-name", type=str, default="")
        parser.add_argument("--extra-path", type=str, default="")
        parser.add_argument("--config-file-name", type=str, default="configs.json")

    @property
    def base_path(self):
        return pathlib.Path(__file__).parent / self.model / self.extra_path

    @property
    def config_file_path(self):
        return self.base_path / self.config_file_name

    @property
    def configs(self):
        with open(self.config_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def env_file(self):
        return self.base_path / f".env.{self.target}"

    @property
    def env_vars(self):
        with open(self.env_file, "r", encoding="utf-8") as f:
            return dict(line.strip().split("=", 1) for line in f if line.strip() and not line.startswith("#"))


def main():
    options = ExperimentOptions.parse_argument()

    user = _get_user()
    logger.info(f"Running experiment as {user}.")

    if options.deploy_image:  # TODO: 가끔 이게 Flag 대로 제어되지 않음.
        logger.info(f"{options.deploy_image}")
        logger.info("Deploying image.")
        # command_utils.run(f"cd {_PROJECT_DIR} && make deploy-image-{options.target}")

    if options.deploy_code:
        logger.info("Deploying code.")
        command_utils.run(f"cd {_PROJECT_DIR} && make deploy-code-{options.target}")

    # sys.exit(0)
    base_env_vars = options.env_vars

    for genre in options.genres:
        logger.info(f"Running {options.target} experiment for {genre}.")
        configs = options.configs[genre]

        for config in configs:
            target_config = config[options.target]

            os.makedirs(options.base_path / "env", exist_ok=True)
            config_file_path = options.base_path / "env" / "config.json"

            with open(config_file_path, "w", encoding="utf-8") as f:
                json.dump(target_config, f, indent=4)

            env_vars = copy.deepcopy(base_env_vars)
            env_vars["PARTITION_PATH"] += f"/genre={genre}"
            env_vars["GIN_PARAMS"] = f'["TRAIN={options.train}", "PREDICT={options.predict}"]'
            env_vars["GIN_PARAMS_JSON_PATH"] = str(config_file_path)
            env_vars["DATA_TAG"] = config[RunTarget.DATA.tag]
            env_vars["MODEL_TAG"] = config[RunTarget.MODEL.tag]

            if options.target == RunTarget.MODEL:
                exp_name = f"{user}-{genre}"
                if options.exp_name:
                    exp_name += f"-{options.exp_name}"
                env_vars["EXP_NAME"] = exp_name.replace("_", "-")

            with open(_PROJECT_DIR / f".env.{options.target}", "w", encoding="utf-8") as f:
                for k, v in env_vars.items():
                    f.write(f"{k}={v}\n")

            command_utils.run(f"cd {_PROJECT_DIR} && make run-{options.target}")


if __name__ == "__main__":
    main()
