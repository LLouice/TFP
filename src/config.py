import sys
import argparse


class Config():
    def __init__(self):
        parser = argparse.ArgumentParser()

        # "train params"
        sub_parser = parser.add_argument_group("train")
        sub_parser.add_argument("--epos", type=int, default=300)
        sub_parser.add_argument("--lr", type=float, default=1e-3)
        sub_parser.add_argument('--lr_decay', type=float, default=0.97)
        sub_parser.add_argument("--wd", type=float, default=1e-4)
        sub_parser.add_argument('--opt',
                                type=str,
                                choices=["adam", "adamw", "sgd"],
                                default="adamw")
        # 'dataloader
        sub_parser = parser.add_argument_group("dataloader")
        sub_parser.add_argument("--bs", type=int, default=64)
        sub_parser.add_argument("--bs_dev", type=int, default=512)
        sub_parser.add_argument("--nw0", type=int, default=4)
        sub_parser.add_argument("--nw1", type=int, default=8)

        # 'device'
        sub_parser = parser.add_argument_group("device")
        sub_parser.add_argument("--gpus", type=str, default="0")
        sub_parser.add_argument("--nogpu",
                                type=str,
                                default="false",
                                choices=["true", "false"])
        sub_parser.add_argument("--tpu",
                                type=str,
                                default="false",
                                choices=["true", "false"])

        # 'trick'
        sub_parser = parser.add_argument_group("debug")
        sub_parser.add_argument("--dbg",
                                type=str,
                                default="false",
                                choices=["true", "false"])
        sub_parser.add_argument("--test",
                                type=str,
                                default="false",
                                choices=["true", "false"])
        sub_parser.add_argument("--lr_find",
                                type=str,
                                default="false",
                                choices=["true", "false"])

        # 'eval'
        sub_parser = parser.add_argument_group("eval")
        sub_parser.add_argument("--check_val", type=int, default=5)

        # 'other'
        sub_parser = parser.add_argument_group("other")
        sub_parser.add_argument("--pb_rate", type=int, default=20)
        sub_parser.add_argument("--exp", type=str, default="default")

        # 'tune'
        sub_parser = parser.add_argument_group("tune")
        sub_parser.add_argument("--tune_name", type=str, default="tune_0")
        sub_parser.add_argument("--tune_schd",
                                type=str,
                                default="asha",
                                choices=["asha", "pbt"])
        sub_parser.add_argument("--tune_num_samples", type=int, default=8)
        sub_parser.add_argument("--tune_num_cpus", type=int, default=16)
        sub_parser.add_argument("--tune_num_gpus", type=int, default=2)
        sub_parser.add_argument("--tune_gups", type=str, default="0,1")
        sub_parser.add_argument("--tune_per_cpu", type=float, default=4)
        sub_parser.add_argument("--tune_per_gpu", type=float, default=0.5)

        # "program params"
        sub_parser = parser.add_argument_group("program")
        sub_parser.add_argument("--data_pt", type=str, default="data/data.h5")
        sub_parser.add_argument("--log_file",
                                type=str,
                                default="runs/logs/train")
        sub_parser.add_argument("--log_level",
                                type=str,
                                choices=["info", "debug", "warning"],
                                default="debug")

        # "model params"
        sub_parser = parser.add_argument_group("model")
        sub_parser.add_argument("--model",
                                type=str,
                                choices=["Model0", "Model1"],
                                default="Model0")
        self.parser = parser

    def get_parser(self, parser=None):
        if parser:
            self.parser = parser
        return self.parser

    def get_config(self, parser=None):
        if parser:
            self.parser = parser
        config, _ = self.parser.parse_known_args()
        _convert(config)
        _add_log_prefix(config)
        return config


def _bool(x):
    assert x in ("true", "false"), "the x is {}".format(x)
    return True if x == "true" else False


def _convert(config):
    for k, v in config.__dict__.items():
        if v in ("true", "false"):
            config.__dict__[k] = _bool(v)
        if v == "None" or v == "none":
            config.__dict__[k] = None


def _add_log_prefix(config):
    prefix = "_".join([arg[2:] for arg in sys.argv[1:]])
    config.log_file += "_{}".format(prefix)


if __name__ == "__main__":
    config = Config().get_config()
    print(config)
