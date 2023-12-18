import argparse
import os
import pathlib
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.csv_logs import CSVLogger

from dataset import DataModule
from lightning_module import (
    PretrainLightningModule,
    SSLStepLightningModule,
    SSLDualLightningModule,
    SemiDualLightningModule
)


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=pathlib.Path)
    parser.add_argument("--ckpt_path", required=True, type=pathlib.Path)
    parser.add_argument(
        "--stage", required=True, type=str,
        choices=[
            "pretrain", "pretrain-v2", "ssl-step", "ssl-dual", "semi", "ssl-dual-v2", "semi-v2"
        ]
    )
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--use_gst", action="store_true")
    parser.add_argument("--use_channel_loss", action="store_true")
    parser.add_argument("--use_perceptual_loss", action="store_true")
    parser.add_argument("--dump_chfeats", action="store_true")
    return parser.parse_args()


def eval(args, config, output_path):

    csvlogger = CSVLogger(save_dir=output_path, name="test_log")
    trainer = Trainer(
        gpus=-1,
        deterministic=False,
        auto_select_gpus=True,
        benchmark=True,
        logger=[csvlogger],
        default_root_dir=os.getcwd(),
        min_steps=1,
        min_epochs=1
    )

    if config["general"]["stage"] == "pretrain":
        model = PretrainLightningModule(config).load_from_checkpoint(
            checkpoint_path=args.ckpt_path, config=config
        )
    elif config["general"]["stage"] == "ssl-step":
        model = SSLStepLightningModule(config).load_from_checkpoint(
            checkpoint_path=args.ckpt_path, config=config, strict=False # For supervised
        )
    elif config["general"]["stage"] == "ssl-dual":
        model = SSLDualLightningModule(config).load_from_checkpoint(
            checkpoint_path=args.ckpt_path, config=config
        )
    elif config["general"]["stage"] == "semi":
        model = SemiDualLightningModule(config).load_from_checkpoint(
            checkpoint_path=args.ckpt_path, config=config
        )
    else:
        raise NotImplementedError()

    datamodule = DataModule(config)
    trainer.test(model=model, verbose=True, datamodule=datamodule)


if __name__ == "__main__":
    args = get_arg()
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    output_path = str(pathlib.Path(config["general"]["output_path"]) / args.run_name)
    config["general"]["stage"] = str(getattr(args, "stage"))
    
    if args.use_gst:
        config["general"]["use_gst"] = args.use_gst
    if args.use_perceptual_loss:
        config["train"]["perceptual_loss"]["enabled"] = args.use_perceptual_loss
    if args.use_channel_loss:
        config["train"]["channel_loss"]["enabled"] = args.use_channel_loss

    config["general"]["dump_chfeats"] = args.dump_chfeats

    eval(args, config, output_path)
