import pickle
import pathlib
import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader
import pytorch_lightning as pl
import numpy as np
import yaml
import torchaudio
import pyworld
import pysptk
import random
import wave
from augmentation import Augmentation
from utils import read_audio_section, get_wav_segment


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batchsize = config["train"]["batchsize"]

    def setup(self, stage):

        if self.config["general"]["stage"].startswith("semi"):
            preprocessed_dir = pathlib.Path(self.config["general"]["data_ssl"]["preprocessed_path"])
        else:
            preprocessed_dir = pathlib.Path(self.config["general"]["preprocessed_path"])

        if not preprocessed_dir.exists():
            raise RuntimeError("Preprocessed directory was not be found")

        if "dual" in self.config:
            if self.config["dual"]["enable"]:
                task_config = yaml.load(
                    open(self.config["dual"]["config_path"], "r"),
                    Loader=yaml.FullLoader,
                )
                task_preprocessed_dir = (
                    preprocessed_dir.parent
                    / pathlib.Path(task_config["general"]["preprocessed_path"]).name
                )
                if not task_preprocessed_dir.exists():
                    raise RuntimeError(
                        "Preprocessed directory for multi-task learning was not be found"
                    )

        self.flnames = {
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
        }

    def get_ds(self, phase):
        ds = Dataset(
            self.flnames[phase],
            pathlib.Path(self.config["general"]["preprocessed_path"]),
            self.config,
            phase,
            self.config["general"]["corpus_type"])
        return ds
    
    def get_multi_ds(self, phase):
        ssl_ds = Dataset(
            self.flnames[phase],
            pathlib.Path(self.config["general"]["data_ssl"]["preprocessed_path"]),
            self.config,
            phase,
            self.config["general"]["data_ssl"]["corpus_type"],
            override_stage="ssl-dual")
        if self.config["general"]["semi_aug"]:
            override_stage = "supervised_aug"
        else:
            override_stage = "supervised_data"
        semi_ds = Dataset(
            self.flnames[phase],
            pathlib.Path(self.config["general"]["data_semi"]["preprocessed_path"]),
            self.config,
            phase,
            self.config["general"]["data_semi"]["corpus_type"],
            override_stage=override_stage)
        return ssl_ds, semi_ds

    def get_loader(self, phase, batchsize):
        if self.config["general"]["stage"].startswith("semi"):
            ds_ssl, ds_semi = self.get_multi_ds(phase)
            dl_ssl = DataLoader(
                ds_ssl,
                batchsize,
                shuffle=True if phase == "train" else False,
                num_workers=self.config["train"]["num_workers"],
                drop_last=True,
            )
            dl_semi = DataLoader(
                ds_semi,
                batchsize,
                shuffle=True if phase == "train" else False,
                num_workers=self.config["train"]["num_workers"],
                drop_last=True,
            )
            if len(dl_ssl) < len(dl_semi):
                loader_mode = "min_size"
            else:
                loader_mode = "max_size_cycle"
            combined_loader = CombinedLoader(
                {"ssl": dl_ssl, "semi": dl_semi}, mode=loader_mode)
            return combined_loader
        else:
            ds = self.get_ds(phase)
            dl = DataLoader(
                ds,
                batchsize,
                shuffle=True if phase == "train" else False,
                num_workers=self.config["train"]["num_workers"],
                drop_last=True,
            )
            return dl

    def train_dataloader(self):
        return self.get_loader(phase="train", batchsize=self.batchsize)

    def val_dataloader(self):
        return self.get_loader(phase="val", batchsize=self.batchsize)

    def test_dataloader(self):
        return self.get_loader(phase="test", batchsize=1)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filetxt, preprocessed_dir, config, phase, corpus_type, override_stage=None):

        self.preprocessed_dir = preprocessed_dir
        self.config = config
        self.spec_module = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["preprocess"]["sampling_rate"],
            n_fft=config["preprocess"]["fft_length"],
            win_length=config["preprocess"]["frame_length"],
            hop_length=config["preprocess"]["frame_shift"],
            f_min=config["preprocess"]["fmin"],
            f_max=config["preprocess"]["fmax"],
            n_mels=config["preprocess"]["n_mels"],
            power=1,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )
        self.phase = phase
        if phase == "test":
            self.segment_length = -1
        else:
            self.segment_length = config["preprocess"]["segment_length"]
        self.corpus_type = corpus_type
        
        self.stage = config["general"]["stage"]
        if override_stage is not None:
            self.stage = override_stage

        self.pretrain_switch_prob = 0.50
        self.aug = Augmentation(config, reverb_prob=0.0)

        with open(self.preprocessed_dir / filetxt, "r") as fr:
            self.filelist = [pathlib.Path(path.strip("\n")) for path in fr]

        self.d_out = dict()
        for item in ["wavs", "wavsaux", "labelname", "labelidx"]:
            self.d_out[item] = []
        if phase == "test":
            self.d_out["wavname"] = []

        for wp in self.filelist:

            if corpus_type == "single":
                basename = str(wp.stem)
            else:
                basename = str(wp.parent.name) + "-" + str(wp.stem)

            with open(self.preprocessed_dir / "{}.pickle".format(basename), "rb") as fw:
                d_preprocessed = pickle.load(fw)

            for item in ["wavs", "wavsaux"]:
                try:
                    self.d_out[item].extend(d_preprocessed[item])
                except:
                    pass
            self.d_out["labelname"].append(d_preprocessed["labelname"])
            self.d_out["labelidx"].append(d_preprocessed["labelidx"])
            if phase == "test":
                self.d_out["wavname"].append(wp.stem)

        for item in ["wavs", "wavsaux", "labelidx"]:
            if self.d_out[item] != None:
                self.d_out[item] = np.asarray(self.d_out[item])

        if "dual" in self.config:
            if self.config["dual"]["enable"]:
                task_config = yaml.load(
                    open(config["dual"]["config_path"], "r"),
                    Loader=yaml.FullLoader,
                )
                task_preprocessed_dir = (
                    self.preprocessed_dir.parent
                    / pathlib.Path(task_config["general"]["preprocessed_path"]).name
                )
                with open(task_preprocessed_dir / filetxt, "r") as fr:
                    task_filelist = [pathlib.Path(path.strip("\n")) for path in fr]
                self.d_out["wavstask"] = []
                task_corpus_type = task_config["general"]["corpus_type"]
                for wp in task_filelist:
                    if task_corpus_type == "single":
                        basename = str(wp.stem)
                    else:
                        basename = str(wp.parent.name) + "-" + str(wp.stem)
                    with open(
                        task_preprocessed_dir / "{}.pickle".format(basename), "rb"
                    ) as fw:
                        d_preprocessed = pickle.load(fw)
                    self.d_out["wavstask"].extend(d_preprocessed["wavs"])
                self.d_out["wavstask"] = np.asarray(self.d_out["wavstask"])

    def __len__(self):
        return len(self.d_out["wavs"])

    def __getitem__(self, idx):

        d_batch = {}

        d_batch["labelname"] = self.d_out["labelname"][idx]
        d_batch["labelidx"] = torch.tensor(self.d_out["labelidx"][idx])
        if self.phase == "test":
            d_batch["wavname"] = self.d_out["wavname"][idx]

        if self.d_out["wavs"].size > 0:
            d_batch["wavs"] = torch.from_numpy(self.d_out["wavs"][idx])

        if self.d_out["wavsaux"].size > 0:
            d_batch["wavsaux"] = torch.from_numpy(self.d_out["wavsaux"][idx])
        
        if (self.d_out["wavs"].size > 0) & (self.segment_length > 0):
            if self.d_out["wavsaux"].size > 0:
                d_batch["wavs"], d_batch["wavsaux"] = self.get_segment(
                    d_batch["wavs"],
                    self.segment_length,
                    d_batch["wavsaux"]
                )
            else:
                d_batch["wavs"] = self.get_segment(d_batch["wavs"], self.segment_length)

        if self.stage.startswith("pretrain"):
            if self.config["train"]["augment"]:
                if np.random.rand() < self.pretrain_switch_prob:
                    d_batch["wavs"], d_batch["wavsaux"] = self.augmentation(d_batch["wavsaux"])
            d_batch["wavs"] = self.normalize_waveform(d_batch["wavs"])
            d_batch["wavsaux"] = self.normalize_waveform(d_batch["wavsaux"])
            if len(d_batch["wavs"]) != len(d_batch["wavsaux"]):
                min_seq_len = min(len(d_batch["wavs"]), len(d_batch["wavsaux"]))
                d_batch["wavs"] = d_batch["wavs"][:min_seq_len]
                d_batch["wavsaux"] = d_batch["wavsaux"][:min_seq_len]
            d_batch["melspecs"] = self.calc_spectrogram(d_batch["wavs"])
            if self.config["general"]["feature_type"] == "melspec":
                d_batch["melspecsaux"] = self.calc_spectrogram(d_batch["wavsaux"])
            elif self.config["general"]["feature_type"] == "vocfeats":
                d_batch["melceps"] = self.calc_melcep(d_batch["wavsaux"])
                d_batch["f0s"] = self.calc_f0(d_batch["wavs"])
                d_batch["melcepssrc"] = self.calc_melcep(d_batch["wavs"])
            else:
                raise NotImplementedError()

        elif self.stage.startswith("supervised"):
            if self.stage == "supervised_aug":
                d_batch["wavs"], d_batch["wavsaux"] = self.augmentation(d_batch["wavsaux"])
            d_batch["wavs"] = self.normalize_waveform(d_batch["wavs"])
            d_batch["wavsaux"] = self.normalize_waveform(d_batch["wavsaux"])
            if len(d_batch["wavs"]) != len(d_batch["wavsaux"]):
                min_seq_len = min(len(d_batch["wavs"]), len(d_batch["wavsaux"]))
                d_batch["wavs"] = d_batch["wavs"][:min_seq_len]
                d_batch["wavsaux"] = d_batch["wavsaux"][:min_seq_len]
            d_batch["melspecs"] = self.calc_spectrogram(d_batch["wavs"])
            if self.config["general"]["feature_type"] == "melspec":
                d_batch["melspecsaux"] = self.calc_spectrogram(d_batch["wavsaux"])
            elif self.config["general"]["feature_type"] == "vocfeats":
                d_batch["melceps"] = self.calc_melcep(d_batch["wavsaux"])
                d_batch["f0s"] = self.calc_f0(d_batch["wavs"])
                d_batch["melcepssrc"] = self.calc_melcep(d_batch["wavs"])
            else:
                raise NotImplementedError()

        elif self.stage.startswith("ssl"):
            d_batch["wavs"] = self.normalize_waveform(d_batch["wavs"])
            d_batch["melspecs"] = self.calc_spectrogram(d_batch["wavs"])
            if self.config["general"]["feature_type"] == "vocfeats":
                d_batch["f0s"] = self.calc_f0(d_batch["wavs"])
                d_batch["melcepssrc"] = self.calc_melcep(d_batch["wavs"])
            if self.d_out["wavsaux"].size > 0:
                d_batch["wavsaux"] = self.normalize_waveform(d_batch["wavsaux"])
                if self.config["general"]["feature_type"] == "melspec":
                    d_batch["melspecsaux"] = self.calc_spectrogram(d_batch["wavsaux"])
                elif self.config["general"]["feature_type"] == "vocfeats":
                    d_batch["melceps"] = self.calc_melcep(d_batch["wavsaux"])
            if "dual" in self.config:
                if self.config["dual"]["enable"]:
                    rand_idx = random.randint(0, len(self.d_out["wavstask"]) - 1)
                    d_batch["wavstask"] = torch.from_numpy(self.d_out["wavstask"][rand_idx])
                    if self.segment_length > 0:
                        d_batch["wavstask"] = self.get_segment(
                            d_batch["wavstask"], self.segment_length
                        )
                    d_batch["wavstask"] = self.normalize_waveform(
                        d_batch["wavstask"]
                    )

                    # Noise augmentation to address additive noises
                    if self.config["dual"]["use_noise_aug"]:
                        noise_dir = pathlib.Path(self.config["dual"]["noise_dir"])
                        noise_path = random.choice(list(noise_dir.glob("*.wav")))
                        with wave.open(str(noise_path)) as mywav:
                            duration_sec = mywav.getnframes() / mywav.getframerate()
                        wav_noise, sr_noise = read_audio_section(
                            noise_path,
                            duration_sec,
                            self.segment_length
                        )
                        assert sr_noise == self.config["preprocess"]["sampling_rate"]
                        if len(wav_noise.shape) > 1:
                            wav_noise = wav_noise[:, 0]
                        seg_size_noise = self.config["model"]["speaker_encoder"]["seq_len"] * sr_noise
                        wav_noise = get_wav_segment(wav_noise, seg_size_noise).astype(np.float32)
                        wav_noise = self.normalize_waveform(
                            torch.from_numpy(wav_noise))
                        wav_noise = self.changegain_waveform(wav_noise)
                        d_batch["wavstaskaug"] = d_batch["wavstask"] + wav_noise
                        
                    if self.config["general"]["feature_type"] == "melspec":
                        d_batch["melspecstask"] = self.calc_spectrogram(
                            d_batch["wavstask"]
                        )
                    elif self.config["general"]["feature_type"] == "vocfeats":
                        d_batch["melcepstask"] = self.calc_melcep(d_batch["wavstask"])
                    else:
                        raise NotImplementedError()
        else:
            raise NotImplementedError()

        return d_batch

    def calc_spectrogram(self, wav):
        specs = self.spec_module(wav)
        log_spec = torch.log(
            torch.clamp_min(specs, self.config["preprocess"]["min_magnitude"])
            * self.config["preprocess"]["comp_factor"]
        ).to(torch.float32)
        return log_spec

    def calc_melcep(self, wav):
        wav = wav.numpy()
        _, sp, _ = pyworld.wav2world(
            wav.astype(np.float64),
            self.config["preprocess"]["sampling_rate"],
            fft_size=self.config["preprocess"]["fft_length"],
            frame_period=(
                self.config["preprocess"]["frame_shift"]
                / self.config["preprocess"]["sampling_rate"]
                * 1000
            ),
        )
        melcep = pysptk.sp2mc(
            sp,
            order=self.config["preprocess"]["cep_order"],
            alpha=pysptk.util.mcepalpha(self.config["preprocess"]["sampling_rate"]),
        ).transpose(1, 0)
        melcep = torch.from_numpy(melcep).to(torch.float32)
        return melcep

    def calc_f0(self, wav):
        if self.config["preprocess"]["f0_extractor"] == "dio":
            return self.calc_f0_dio(wav)
        elif self.config["preprocess"]["f0_extractor"] == "harvest":
            return self.calc_f0_harvest(wav)
        elif self.config["preprocess"]["f0_extractor"] == "swipe":
            return self.calc_f0_swipe(wav)
        else:
            raise NotImplementedError()

    def calc_f0_dio(self, wav):
        wav = wav.numpy()
        _f0, _t = pyworld.dio(
            wav.astype(np.float64),
            self.config["preprocess"]["sampling_rate"],
            frame_period=(
                self.config["preprocess"]["frame_shift"]
                / self.config["preprocess"]["sampling_rate"]
                * 1000
            ),
        )
        f0 = pyworld.stonemask(
            wav.astype(np.float64), _f0, _t, self.config["preprocess"]["sampling_rate"]
        )
        f0 = torch.from_numpy(f0).to(torch.float32)
        return f0

    def calc_f0_harvest(self, wav):
        wav = wav.numpy()
        _f0, _t = pyworld.harvest(
            wav.astype(np.float64),
            self.config["preprocess"]["sampling_rate"],
            frame_period=(
                self.config["preprocess"]["frame_shift"]
                / self.config["preprocess"]["sampling_rate"]
                * 1000
            ),
        )
        f0 = pyworld.stonemask(
            wav.astype(np.float64), _f0, _t, self.config["preprocess"]["sampling_rate"]
        )
        f0 = torch.from_numpy(f0).to(torch.float32)
        return f0

    def calc_f0_swipe(self, wav):
        wav = wav.numpy()
        f0 = pysptk.sptk.swipe(
            wav.astype(np.float64),
            fs=self.config["preprocess"]["sampling_rate"],
            min=71,
            max=800,
            hopsize=self.config["preprocess"]["frame_shift"],
            otype="f0",
        )
        f0 = torch.from_numpy(f0).to(torch.float32)
        return f0

    def augmentation(self, wav):
        wav_np = wav.numpy()
        wav_deg_np, wav_clean_np = self.aug.process(wav_np)
        wav_deg = torch.from_numpy(wav_deg_np.astype(np.float32))
        wav_clean = torch.from_numpy(wav_clean_np.astype(np.float32))
        return wav_deg, wav_clean

    def normalize_waveform(self, wav):
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav.unsqueeze(0),
            self.config["preprocess"]["sampling_rate"],
            [["gain", "-n"]],
        )
        return wav.squeeze(0)

    def changegain_waveform(self, wav):
        """Change gain of wav based on sox
        Input: (seq_len,)
        Output: (seq_len,)
        """
        db = int(random.randrange(-30, -10, 1))
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav.unsqueeze(0),
            self.config["preprocess"]["sampling_rate"],
            [["gain", "{}".format(db)]],
            )
        wav = wav.squeeze(0)
        return wav

    def get_segment(self, wav, segment_length, wavaux=None):
        seg_size = self.config["preprocess"]["sampling_rate"] * segment_length
        if len(wav) >= seg_size:
            max_wav_start = len(wav) - seg_size
            wav_start = random.randint(0, max_wav_start)
            wav = wav[wav_start : wav_start + seg_size]
            if wavaux != None:
                wavaux = wavaux[wav_start : wav_start + seg_size]
        else:
            wav = torch.nn.functional.pad(wav, (0, seg_size - len(wav)), "constant")
            if wavaux != None:
                wavaux = torch.nn.functional.pad(wavaux, (0, seg_size - len(wavaux)), "constant")
        if wavaux != None:
            return wav, wavaux
        else:
            return wav
