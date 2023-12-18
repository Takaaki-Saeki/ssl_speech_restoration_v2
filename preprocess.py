import numpy as np
import os
import librosa
import tqdm
import pickle
import random
import argparse
import yaml
import pathlib


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=pathlib.Path)
    parser.add_argument("--corpus_type", default=None, type=str)
    parser.add_argument("--source_path", default=None, type=pathlib.Path)
    parser.add_argument("--source_path_task", default=None, type=pathlib.Path)
    parser.add_argument("--aux_path", default=None, type=pathlib.Path)
    parser.add_argument("--preprocessed_path", default=None, type=pathlib.Path)
    parser.add_argument("--n_train", default=None, type=int)
    parser.add_argument("--n_val", default=None, type=int)
    parser.add_argument("--n_test", default=None, type=int)
    return parser.parse_args()


def preprocess(config):

    # configs
    preprocessed_dir = pathlib.Path(config["general"]["preprocessed_path"])
    n_val = config["preprocess"]["n_val"]
    n_test = config["preprocess"]["n_test"]
    SR = config["preprocess"]["sampling_rate"]

    os.makedirs(preprocessed_dir, exist_ok=True)

    sourcepath = pathlib.Path(config["general"]["source_path"]).resolve()

    test_sub_filelist = None
    if config["general"]["corpus_type"] == "single":
        fulllist = list(sourcepath.glob("*.wav"))
        random.seed(0)
        random.shuffle(fulllist)
        if len(fulllist) <= (n_val + n_test):
            # Only for test processing (see configs/test/*.yaml).
            test_filelist = fulllist[: n_test]
            val_filelist = fulllist[n_test : n_test+n_val]
            train_filelist = fulllist[n_test+n_val:]
        else:
            val_filelist = fulllist[-n_val :]
            test_filelist = fulllist[-n_val-n_test : -n_val]
            train_filelist = fulllist[ :-n_val-n_test]
        filelist = train_filelist + val_filelist + test_filelist
    elif config["general"]["corpus_type"] == "multi-seen":
        if config["general"]["mix_channel"]:
            ch_list = list(set([x.parent for x in sourcepath.glob("*/*.wav")]))
            fulllist = list(sourcepath.glob("*/*.wav"))
            train_filelist = []
            val_filelist = []
            test_filelist = []
            test_sub_filelist = []
            random.seed(0)
            random.shuffle(fulllist)
            for ch in ch_list:
                sourcechpath = sourcepath / ch
                ch_filelist = list(sourcechpath.glob("*.wav"))
                val_filelist.extend(ch_filelist[-n_val :])
                test_filelist.extend(ch_filelist[-n_val-n_test : -n_val])
                train_filelist.extend(ch_filelist[ :-n_val-n_test])
                test_sub_filelist.extend(ch_filelist[-n_val-2*n_test:-n_val-n_test])
            filelist = train_filelist + val_filelist + test_filelist
        else:
            fulllist = list(sourcepath.glob("*/*.wav"))
            random.seed(0)
            random.shuffle(fulllist)
            val_filelist = fulllist[-n_val :]
            test_filelist = fulllist[-n_val-n_test : -n_val]
            train_filelist = fulllist[ :-n_val-n_test]
            filelist = train_filelist + val_filelist + test_filelist
    elif config["general"]["corpus_type"] == "multi-unseen":
        spk_list = list(set([x.parent for x in sourcepath.glob("*/*.wav")]))
        train_filelist = []
        val_filelist = []
        test_filelist = []
        random.seed(0)
        random.shuffle(spk_list)
        for i, spk in enumerate(spk_list):
            sourcespkpath = sourcepath / spk
            if i < n_val:
                val_filelist.extend(list(sourcespkpath.glob("*.wav")))
            elif i < n_val+n_test:
                test_filelist.extend(list(sourcespkpath.glob("*.wav")))
            else:
                train_filelist.extend(list(sourcespkpath.glob("*.wav")))
        filelist = train_filelist + val_filelist + test_filelist
    else:
        raise NotImplementedError(
            "corpus_type specified in config.yaml should be {single, multi-seen, multi-unseen}"
        )

    with open(preprocessed_dir / "train.txt", "w", encoding="utf-8") as f:
        for m in train_filelist:
            f.write(str(m) + "\n")
    with open(preprocessed_dir / "val.txt", "w", encoding="utf-8") as f:
        for m in val_filelist:
            f.write(str(m) + "\n")
    with open(preprocessed_dir / "test.txt", "w", encoding="utf-8") as f:
        for m in test_filelist:
            f.write(str(m) + "\n")
    if test_sub_filelist is not None:
        with open(preprocessed_dir / "test_sub.txt", "w", encoding="utf-8") as f:
            for m in test_sub_filelist:
                f.write(str(m) + "\n")
    
    # Counting labels
    labels = set()
    label2idx = {}
    for wp in filelist:
        labels.add(wp.parent.name)
    labels = sorted(list(labels))
    for idx, label in enumerate(labels):
        label2idx[label] = idx

    for wp in tqdm.tqdm(filelist):

        if config["general"]["corpus_type"] == "single":
            basename = str(wp.stem)
        else:
            basename = str(wp.parent.name) + "-" + str(wp.stem)

        wav, _ = librosa.load(wp, sr=SR)
        wavsegs = []

        if config["general"]["aux_path"] != None:
            auxpath = pathlib.Path(config["general"]["aux_path"])
            if config["general"]["corpus_type"] == "single":
                wav_aux, _ = librosa.load(auxpath / wp.name, sr=SR)
            else:
                wav_aux, _ = librosa.load(auxpath / wp.parent.name / wp.name, sr=SR)
            wavauxsegs = []

        if config["general"]["aux_path"] == None:
            wavsegs.append(wav)
        else:
            min_seq_len = min(len(wav), len(wav_aux))
            wav = wav[:min_seq_len]
            wav_aux = wav_aux[:min_seq_len]
            wavsegs.append(wav)
            wavauxsegs.append(wav_aux)

        wavsegs = np.asarray(wavsegs).astype(np.float32)
        if config["general"]["aux_path"] != None:
            wavauxsegs = np.asarray(wavauxsegs).astype(np.float32)
        else:
            wavauxsegs = None

        labelname = wp.parent.name
        labelidx = label2idx[labelname]

        d_preprocessed = {
            "wavs": wavsegs, "wavsaux": wavauxsegs,
            "labelidx": labelidx, "labelname": labelname
        }

        with open(preprocessed_dir / "{}.pickle".format(basename), "wb") as fw:
            pickle.dump(d_preprocessed, fw)


def make_config(config, name):
    d_config = {}
    d_config["general"] = {}
    d_config["preprocess"] = {}
    d_config["general"]["source_path"] = config["general"][name]["source_path"]
    d_config["general"]["aux_path"] = config["general"][name]["aux_path"]
    d_config["general"]["preprocessed_path"] = config["general"][name]["preprocessed_path"]
    d_config["general"]["corpus_type"] = config["general"][name]["corpus_type"]
    d_config["preprocess"]["sampling_rate"] = config["preprocess"]["sampling_rate"]

    if name == "data_ssl":
        d_config["general"]["mix_channel"] = config["general"]["mix_channel"]
        d_config["preprocess"]["n_val"] = config["preprocess"]["n_val"]
        d_config["preprocess"]["n_test"] = config["preprocess"]["n_test"]
    elif name == "data_semi":
        d_config["general"]["mix_channel"] = False
        if config["general"]["mix_channel"]:
            # Converting channel-level number to entire number
            d_config["preprocess"]["n_val"] = config["preprocess"]["n_val"] * 4
            d_config["preprocess"]["n_test"] = config["preprocess"]["n_test"] * 4
        else:
            d_config["preprocess"]["n_val"] = config["preprocess"]["n_val"]
            d_config["preprocess"]["n_test"] = config["preprocess"]["n_test"]
    else:
        raise NotImplementedError(
            "name specified in config.yaml should be {data_ssl, data_semi}"
        )
    return d_config


def process_all(config):
    if config["general"]["stage"] == "semi":
        ssl_config = make_config(config, "data_ssl")
        semi_config = make_config(config, "data_semi")
        preprocess(ssl_config)
        preprocess(semi_config)
    else:
        preprocess(config)


if __name__ == "__main__":
    args = get_arg()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    for key in ["corpus_type", "source_path", "aux_path", "preprocessed_path"]:
        if getattr(args, key) != None:
            config["general"][key] = str(getattr(args, key))
    for key in ["n_train", "n_val", "n_test"]:
        if getattr(args, key) != None:
            config["preprocess"][key] = getattr(args, key)

    print("Performing preprocessing ...")
    process_all(config)

    if "dual" in config:
        if config["dual"]["enable"]:
            task_config = yaml.load(
                open(config["dual"]["config_path"], "r"), Loader=yaml.FullLoader
            )
            if config["general"]["stage"] == "semi":
                preprocessed_dir = pathlib.Path(config["general"]["data_ssl"]["preprocessed_path"])
            else:
                preprocessed_dir = pathlib.Path(config["general"]["preprocessed_path"])
            task_preprocessed_dir = (
                preprocessed_dir.parent
                / pathlib.Path(task_config["general"]["preprocessed_path"]).name
            )
            task_config["general"]["preprocessed_path"] = task_preprocessed_dir
            if args.source_path_task != None:
                task_config["general"]["source_path"] = args.source_path_task
            print("Performing preprocessing for dual learning ...")
            preprocess(task_config)