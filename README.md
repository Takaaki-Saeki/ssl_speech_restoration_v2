# SelfRemaster: Self-Supervised Speech Restoration for Historical Audio Resources

Implementation of [SelfRemaster: Self-Supervised Speech Restoration for Historical Audio Resources](https://arxiv.org/abs/2203.12937) (in *IEEE Access, 2024*).
This repository includes the improved models of [the initial version presented in Interspeech 2022](https://github.com/Takaaki-Saeki/ssl_speech_restoration).

## Setup

1. Clone this repository: `git clone https://github.com/Takaaki-Saeki/ssl_speech_restoration_v2.git`.
2. CD into this repository: `cd ssl_speech_restoration_v2`.
3. Install python packages and download some pretrained models: `./setup.sh`.

## Data Preparation

You can speficy three types of datasets to be restored.
- `single`: The dataset consists of a single speaker or a single data domain, assuming a directory structure `data/{data_dir}/*.wav`. You should specify the number of wavfiles for validation and test sets for `n_val/n_test` in the config yaml file.
- `multi-seen`: The dataset consists of multiple speakers or data domains and train/val/test sets can contain data from the same speaker or domain. We assume a directory structure: `data/{data_dir}/{speaker_or_domain_name}/*.wav`. You should specify the number of wavfiles for validation and test sets for `n_val/n_test` in the config yaml file.
- `multi-unseen`: The dataset consists of multiple speakers or data domains and train/val/test should NOT contain data from the same speaker or domain. We assume a directory structure: `data/{data_dir}/{speaker_or_domain_name}/*.wav`. The number of speakers and domains is specified in `n_val/n_test` in the config yaml file.

In addition, you might want to use another clean speech dataset for dual-learning, supervised pretraining, and semi-supervised learning.

For example, if you use a single-speaker Japanese corpora with a single type of acoustic distortion:
- Download [JSUT Basic5000](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [JVS Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- Downsample them to 22.05 kHz and place them under `data/` as `jsut_22k` and `jvs_22k`.
    - JSUT is a single-speaker dataset and requires the structure as `jsut_22k/*.wav`. Note that this is the ground-truth clean speech data which correspond to the simulated data and is not used for training. You may want to use `jsut_22k` only to compare the restored speech and ground-truth speech.
    - JVS parallel100 includes 100-speaker data and requires the structure as `jvs_22k/${spkr_name}/*.wav`. This is a clean speech dataset used for the dual-learning, supervised pretraining, and semi-supervised learning. 
- Place simulated low-quality data under `./data` as `jsut_22k-degraded`.

## Training

First, you need to perform the preprocessing: splitting the data to train/val/test and dumping them.
```shell
python preprocess.py --config_path ${path_to_config_file}
```

To run the training, use the following command.
```shell
python train.py \
    --config_path ${path_to_config_file} \
    --stage ${stage_name} \
    --run_name ${run_name}
```

We have several example config files under `configs/`.
Each config file has the format: `{speaker/domain condition (single or multi)}_{feature (mel or sf)}_{training_condition (ssl or semi)}.yaml`
For example,
- `single_mel_ssl.yaml`: Using a single speaker/domain, mel spectrogram features, self-supervised learning.
- `single_sf_ssl.yaml`: Using a single speaker/domain, source-filter features, self-supervised learning.
- `multi_mel_semi.yaml`: Using multiple speakers/domains, mel spectrogram features, semi-supervised learning.

If you want to use the perceptual loss, enable the flag: `--use_perceptual_loss`.

If you want to use the supervised pretraining, you need to perform the pretraining as:
```shell
python train.py \
    --config_path configs/pretrain_jvs.yaml \
    --stage pretrain \
    --run_name pretrain_jvs
```
Then you need to specify the flag `--load_pretrained` and `--pretrained_path` when you run the self-supervised or semi-supervised learning.

For example, if you want to perform semi-supervised learning using the perceptual loss and supervised pretraining, you can use the following command.
```shell
python train.py \
    --config_path multi_mel_semi.yaml \
    --stage semi \
    --run_name multi_semi_pre_ploss \
    --use_perceptual_loss \
    --load_pretrained \
    --pretrained_path pretrained.ckpt
```

For other options, see `train.py`.

## Speech restoration
To perform speech restoration of the test data, run the following command.
```shell
python eval.py \
    --config_path ${path_to_config_file} \
    --ckpt_path ${path_to_checkpoint} \
    --stage s${stage_name} \
    --run_name ${run_name}
```

Note that you might need to use the same flags as the training procedure, otherwise checkpoint key mismatch can occur.
For example, if you want to run the pretrained model with the semi-supervised learning, perceptual loss, and supervised pretraining, you can use the following command.
```shell
python eval.py \
    --config_path multi_mel_semi.yaml \
    --ckpt_path multi_semi_pre_ploss.ckpt
    --stage semi \
    --run_name multi_semi_pre_ploss \
    --use_perceptual_loss \
    --load_pretrained \
    --pretrained_path pretrained.ckpt
```

For other options, see `eval.py`.
