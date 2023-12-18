import pickle
import numpy as np
import random
import scipy.signal
from utils import get_wav_segment
import librosa
import pathlib
import soundfile as sf
import tqdm
import os

class Augmentation:
    
    def __init__(self, config, reverb_prob=0.25):
        
        self.p_reverb = reverb_prob
        self.p_clip = 0.25
        self.p_blimit = 0.50
        self.p_blimit_noise = 0.50
        self.p_noise = 0.50
        self.sr = 22050
        self.eps = 1e-10
        self.noise_dir = pathlib.Path(config["general"]["aug_noise_dir"])

        if self.p_reverb > 0:
            with open("rir_10k.pickle", "rb") as fr:
                self.rir_list = pickle.load(fr)
        else:
            self.rir_list = None
        self.clip_range = (0.06, 0.9)
        self.blimit_cutoff_range = (850, 11025)
        self.blimit_order_range = (2, 10)
        self.blimit_filters = ['butter', 'cheby1', 'cheby2', 'bessel', 'ellip']
        self.noise_snr_range = (-5, 40)

        self.noisepaths = list(self.noise_dir.glob("*.wav"))
    
    def process(self, wav):

        wav_clean = wav.copy()

        # reverb
        if np.random.rand() < self.p_reverb:
            rir = random.choice(self.rir_list)
            wav = self.reverb(wav, rir)
        
        # clip
        if np.random.rand() < self.p_clip:
            eta = np.random.uniform(*self.clip_range)
            wav = self.clip(wav, eta)

        noise_path = random.choice(self.noisepaths)
        noise, _ = librosa.load(noise_path, sr=self.sr)
        noise_seg = get_wav_segment(noise, len(wav))

        # bandlimit
        if np.random.rand() < self.p_blimit:
            cutoff = np.random.uniform(*self.blimit_cutoff_range)
            order = np.random.randint(*self.blimit_order_range)
            filt = np.random.choice(self.blimit_filters)
            wav = self.bandlimit(wav, cutoff, order, filt)
            if np.random.rand() < self.p_blimit_noise:
                noise_seg = self.bandlimit(noise_seg, cutoff, order, filt)
        
        # noise & scale
        if np.random.rand() < self.p_noise:
            snr = np.random.uniform(*self.noise_snr_range)
            wav = self.noise(wav, noise_seg, snr)

        return wav, wav_clean

    def reverb(self, wav, rir):
        a = np.array([1.0])
        convolved_wav = scipy.signal.filtfilt(rir, a, wav, padtype=None)
        return convolved_wav
        
    def clip(self, wav, eta):
        amp = eta * np.max(wav)
        wav_clipped = np.maximum(np.minimum(wav, amp), -amp)
        return wav_clipped

    def noise(self, wav, noise_seg, snr):
        noise_seg = noise_seg * np.mean(np.abs(wav)) / (np.mean(np.abs(noise_seg)) + self.eps)
        noise_seg *= 10**(-snr/20)
        wav_noisy = wav + noise_seg
        return wav_noisy

    def bandlimit(self, wav, cutoff, order, filter_type='butter'):
        normalized_cutoff = cutoff / (0.5 * self.sr)
        if filter_type == 'butter':
            b, a = scipy.signal.butter(order, normalized_cutoff, btype='low')
        elif filter_type == 'cheby1':
            b, a = scipy.signal.cheby1(order, 0.5, normalized_cutoff, btype='low')
        elif filter_type == 'cheby2':
            b, a = scipy.signal.cheby2(order, 60, normalized_cutoff, btype='low')
        elif filter_type == 'bessel':
            b, a = scipy.signal.bessel(order, normalized_cutoff, btype='low', norm='phase')
        elif filter_type == 'ellip':
            b, a = scipy.signal.ellip(order, 0.5, 60, normalized_cutoff, btype='low')
        else:
            raise ValueError('Invalid filter_type')
        wav_filtered = scipy.signal.filtfilt(b, a, wav, padtype=None)
        return wav_filtered

def main():
    inp_dir = pathlib.Path("data/jvs_22k-spk")
    out_dir = pathlib.Path("jvs_22k-spk_aug")
    os.makedirs(out_dir, exist_ok=True)
    wavpaths = list(inp_dir.glob("*/*.wav"))
    config = {
        "general":{
            "aug_noise_dir": "/home/saeki/workspace/hdd1/TUT2017_22k/train"
        }
    }
    aug = Augmentation(config)
    SR = 22050

    for wp in tqdm.tqdm(wavpaths):
        spk = wp.parent.name
        wav, sr = sf.read(wp)
        assert sr == SR
        wav_aug, _ = aug.process(wav)
        out_path = out_dir / spk / wp.name
        os.makedirs(out_path.parent, exist_ok=True)
        sf.write(out_path, wav_aug, SR)


if __name__ == "__main__":
    main()


