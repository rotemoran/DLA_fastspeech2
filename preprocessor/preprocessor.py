import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
import pycwt as cwt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


def icwt(cwt_spec, num_scales=None):
    """
    Inverse Continuous Wavelet Transform (iCWT) per FastSpeech 2 paper Appendix C, Equation 2.
    Recovers 1D pitch contour from CWT spectrogram: F̂_0(t) = Σ_{i=1}^{n} Ŵ_i(t)(i+2.5)^{-5/2}.
    (Ming et al., 2016; Suni et al., 2013)
    Args:
        cwt_spec: np.ndarray shape (num_scales, T) or (T, num_scales). CWT coefficients.
        num_scales: int, number of scales (default: first dimension if (num_scales, T)).
    Returns:
        pitch_1d: np.ndarray shape (T,) recovered pitch contour in normalized log domain.
    """
    cwt_spec = np.asarray(cwt_spec, dtype=np.float64)
    if cwt_spec.ndim != 2:
        raise ValueError("cwt_spec must be 2D (num_scales, T) or (T, num_scales)")
    if num_scales is None:
        num_scales = cwt_spec.shape[0]
    # Assume (num_scales, T); if (T, num_scales) transpose
    if cwt_spec.shape[1] == num_scales and cwt_spec.shape[0] != num_scales:
        cwt_spec = cwt_spec.T  # (T, num_scales) -> (num_scales, T)
    n_scale, T = cwt_spec.shape
    # Scale indices i = 1..n (paper 1-based); weight for index j (0-based) = (j+1+2.5)**(-5/2)
    weights = np.array([(j + 1 + 2.5) ** (-5.0 / 2.0) for j in range(n_scale)], dtype=np.float64)
    # F̂_0(t) = Σ_i Ŵ_i(t) * weight_i
    pitch_1d = np.dot(weights, cwt_spec)
    return pitch_1d


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_mean")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_std")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        
        energy_scaler = StandardScaler()
        pitch_scaler = None

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)

                # CWT: pitch is 2D (num_scales, T); compute mean/std per scale across dataset
                if pitch is not None and len(pitch) > 0 and getattr(pitch, "ndim", 1) == 2:
                    cwt_spec = pitch
                    num_scales = cwt_spec.shape[0]
                    if pitch_scaler is None:
                        pitch_scaler = [StandardScaler() for _ in range(num_scales)]
                    for s in range(num_scales):
                        pitch_scaler[s].partial_fit(cwt_spec[s, :].reshape(-1, 1))
                elif pitch is not None and len(pitch) > 0 and getattr(pitch, "ndim", 1) == 1:
                    if pitch_scaler is None:
                        pitch_scaler = StandardScaler()
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))

                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # CWT: normalize CWT spectrograms per scale (mean/std per scale)
        use_cwt = self.config["preprocessing"]["pitch"].get("use_cwt", True)
        if use_cwt and pitch_scaler is not None and isinstance(pitch_scaler, list):
            pitch_means = [pitch_scaler[s].mean_[0] for s in range(len(pitch_scaler))]
            pitch_stds = [pitch_scaler[s].scale_[0] for s in range(len(pitch_scaler))]
            pitch_min, pitch_max = self.normalize_cwt(
                os.path.join(self.out_dir, "pitch"), pitch_means, pitch_stds
            )
            pitch_mean = 0.0
            pitch_std = 1.0
        elif not use_cwt or pitch_scaler is None:
            pitch_min, pitch_max, pitch_mean, pitch_std = 0.0, 1.0, 0.0, 1.0
        else:
            if self.pitch_normalization and pitch_scaler is not None:
                pitch_mean = pitch_scaler.mean_[0]
                pitch_std = pitch_scaler.scale_[0]
            else:
                pitch_mean = 0
                pitch_std = 1
            pitch_min, pitch_max = self.normalize(
                os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
            )
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        stats = {
            "pitch": [
                float(pitch_min),
                float(pitch_max),
                float(pitch_mean),
                float(pitch_std),
            ],
            "energy": [
                float(energy_min),
                float(energy_max),
                float(energy_mean),
                float(energy_std),
            ],
        }
        if use_cwt and pitch_scaler is not None and isinstance(pitch_scaler, list):
            stats["pitch_cwt_scale_means"] = [float(pitch_scaler[s].mean_[0]) for s in range(len(pitch_scaler))]
            stats["pitch_cwt_scale_stds"] = [float(pitch_scaler[s].scale_[0]) for s in range(len(pitch_scaler))]
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate) #f0

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None
        
        ### (ourcode) adjusted to apply cwt ourcode
        # Interpolate unvoiced regions
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(nonzero_ids, pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]), bounds_error=False)
        pitch = interp_fn(np.arange(len(pitch)))
        
        # Log scale
        pitch = np.log(pitch + 1e-6)
        
        # Utterance-level normalization
        pitch_mean = np.mean(pitch)
        pitch_std = np.std(pitch)
        pitch_norm = (pitch - pitch_mean) / (pitch_std + 1e-6)

        # save per-utterance mean/std for CWT reconstruction (paper: denormalize after inverse CWT)
        np.save(os.path.join(self.out_dir, "pitch_mean", "{}-{}-mean.npy".format(speaker, basename)), pitch_mean)
        np.save(os.path.join(self.out_dir, "pitch_std", "{}-{}-std.npy".format(speaker, basename)), pitch_std)
        
        # 4) Convert to CWT spectrogram (paper: fixed number of scales, e.g. 10)
        # pycwt.cwt(signal, dt, ..., freqs) returns (wave, sj, freqs, coi, fft, fftfreqs)
        cwt_num_scales = self.config["preprocessing"]["pitch"].get("cwt_num_scales", 10)
        scales = np.arange(1, cwt_num_scales + 1, dtype=np.float64)
        dt = 1.0  # pitch is per-frame
        mother = cwt.Morlet(6)
        freqs = 1.0 / (mother.flambda() * scales)
        wave, _, _, _, _, _ = cwt.cwt(pitch_norm, dt, freqs=freqs, wavelet=mother)
        cwt_spec = np.real(wave)  # real coefficients for downstream (outlier removal, save)
        cwt_spec = self.remove_outlier_cwt(cwt_spec)

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        energy = self.remove_outlier(energy)

        # if self.pitch_phoneme_averaging:
        #     # perform linear interpolation
        #     nonzero_ids = np.where(pitch != 0)[0]
        #     interp_fn = interp1d(
        #         nonzero_ids,
        #         pitch[nonzero_ids],
        #         fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        #         bounds_error=False,
        #     )
        #     pitch = interp_fn(np.arange(0, len(pitch)))

        #     # Phoneme-level average
        #     pos = 0
        #     for i, d in enumerate(duration):
        #         if d > 0:
        #             pitch[i] = np.mean(pitch[pos : pos + d])
        #         else:
        #             pitch[i] = 0
        #         pos += d
        #     pitch = pitch[: len(duration)]
        
        ### (ourcode) adjusted to apply cwt ourcode 
        if self.pitch_phoneme_averaging:
            # Phoneme-level average across time dimension
            pos = 0
            cwt_ph = []

            for d in duration:
                if d > 0:
                    cwt_ph.append(np.mean(cwt_spec[:, pos:pos+d], axis=1))
                else:
                    cwt_ph.append(np.zeros(cwt_spec.shape[0]))
                pos += d

            cwt_spec = np.stack(cwt_ph, axis=1)
            # shape: (num_scales, num_phonemes)

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), cwt_spec)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            cwt_spec,
            energy,
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)

        return np.clip(values, lower, upper)
    
    ### (ourcode)
    def remove_outlier_cwt(self, cwt_spec):
        """
        Remove outliers per scale from a 2D (or 3D) CWT array.
        Input:
            cwt_spec: np.array of shape (num_scales, T) or (num_scales, T, ...)
        Output:
            cleaned_cwt: np.array, same shape but outliers replaced with nearest bounds
        """
        cwt_spec = np.array(cwt_spec, copy=True)
        num_scales = cwt_spec.shape[0]

        for s in range(num_scales):
            scale_values = cwt_spec[s, ...]  # 1D or 2D per scale
            flat = np.asarray(scale_values).ravel()
            if flat.size == 0:
                continue
            p25 = np.percentile(flat, 25)
            p75 = np.percentile(flat, 75)
            lower = p25 - 1.5 * (p75 - p25)
            upper = p75 + 1.5 * (p75 - p25)

            # Clip outliers instead of removing frames (preserves shape)
            scale_values = np.clip(scale_values, lower, upper)
            cwt_spec[s, ...] = scale_values

        return cwt_spec
 

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

    def normalize_cwt(self, in_dir, means_per_scale, stds_per_scale):
        """Normalize CWT spectrograms per scale: (x[s,:] - mean[s]) / (std[s] + 1e-6)."""
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        num_scales = len(means_per_scale)
        for fn in os.listdir(in_dir):
            filepath = os.path.join(in_dir, fn)
            values = np.load(filepath)  # (num_scales, T)
            for s in range(min(num_scales, values.shape[0])):
                values[s, :] = (values[s, :] - means_per_scale[s]) / (stds_per_scale[s] + 1e-6)
            np.save(filepath, values)
            max_value = max(max_value, np.max(values))
            min_value = min(min_value, np.min(values))
        return min_value, max_value
