import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if len(data) == 13:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            pitch_mean_vars,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        pitch_mean_vars = torch.from_numpy(pitch_mean_vars).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            pitch_mean_vars,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def cwt_to_pitch_1d(
    cwt_spec,
    pitch_mean_var=None,
    cwt_scale_means=None,
    cwt_scale_stds=None,
    return_log=False,
):
    """
    Convert CWT pitch spectrogram to 1D pitch contour (paper: iCWT then denormalize).
    Preprocessing used: log(pitch+1e-6), norm = (log - mean) / std, then CWT.
    So we: (optional) denorm per-scale, iCWT, then denorm with mean/var, then exp.
    Args:
        cwt_spec: (T, num_scales) or (num_scales, T)
        pitch_mean_var: (2,) or (B, 2) [mean, std] for utterance denorm (log domain)
        cwt_scale_means: list of length num_scales (reverse per-scale norm)
        cwt_scale_stds: list of length num_scales
        return_log: if True return log F0, else linear F0
    Returns:
        1D pitch (T,) linear F0 in Hz (or log F0 if return_log=True)
    """
    from preprocessor.preprocessor import icwt

    cwt_spec = np.asarray(cwt_spec, dtype=np.float64)
    if cwt_spec.ndim != 2:
        raise ValueError("cwt_spec must be 2D")
    # (T, num_scales) -> (num_scales, T) for icwt
    if cwt_spec.shape[0] < cwt_spec.shape[1]:
        cwt_spec = cwt_spec.T
    num_scales, T = cwt_spec.shape

    if cwt_scale_means is not None and cwt_scale_stds is not None:
        cwt_scale_means = np.asarray(cwt_scale_means, dtype=np.float64)
        cwt_scale_stds = np.asarray(cwt_scale_stds, dtype=np.float64)
        for s in range(min(num_scales, len(cwt_scale_means))):
            cwt_spec[s, :] = cwt_spec[s, :] * (cwt_scale_stds[s] + 1e-6) + cwt_scale_means[s]

    pitch_norm_1d = icwt(cwt_spec, num_scales=num_scales)

    if pitch_mean_var is not None:
        pitch_mean_var = np.asarray(pitch_mean_var, dtype=np.float64).flatten()
        mean_, std_ = pitch_mean_var[0], pitch_mean_var[1] + 1e-6
        pitch_log = pitch_norm_1d * std_ + mean_
    else:
        pitch_log = pitch_norm_1d

    if return_log:
        return pitch_log
    return np.exp(pitch_log) - 1e-6


def _cwt_to_1d_for_plot(pitch_cwt, duration, stats=None, pitch_mean_var=None, use_icwt=True):
    """
    Convert CWT pitch to 1D for plotting. If use_icwt and stats (with pitch_cwt_scale_means/stds)
    and optional pitch_mean_var are provided, use proper iCWT + denormalization (paper).
    Otherwise fall back to mean across scales.
    """
    p = np.asarray(pitch_cwt)
    if p.ndim == 2 and use_icwt and stats is not None and "pitch_cwt_scale_means" in stats:
        cwt_means = stats.get("pitch_cwt_scale_means")
        cwt_stds = stats.get("pitch_cwt_scale_stds")
        if cwt_means is not None and cwt_stds is not None:
            p = cwt_to_pitch_1d(
                p,
                pitch_mean_var=pitch_mean_var,
                cwt_scale_means=cwt_means,
                cwt_scale_stds=cwt_stds,
                return_log=False,
            )
    elif p.ndim == 2:
        p = p.mean(axis=-1)
    if duration is not None and len(duration) == len(p):
        p = expand(p, duration)
    return p


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    # predictions: 0=mel, 1=postnet_mel, 2=pitch_cwt, 3=pitch_mean_var, 4=energy, 5=log_d, 6=d_rounded, 7=src_masks, 8=mel_masks, 9=src_lens, 10=mel_lens
    src_len = predictions[9][0].item()
    mel_len = predictions[10][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[12][0, :src_len].detach().cpu().numpy()
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats_full = json.load(f)
    stats = stats_full["pitch"] + stats_full["energy"][:2]
    pitch_mean_var_target = targets[10][0].detach().cpu().numpy() if len(targets) > 10 else None
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
        pitch = _cwt_to_1d_for_plot(pitch, duration, stats=stats_full, pitch_mean_var=pitch_mean_var_target, use_icwt=True)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
        pitch = _cwt_to_1d_for_plot(pitch, None, stats=stats_full, pitch_mean_var=pitch_mean_var_target, use_icwt=True)
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[11][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[11][0, :mel_len].detach().cpu().numpy()

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats_full = json.load(f)
    stats_plot = stats_full["pitch"] + stats_full["energy"][:2]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[9][i].item()
        mel_len = predictions[10][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[6][i, :src_len].detach().cpu().numpy()
        pitch_mean_var_i = predictions[3][i].detach().cpu().numpy() if predictions[3] is not None else None
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = _cwt_to_1d_for_plot(pitch, duration, stats=stats_full, pitch_mean_var=pitch_mean_var_i, use_icwt=True)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
            pitch = _cwt_to_1d_for_plot(pitch, None, stats=stats_full, pitch_mean_var=pitch_mean_var_i, use_icwt=True)
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[4][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[4][i, :mel_len].detach().cpu().numpy()

        stats = stats_plot

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[10] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = np.asarray(pitch)
        if pitch.ndim == 2:
            pitch = pitch.mean(axis=-1)
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
