import os
import json

import torch
import numpy as np

from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def _load_parallel_wavegan(device, ckpt_path=None, tag="ljspeech_parallel_wavegan.v1"):
    """Load a kan-bayashi pretrained Parallel WaveGAN generator.

    If ckpt_path is provided, loads directly from that local .pkl file.
    Otherwise attempts to download via the parallel-wavegan library.

    Args:
        device: torch.device to load the model onto.
        ckpt_path: Optional path to a local checkpoint .pkl file.
                   E.g. "parallel_wavegan/checkpoint-400000steps.pkl"
        tag: Pretrained tag used only if ckpt_path is not given.
    """
    try:
        from parallel_wavegan.utils import load_model as pwg_load_model
    except ImportError:
        raise ImportError(
            "parallel-wavegan is required.\n"
            "Install it with: pip install parallel-wavegan"
        )

    if ckpt_path is not None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Parallel WaveGAN checkpoint not found at '{ckpt_path}'. "
                "Make sure the .pkl file is placed there."
            )
        print(f"[ParallelWaveGAN] Loading from local checkpoint: {ckpt_path}")
    else:
        # Fall back to library download
        try:
            from parallel_wavegan.utils import download_pretrained_model
            ckpt_path = download_pretrained_model(tag)
        except Exception as e:
            raise RuntimeError(
                f"Automatic download failed: {e}\n"
                "Download the checkpoint manually and set 'ckpt_path' in model.yaml.\n"
                "E.g.:\n"
                "  vocoder:\n"
                "    model: \"Parallel-WaveGAN\"\n"
                "    speaker: \"LJSpeech\"\n"
                "    ckpt_path: \"parallel_wavegan/checkpoint-400000steps.pkl\""
            )

    vocoder = pwg_load_model(ckpt_path)
    vocoder.remove_weight_norm()
    vocoder.eval()
    vocoder.to(device)
    return vocoder


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "Parallel-WaveGAN":
        ckpt_path = config["vocoder"].get("ckpt_path", None)
        tag = config["vocoder"].get("tag", "ljspeech_parallel_wavegan.v1")
        vocoder = _load_parallel_wavegan(device, ckpt_path=ckpt_path, tag=tag)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "Parallel-WaveGAN":
            # 1. ln -> log10 (ming uses ln, kan-bayashi PWG trained on log10)
            mels_cpu = mels.to("cpu") / np.log(10)
            # 2. Apply per-channel mean/scale normalization from checkpoint
            voc_cpu = vocoder.to("cpu")
            if hasattr(voc_cpu, "mean") and hasattr(voc_cpu, "scale"):
                mean = voc_cpu.mean.clone().detach().float().view(1, -1, 1)
                scale = voc_cpu.scale.clone().detach().float().view(1, -1, 1)
                mels_cpu = (mels_cpu - mean) / scale
            c_up = voc_cpu.upsample_net(mels_cpu)
            T_wav = c_up.shape[-1]
            noise = torch.randn(mels.shape[0], 1, T_wav)
            wavs = voc_cpu(noise, mels_cpu).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs