import os
import json
import yaml

import torch
import numpy as np

import hifigan
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
        model.load_state_dict(ckpt["model"])

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
            hifigan_cfg = json.load(f)
        hifigan_cfg = hifigan.AttrDict(hifigan_cfg)
        vocoder = hifigan.Generator(hifigan_cfg)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=torch.device('cpu'))
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "Parallel WaveGAN" or name == "PWG":
        try:
            from parallel_wavegan.utils import load_model
        except ImportError:
            raise ImportError(
                "Parallel WaveGAN requires the parallel-wavegan package. "
                "Install with: pip install parallel-wavegan"
            )
        pwg_config_path = config["vocoder"].get("config", "config/pwg_parallel_wavegan.v1.yaml")
        pwg_checkpoint_path = config["vocoder"].get("checkpoint", "pwg/checkpoint-400000steps.pkl")
        with open(pwg_config_path, "r") as f:
            pwg_config = yaml.safe_load(f)
        vocoder = load_model(pwg_checkpoint_path, pwg_config)
        vocoder.remove_weight_norm()
        vocoder.eval()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]

    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
            wavs = wavs.cpu().numpy()
            wavs = [wav for wav in wavs]
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
            wavs = wavs.cpu().numpy()
            wavs = [wav for wav in wavs]
        elif name == "Parallel WaveGAN" or name == "PWG":
            wavs = []
            for i in range(mels.size(0)):
                w = vocoder.inference(c=mels[i : i + 1])
                wavs.append(w.view(-1).cpu().numpy())
        else:
            raise ValueError(f"Unknown vocoder: {name}")

    wavs = [(w * max_wav_value).astype("int16") for w in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
