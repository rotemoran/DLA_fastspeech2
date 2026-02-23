import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        """
        inputs: ground-truth tensors
            mel_targets, mel_lengths, ...
            pitch_targets (B, T, num_scales)
            pitch_mean_var_target (B, 2)
            energy_targets (B, T)
            duration_targets (B, T)
        predictions: model outputs
            mel_predictions, postnet_mel_predictions,
            pitch_predictions (B, T, num_scales),
            pitch_mean_var (B, 2),
            energy_predictions, log_duration_predictions,
            duration_rounded, mel_len, mel_mask
        """

        (
            mel_targets,
            _,
            _,
            pitch_targets,
            pitch_mean_var_target,
            energy_targets,
            duration_targets,
        ) = inputs[6:]  # adjust if your input ordering differs

        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            pitch_mean_var,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions

        # Invert masks for masked_select
        src_masks = ~src_masks
        mel_masks = ~mel_masks

        # Log-scale duration targets
        log_duration_targets = torch.log(duration_targets.float() + 1)

        # Truncate mel to match predicted length
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, : mel_masks.shape[1]]

        # Freeze targets
        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        pitch_mean_var_target.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        # ------------------------
        # Pitch loss (CWT): targets are always (B, mel_len, num_scales) after dataset expansion
        # ------------------------
        mask_for_pitch = mel_masks.unsqueeze(-1)  # (B, mel_len, 1)

        # Flatten batch and time but keep num_scales
        pitch_loss_spec = self.mse_loss(
            pitch_predictions.masked_select(mask_for_pitch).view(-1, pitch_predictions.shape[-1]),
            pitch_targets.masked_select(mask_for_pitch).view(-1, pitch_targets.shape[-1])
        )

        # Utterance-level mean/variance loss
        pitch_stat_loss = self.mse_loss(pitch_mean_var, pitch_mean_var_target)

        # ------------------------
        # Energy loss
        # ------------------------
        if self.energy_feature_level == "phoneme_level":
            energy_pred_masked = energy_predictions.masked_select(src_masks)
            energy_target_masked = energy_targets.masked_select(src_masks)
        else:  # frame_level
            energy_pred_masked = energy_predictions.masked_select(mel_masks)
            energy_target_masked = energy_targets.masked_select(mel_masks)

        energy_loss = self.mse_loss(energy_pred_masked, energy_target_masked)

        # ------------------------
        # Duration loss
        # ------------------------
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # ------------------------
        # Mel loss
        # ------------------------
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        # ------------------------
        # Total loss
        # ------------------------
        total_loss = (
            mel_loss
            + postnet_mel_loss
            + duration_loss
            + energy_loss
            + pitch_loss_spec
            + pitch_stat_loss
        )

        pitch_loss = pitch_loss_spec + pitch_stat_loss
        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
