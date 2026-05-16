# models/utils/training/checkpointing.py

import os
import torch


def save_episode_checkpoint(model, ckpt_dir, plot, writer, episode, best_loss):
    # 既存仕様を維持するため、保存名は train.py の元コードと同じにする
    model_name = f"{episode}.pth"
    model_path = os.path.join(ckpt_dir, model_name)
    torch.save(model.state_dict(), model_path)

    current_loss = plot.epi_loss_return()

    if current_loss < best_loss:
        best_loss = current_loss
        model_name = "best.pth"
        model_path = os.path.join(ckpt_dir, model_name)
        torch.save(model.state_dict(), model_path)

        writer.write(
            f"New best model at episode {episode + 1}, "
            f"avg_epi_loss={best_loss:.6f}\n"
        )

    return best_loss, model_path