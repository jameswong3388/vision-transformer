import torch
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.dataset import WayangKulitDataModule
from src.models.basic import ViT

BASE_DIR = Path(__file__).parent
LIGHTNING_DIR = BASE_DIR.joinpath("lightning")
MODELS_DIR = LIGHTNING_DIR.joinpath("models")

LOG_EVERY_N_STEPS = 50
MAX_EPOCHS = 100

BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
PATCH_SIZE = 4

SIZE = PATCH_SIZE * PATCH_SIZE * 3  # 4 * 4 * 3 (RGB colors)
HIDDEN_SIZE = 512
NUM_PATCHES = int(32 * 32 / PATCH_SIZE ** 2)  # 32 x 32 is the size of image in CIFAR10

NUM_HEADS = 8
NUM_ENCODERS = 4

DROPOUT = 0.1
EMB_DROPOUT = 0.16

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6

torch.set_float32_matmul_precision('medium')

# TODO: update the code in the tutorial
# TODO: update train paragraph in the tutorial: text, code, and images

if __name__ == '__main__':
    data = WayangKulitDataModule(batch_size=BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE, patch_size=PATCH_SIZE)

    model = ViT(
        size=SIZE,
        hidden_size=HIDDEN_SIZE,
        num_patches=NUM_PATCHES,
        num_classes=data.classes,
        num_heads=NUM_HEADS,
        num_encoders=NUM_ENCODERS,
        emb_dropout=EMB_DROPOUT,
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=MAX_EPOCHS
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODELS_DIR,
        monitor="val_loss",
        save_last=True,
        verbose=True
    )
    es = EarlyStopping(monitor="val_loss", mode="min", patience=16)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        accelerator="mps",
        default_root_dir=LIGHTNING_DIR,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, es, lr_monitor],
        resume_from_checkpoint=MODELS_DIR.joinpath("last.ckpt") # deprecated
    )
    trainer.fit(
        model,
        data,
        # ckpt_path=str(MODELS_DIR.joinpath("last.ckpt"))  # to resume training on a model
    )

