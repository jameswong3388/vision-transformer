## Install

After you've cloned the repository, you need to install required packages.

### Install for CPU / MPS

Run the following command to install packages:

```shell
pip install -r requirements.txt
```

### Install for CUDA

Install PyTorch libraries with the command from [official web-site](https://pytorch.org/get-started/locally/):

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest of the libraries with the command:

```shell
pip install -r requirements-cuda.txt
```

## View logs
```python
python3 -m tensorboard.main --logdir=lightning/lightning_logs/
```

## Credits
https://github.com/MikhailKravets/vision_transformer