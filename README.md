# nn-denoising

## Usage

#### installing required packages:

    - keras
    - tensorflow-gpu
    - numpy
    - scikit-image
    - tifffile
    - flask (if using web UI)

#### fetching pre-trained model:

Download `denoiser.model` from https://drive.google.com/open?id=1uD3LvmNRTc5clM9x3fkhiQ4X-cedflMc, and place it in the root of the repo

#### example code:

```python
from nn_denoising import predict
predict('./experimental.tif',  log_flag=True)
# or without log
predict('./experimental.tif')
```

### starting as a web service

Enabling web service at a server:

```bash
python ./web.py
```

then visit `http://SERVICE-IP-ADDRESS:4567` to upload and get denoised result.

### license:

AGPL-3.0

### Alternative model

A new pre-model trained with [MCNN](https://arxiv.org/abs/1810.12183) (and GAN) is available in [MCNN-DEMO](https://github.com/fengwang/mcnn-demo/blob/master/demo/denoising/make_denoising.py).


