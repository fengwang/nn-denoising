# nn-denoising

#### installing required packages:

    - keras
    - numpy
    - scikit-image
    - tifffile

#### fetching pre-trained model:

Download `denoiser.model` from https://drive.google.com/open?id=1uD3LvmNRTc5clM9x3fkhiQ4X-cedflMc, and place it in the root of the folder

#### example usage:

```python
predict('./experimental.tif',  log_flag=True)
# or without log
predict('./experimental.tif')
```

#### license:

AGPL-3.0

