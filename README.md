# NNGameCopier

## Datasets

The datasets and models are available [here](https://fhooe-my.sharepoint.com/:f:/g/personal/s2310454029_fhooe_at/Es5zNaKj9ONHjDIaK1de8jUBMK3a_dK2666vU4dMnZIwkQ?e=4sRE70).

## How to use

### Prerequisites

Download a dataset and run the following command to compress and normalize the dataset:

```bash
cd src
python file_compressor.py
```

If you just want to play, download the according models (**encoder**, **decoder**, **lstm**) and place them in the `src/models` folder.

Also make sure the python packages are installed using the `requirements_*.txt` files.

### Train the VAE

Make sure to fix the environment issues, if any arise, or open the project wholely in PyCharm

```bash
cd src/training
python train_auto_encoder.py
```

Verify with:

```bash
cd src/testing
python test_auto_encoder.py
```


### Train the LSTM

Analog to the auto encoder. 

```bash
cd src/training
python train_lstm.py
```

Verify with:

```bash
cd src/testing
python test_lstm.py
```

### Play the game

```bash
cd src
python game.py
```

If on linux, make sure not to use Wayland (that cost me a few hours)
