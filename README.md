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

#### ROCm

For **training** purposes, the ROCm docker container can be used.

Use the following docker command in the current directory (`GameCaptcha`) to start the container correctly:

```bash
sudo docker run -it -v ${PWD}:/work --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined rocm/tensorflow:latest
```
Then run all necessary scripts from the `/work`-directory inside the started container. This container can be reused.

To install all required packages run the following command:

```bash
pip install -r requirements_rocm.txt
```

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

> Make sure to do this in an environment, where windows can be created! The `requirements_game.txt`-file could be helpful here. If on linux, make sure not to use Wayland (that cost me a few hours).


```bash
cd src
python game.py
```

