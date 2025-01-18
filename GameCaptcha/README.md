# GameCaptcha

A Deep-Learning project by Tobias Kneidinger, Florian Kreuzer and David Matousch

## Prerequisites

### ROCm

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

## How to run

### How to train

1) Make sure you have the dataset in the `src/captured_frames` folder and that you have created the folder `src/models`.

2) Run the frame compressor:

```bash
python file_compressor.py
```

3) Train the auto encoder:

```bash
python train_auto_encoder.py
```

4) Train the frame predictor:

```bash
python train_auto_encoder.py
```

5) Run the game:
> Make sure to do this in an environment, where windows can be created! The `requirements_game.txt`-file could be helpful here.

```bash
python game.py
```

### Games

#### Snake

https://playsnake.org/