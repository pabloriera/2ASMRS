# Audio Autoencoder for Sound Morphing in Realtime Synthetizer

# Install

Clone with submodules

```
git clone --recursive https://github.com/pabloriera/2ASMRS.git
```

# Model

## Train example:
```console
python run.py
```

## TensorBoard:
```console
tensorboard --logdir=logs
```
## Train on colab

[Colab](https://colab.research.google.com/drive/1VMalSDqbO-idkTdtX47v7qRcLDioZ5GT#scrollTo=4zlUqBjwMCeu)

# Synthetizer:

## Building:

Download the .zip file containing the libtorch (PyTorch C++ API) source and unzip in external folder

```console
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
```

### Linux dependencies

```
sudo apt update
sudo apt install libasound2-dev libjack-jackd2-dev \
    ladspa-sdk \
    libcurl4-openssl-dev  \
    libfreetype6-dev \
    libx11-dev libxcomposite-dev libxcursor-dev libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev libxrender-dev \
    libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev mesa-common-dev
```

### Build 
```console
cmake -Bbuild
cmake --build build --config Release
```

### Run
```console
build/AE_artefacts/Debug/AE test_torchscript.json
```