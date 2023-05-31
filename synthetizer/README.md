# Synthetizer:

Download the .zip file containing the libtorch (PyTorch C++ API) source and unzip in external folder

```console
git submodule init
git submodule update
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
```

Manually adding submodules:
```console
git submodule init
git add submodule https://github.com/nlohmann/json.git synthetizer/external/json 
git add submodule https://github.com/juce-framework/JUCE.git synthetizer/external/JUCE
```


## Linux dependencies

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

## Build 
```console
cmake -Bbuild
cmake --build build --config Release
```

## Run
```console
build/AE_artefacts/Debug/AE test_torchscript.json
```