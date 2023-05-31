# Synthetizer:

Download the .zip file containing the libtorch (PyTorch C++ API) source and unzip in external folder

```console
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip -d external/ 
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
cmake -Bbuild # or cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Run
```console
./build/2ASRMS_artefacts/Release/2ASRMS test_torchscript.json
```