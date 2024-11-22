# stylistic-gesture
Official repository for the paper Stylistic Co-Speech Gesture Generation: Modeling Personality and Communicative Styles in Virtual Agents.
[![DOI](https://zenodo.org/badge/876027763.svg)](https://doi.org/10.5281/zenodo.14204495)

## Preparing environment

1. Git clone this repo

2. Enter the repo and create docker image using 

```sh
docker build -t stylistic-gesture .
```

3. Run container using

```sh
nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES={GPU} --runtime=nvidia --userns=host --shm-size 64G -v {LOCAL_DIR}:{CONTAINER_DIR} -p {PORT} --name {CONTAINER_NAME} stylistic-gesture:latest /bin/bash
```

for example:
```sh
docker run --rm -it --gpus device=0 --userns=host --shm-size 64G -v C:\ProgramFiles\stylistic-gesture:/workspace/stylistic-gesture -p '8888:8888' --name stylistic-gesture-container stylistic-gesture:latest /bin/bash
```

4. Activate cuda environment:
```sh
source activate stylistic-env
```

## Data pre-processing

1. Get the BRG-Unicamp dataset following the instructions from [here](https://ai-unicamp.github.io/BRG-Unicamp/) and put it into `./dataset/`

2. Download the [WavLM Base +](https://github.com/microsoft/unilm/tree/master/wavlm) and put it into the folder `/wavlm/`

3. In the container with the active environment, enter the folder `/workspace/stylistic-gesture`, run

```sh
python -m data_loaders.gesture.scripts.ptbrgesture_prep
```

This will convert the bvh files to npy representations, downsample wav files to 16k and save them as npy arrays, and convert these arrays to wavlm representations. The VAD data must be processed separetely due to python libraries incompatibility. 

4. (Optional) Process VAD data

BRG-Unicamp provides the speech activity information (from speechbrain's VAD) data, but if you wish to process them yourself you should redo the steps of "Preparing environment" as before, but for the speechbrain environment: Build the image using the Dockerfile inside speechbrain (`docker build -t speechbrain .`), run the container (`docker run ... --name CONTAINER_NAME speechbrain:latest /bin/bash`) and run:

```sh
python -m data_loaders.gesture.scripts.ptbrgesture_prep_vad
```

## Train model

To train the model described in the paper use the following command inside the repo:

```sh
python -m train.train_mdm --save_dir save/my_model_run --dataset ptbr --step 10  --use_vad True --use_wavlm True --use_style_enc True
```

## Gesture Generation

Generate motion using the trained model by running the following command. If you wish to generate gestures with the pretrained model of the Genea Challenge, use `--model_path ./save/stylistic-gesture/model000600000.pt` 

```sh
python -m sample.ptbrgenerate --model_path ./save/my_model_run/model000XXXXXX.pt 
```

## Render

In our perceptual evaluation, we used the render procedure from the official GENEA Challenge 2023 visualizations. Instructions provided [here](https://github.com/TeoNikolov/genea_visualizer/)

## Cite

If you with to cite this repo or the paper
[![DOI](https://zenodo.org/badge/876027763.svg)](https://doi.org/10.5281/zenodo.14204495)
