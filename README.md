# Paper
Wang, Jingyao, Issam Falih, and Emmanuel Bergeret. "Skeleton-Based Action Recognition with Spatial-Structural Graph Convolution." 2024 International Joint Conference on Neural Networks (IJCNN). IEEE, 2024.
link: https://arxiv.org/pdf/2407.21525

# Prerequisites
## Libraries
This code is based on Python3 (anaconda, >= 3.5) and PyTorch (>= 1.6.0).

Other Python libraries are presented in the 'scripts/requirements.txt', which can be installed by

pip install -r scripts/requirements.txt

## Experimental Dataset
Our models are experimented on the NTU RGB+D 60 & 120 datasets.
There are 302 samples of NTU RGB+D 60 and 532 samples of NTU RGB+D 120 need to be ignored, which are shown in the 'src/reader/ignore.txt'.

# Parameters
Before training and evaluating, there are some parameters should be noticed. 

(1) '--config' or '-c': The configs. You must use this parameter in the command line or the program will output an error. There are some configs given in the configs folder, which can be illustrated in the following tabel. 
config	2001	2002	2003	2004
model	B0	B0	B0	B0
benchmark	X-sub	X-view	X-sub120	X-set120

(2) '--work_dir' or '-w': The path to workdir, for saving checkpoints and other running files. Default is './workdir'.

(3) '--pretrained_path' or '-pp': The path to the pretrained models. pretrained_path = None means using randomly initial model. Default is None.

(4) '--resume' or '-r': Resume from the recent checkpoint ('<--work_dir>/checkpoint.pth.tar').

(5) '--evaluate' or '-e': Only evaluate models. You can choose the evaluating model according to the instructions.

(6) '--extract' or '-ex': Extract features from a trained model for visualization. Using this parameter will make a data file named 'extraction_<--config>.npz' at the './visualization' folder.

(7) '--visualization' or '-v': Show the information and details of a trained model. You should extract features by using <--extract> parameter before visualizing.

(8) '--dataset' or '-d': Choose the dataset. (Choice: [ntu-xsub, ntu-xview, ntu-xsub120, ntu-xset120])

(9) '--model_type' or '-mt': Choose the model. (Format: SpSt-GCN-B{coefficient}, e.g., SpSt-GCN-B0, SpSt-GCN-B2, SpSt-GCN-B4)


# Running
## Modify Configs
Firstly, you should modify the 'path' parameters in all config files of the 'configs' folder.

A python file 'scripts/modify_configs.py' will help you to do this. You need only to change three parameters in this file to your path to NTU datasets.

python scripts/modify_configs.py --path <path/to/save/numpy/data> --ntu60_path <path/to/ntu60/dataset> --ntu120_path <path/to/ntu120/dataset> --pretrained_path <path/to/save/pretraiined/model> --work_dir <path/to/work/dir>
All the commands above are optional.

## Generate Datasets
After modifing the path to datasets, please generate numpy datasets by using 'scripts/auto_gen_data.sh'.

```
bash scripts/auto_gen_data.sh
```

It may takes you several days, due to the calculation of dymanic edges.
```
python main.py -c <config> -gd
```
where <config> is the config file name in the 'configs' folder, e.g., 2001_B.

## Train
You can simply train the model by
```
python main.py -c <config>
```
If you want to restart training from the saved checkpoint last time, you can run
```
python main.py -c <config> -r
```
## Merge the results of multiple branches
```
python fusion.py
```
## Evaluate
Before evaluating, you should ensure that the trained model corresponding the config is already existed in the <--pretrained_path> or '<--work_dir>' folder. Then run
```
python main.py -c <config> -e
```
## Visualization
To visualize the details of the trained model, you can run
```
python main.py -c <config> -ex -v
```
where '-ex' can be removed if the data file 'extraction_<config>.npz' already exists in the './visualization' folder.
