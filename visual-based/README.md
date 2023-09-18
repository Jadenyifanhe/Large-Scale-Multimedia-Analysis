# CMU 11-775 HW2: Video-based Multimedia Event Detection

Author: Jaden He\
Andrew ID: yifanhe\
Kaggle ID: jadenhe


## Install Dependencies

A set of dependencies is listed in [environment.yml](environment.yml). You can use `conda` to create and activate the environment easily.

```bash
conda env create -f environment.yml
conda activate 11775-hw2
```


## Dataset

You will be using two parts of data for this homework:

* Data from [Homework 1](https://github.com/11775website/11775-hws/tree/master/spring2022/hw1#data-and-labels) which you should have downloaded. [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip).
* A new larger set of test videos. [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data_p2.zip).

Both parts should be decompressed under the `data` directory.
You can directly download them into your AWS virtual machine:

```bash
mkdir data && cd data
# Download and decompress part 1 data (no need if you still have it from HW1)
wget https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip
unzip 11775_s22_data.zip
# Download and decompress part 2 data
wget https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data_p2.zip
unzip 11775_s22_data_p2.zip
```

## Development and Debugging

Some functions in the pipeline are deliberately left blank for you to implement, where an `NotImplementedError` will be raised.
We recommend you generate a small file list (e.g. `debug.csv` with 20 lines) for fast debugging during initial development.
The `--debug` option in some scripts are also very helpful.
In addition, you can enable `pdb` debugger upon exception

```bash
# Instead of 
python xxx.py yyy zzz
# Run
ipython --pdb xxx.py -- yyy zzz
```

## SIFT Features

To extract SIFT features, use

```bash
python code/run_sift.py data/labels/xxx.csv
```

By default, features are stored under `data/sift`. As an estimate for you, it took around 1 hour to run the `train_val` set on a server with 10 hyperthreaded CPU cores when we only select every 20 frames and extract 32 key points from each.

To train K-Means with SIFT feature for 128 clusters, use

```bash
python code/train_kmeans.py data/labels/xxx.csv data/sift 128 sift_128
```

To train K-Means with SIFT feature for 1024 clusters, use

```bash
python code/train_kmeans.py data/labels/xxx.csv data/sift 1024 sift_1024
```

By default, model weights are stored under `data/kmeans`. With 10% of the feature vectors, it took less than 5 minutes to train. You can use more data for a potentially better performance but longer training time.

To extract Bag-of-Words representation with the trained model, use

```bash
python code/run_bow.py data/labels/xxx.csv sift_128 (or sift_1024) data/sift
```

By default, features are stored under `data/bow_<model_name>` (e.g., `data/bow_sift_128`).

## CNN Features

To extract CNN features, use

```bash
python code/run_cnn.py data/labels/xxx.csv --video_dir=data/videos --cnn_dir=data/cnn_resnet18(or cnn_resnet101 or cnn_resnet152)
```

By default, features are stored under `data/cnn`.

The current pipeline processes images one by one, which is not so friendly with GPU.
You can try to optimize it into batch processing for faster speed.

## MLP Classifier

The training script automatically and deterministically split the `train_val` data into training and validation, so you do not need to worry about it.

To train MLP with SIFT Bag-of-Words, run

```bash
python code/run_mlp.py sift --feature_dir data/bow_sift_128 (or bow_sift_1024) --num_features 128 (or 1024) --max_epochs=150 --batch_size=1024
```

To train MLP with CNN features, run

```bash
python code/run_mlp.py cnn --feature_dir data/cnn_resnet18 (or data/cnn_resnet101 or data/cnn_resnet152) --num_features <num_feat> (512 for cnn_resnet18, 2048 for cnn_resnet101 and cnn_resnet152) --max_epochs=150 --batch_size=1024
```

By default, training logs and predictions are stored under `data/mlp/cnn/version_xxx/`.
