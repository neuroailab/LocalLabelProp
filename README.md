Codes for [Local Label Propagation for Large-Scale Semi-supervised Learning](https://arxiv.org/abs/1905.11581).

# Prerequisites

Almost all network training is done with python=2.7, but python 3.7 should work too.

`1.9 < tensorflow_gpu < 2.0.0`.

# Instructions for training

## Install `tfutils`.

```
git clone https://github.com/neuroailab/tfutils.git
cd tfutils
pip install -e ./
```
Due to a conflict in the requirements of `tfutils` which will not influence this repo, you need to reinstall `tensorflow_gpu==1.13.1` after installing tfutils 
(any version of tensorflow\_gpu smaller than 2.0.0, bigger than 1.9 should work).
To use tfutils, you need to host a mongodb, please see general instructions about how to do that.
The default port number of the mongodb is 26001, but you can change it through setting environment variable `MG_PORT`.

## Prepare training datasets.

Prepare the ImageNet data as the raw JPEG format used in pytorch ImageNet training (see [link](https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py)).
Then run the following commands:
```
DATA_DIR=/path/to/put/your/data python misc/build_tfrs.py --img_folder /path/to/imagenet/raw/folder
DATA_DIR=/path/to/put/your/data python misc/make_balance_partIN.py
```
The first command will create tfrecords in directory `/path/to/put/your/data/tfrs`. 
The second command will create metas about partial ImageNet in directory `/path/to/put/your/data/metas`.

## Run training.
As this training needs a warm start from another unsupervised learning algorithm, we need to first clone another [repo](https://github.com/neuroailab/LocalAggregation) 
and then run the following command in the repo directory (the dataset is already prepared to be usable by this repo):
```
python train_tfutils.py --config exp_configs/la_final.json:res18_IR --image_dir /path/to/put/your/data/tfrs/ --gpu [your gpu number] --cache_dir [your cache dir] --port [mongodb port number]
```
`cache_dir` is where your model is temporarily placed, the models will also be saved in the mongodb.

After this training is done (this training will only have 10 epochs). Run the following command:
```
DATA_DIR=/path/to/put/your/data python train_tfutils.py --setting llp_res18.p10 --cache_dir [your cache dir] --gpu [your gpu number]
```
Make sure `DATA_DIR` is the same as what is used in previous commands.
This will start a ResNet-18 training with 10% ImageNet images labeled.
Check `saved_settings/llp_res18.py` to see details.

To train a ResNet-50, just change the `exp_configs/la_final.json:res18_IR` in the first command to `exp_configs/la_final.json:res50_IR` and change `llp_res18.p10` in the second command to `llp_res50.p10`.

Using gpu numbers separated by `,` will start multi-gpu training.
