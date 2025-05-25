# EGANet


Efficient Global Attention Network via Multi-Granularity Dynamic Convolution for Lightweight Image Super-Resolution

Qian Wang, Jing Wei, Mengyang Wang, Yao Tang, Han Pan

## 💻Environment

- [PyTorch >= 1.9](https://pytorch.org/)
- [Python 3.7](https://www.python.org/downloads/)
- [Numpy](https://numpy.org/)
- [BasicSR >= 1.3.4.9](https://github.com/XPixelGroup/BasicSR)

## 🔧Installation

```python
pip install -r requirements.txt
```

## 📜Data Preparation

The trainset uses the DIV2K (800). In order to effectively improve the training speed, images are cropped to 480 * 480 images by running script extract_subimages.py, and the dataloader will further randomly crop the images to the GT_size required for training. GT_size defaults to 128/192/256 (×2/×3/×4). 

```python
python extract_subimages.py
```

The input and output paths of cropped pictures can be modify in this script. Default location: ./datasets/DIV2K.

## 🚀Train

▶️ You can change the training strategy by modifying the configuration file. The default configuration files are included in ./options/train/EGANet. Take one GPU as the example.

```python
### Train ###
### EGANet ###

python train.py -opt ./options/train/EGANet/train_eganet_x4.yml --auto_resume  # ×4
```

For more training commands, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)

## 🚀Test

▶️ You can modify the configuration file about the test, which is located in ./options/test/EGANet. At the same time, you can change the benchmark datasets and modify the path of the pre-train model. 

▶️ We have uploaded the pre-train weights for EGANet.Please change the path to the corresponding configuration file when testing.


```python
### Test ###
### EGANet for Lightweight Image Super-Resolution ###
python basicsr/test.py -opt ./options/test/EGANet/test_eganet_x4.yml  # ×4

### EGANet for Large Image Super-Resolution ###
### Flicker2K  Test2K  Test4K  Test8K ###
python basicsr/test.py -opt ./options/test/EGANet/test_act_large.yml  # large image

```

## 🚩Results

The inference results on benchmark datasets will be available at [Google Drive](https://drive.google.com/file/d).

## :mailbox:Contact

If you have any questions, please feel free to contact us wqabby@xupt.edu.cn and [bolttt@stu.xupt.edu.cn](mailto:bolttt@stu.xupt.edu.cn).
