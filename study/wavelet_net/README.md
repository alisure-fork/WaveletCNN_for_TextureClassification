
## Accuracy
batch size = 128

| Model | Acc. |
|:-----:|:----:|
| [VGG16](https://arxiv.org/abs/1409.1556)              | 93.50% |
| [ResNet18](https://arxiv.org/abs/1512.03385)          |        |
| [ResNet50](https://arxiv.org/abs/1512.03385)          |        |
| [ResNet101](https://arxiv.org/abs/1512.03385)         |        |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       |        |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  |        |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  |        |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       |        |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    |        |
| [DPN92](https://arxiv.org/abs/1707.01629)             |        |


## Learning rate adjustment
I manually change the `lr` during training:
    - `0.1` for epoch `[0,150)`
    - `0.01` for epoch `[150,250)`
    - `0.001` for epoch `[250,350)`


## Level

|  id   | level 0 | level 1 | level 2 | level 3 | level 4 |
|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  128  |  94.10  |  94.42  |  94.49  |  94.10  |  94.67  |
|  64   |  93.22  |  93.66  |  93.84  |  93.76  |  93.85  |
|  32   |  92.28  |  92.83  |  92.84  |  92.82  |  92.60  |
|  16   |  89.80  |  90.51  |  91.02  |  90.60  |  90.65  |
|   8   |  85.67  |  87.64  |  87.68  |  87.98  |  88.09  |
|   8   |  85.67  |  87.64  |  87.68  |  87.98  |  88.09  |
|   2   |  64.37  |  68.47  |  71.74  |  74.59  |  75.07  |


|  id   | level 0 | level 1 | level 2 | level 3 | level 4 |
|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  64   |  93.55  |  90.77  |  84.09  |  68.39  |  00.00  |


