# weightnet.pytorch
The unoffical PyTorch implementation of [WeightNet](https://arxiv.org/abs/2007.11823), based on [pycls](https://github.com/facebookresearch/pycls).

## Installation
See [`INSTALL.md`](docs/INSTALL.md) and [`GETTING_STARTED.md`](docs/GETTING_STARTED.md). Learn more at [pycls's documentation](docs/README.md).

## Experiments

> **_NOTE:_** We use the pre-trained [ShuffleNet V2 1.0x](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2) as the default backbone.

```bash
python tools/train_net.py --cfg configs/shufflenet/ShuffleNet_1x_imagenet_4gpu.yaml
# folder: output/ShuffleNet_1x_imagenet_4gpu
# result: top1: 	top5: 
python tools/train_net.py --cfg configs/shufflenet/WeightNet_1x_imagenet_4gpu.yaml
# folder: output/WeightNet_1x_imagenet_4gpu
# result: top1: 	top5: 
```

## Acknowledgement

- [pycls](https://github.com/facebookresearch/pycls)
- [WeightNet](https://github.com/megvii-model/WeightNet)
- [ShuffleNet-Series](https://github.com/megvii-model/ShuffleNet-Series)