# SphereNet-pytorch
This is an unofficial implementation of ECCV 18 [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Coors_SphereNet_Learning_Spherical_ECCV_2018_paper.pdf) **"SphereNet: Learning Spherical Representations for Detection and Classification in Omnidirectional Images"**.

Currently only 3x3 SphereNet's Conv2D and MaxPool2D are implemented. For now only `mode=bilinear` is allowed, `mode=nearest` have to wait until Pytorch1.0. You can replace any model's CNN with SphereNet's CNN, they are implemented such that you can directly load pretrained weight to SphereNet's CNN.

## Requirements
- python3
- pytorch>=0.4.1
- numpy
- scipy

## Installation
Copy spherenet directory to your project.  
If you want to install as an package such that you can import spherenet everywhere:
``` 
cd $YOUR_CLONED_SPHERENET_DIR
pip install .
```

## Example
``` python
from spherenet import SphereConv2D, SphereMaxPool2D

conv1 = SphereConv2D(1, 32, stride=1)
pool1 = SphereMaxPool2D(stride=2)

# toy example
img = torch.randn(1, 1, 60, 60)  # (batch, channel, height, weight)
out = conv1(img)  # (1, 32, 60, 60)
out = pool1(out)  # (1, 32, 30, 30)
```
- To apply SphereNet in your trained model, simply replace the ```nn.Conv2d``` with ```SphereConv2D```, and replace ```nn.MaxPool2d``` with ```SphereMaxPool2D```. They should work well with `load_state_dict`.

## Results
- Classification OminiMNIST data (`spherenet.OmniMNIST`, `spherenet.OmniFashionMNIST`)
    - <img src="https://imgur.com/WvEZM2V.png" height="150" />
- Reproduce OmniMNIST Result
    - | Method        | Test Error (%) |
      | ------------- |:--------------:|
      | SphereNet ( paper )     | 5.59 |
      | SphereNet ( ours ) | 5.77 |
      | EquirectCNN ( paper )   | 9.61 |
      | EquirectCNN ( ours )| 9.63 |
    - <img src="https://i.imgur.com/BKnGqf1.png" width="400" />
    
## References
- [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Coors_SphereNet_Learning_Spherical_ECCV_2018_paper.pdf)
    - Benjamin Coors, Alexandru Paul Condurache, Andreas Geiger
    - ECCV2018
      ```
        @inproceedings{coors2018spherenet,
          title={SphereNet: Learning Spherical Representations for Detection and Classification in Omnidirectional Images},
          author={Coors, Benjamin and Condurache, Alexandru Paul and Geiger, Andreas},
          booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
          pages={518--533},
          year={2018}
        }
      ```
