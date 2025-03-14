MCArb HSI SR
====  
This is a code demo for the paper ["Meta-Collaborative Learning for Arbitrarily Scaled Hyperspectral Image Super-Resolution"](https://ieeexplore.ieee.org/document/10902151). 

Brief introduction:
-------  
Deep learning-based methods for hyperspectral image super-resolution (SR) have achieved significant success in recent years. These methods typically consist of feature extraction module and upsampling module. However, due to structural limitations of the upsampling module, most current methods focus on training separate models for different scale factors, which ignores the exploration of potential feature interdependence among different scale factors. In response to these challenges, we introduce a novel framework, called “Meta-Collaborative Learning for Arbitrarily Scaled Hyperspectral Image Super-Resolution” (MCArb). Specifically, MCArb integrates a collaborative learning framework with a meta-learning-based 3D upsampling module (3DMetaUM) and a scale-aware feature adaptation module (SAFAM). It enables training multiple SR tasks at different scale factors within a single network at the same time. This strategy is able not only to process arbitrary scale factor SR for hyperspectral images but also to harness the latent feature interdependence among different scales. In this study, we applied the MCArb framework to transform three deep learning-based hyperspectral image SR networks to MCArb-methods, resulting in significant performance enhancements across five hyperspectral datasets. These improvements showcase the proposed MCArb framework’s ability to enhance feature extraction efficiency and to capitalize on latent inter-scale correlations. 

Requirements:
-------  

Dataset:
-------  
[Pavia Centre](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene)

[Chikusei](https://www.sal.t.u-tokyo.ac.jp/hyperdata/)

[Houston University](https://hyperspectral.ee.uh.edu/?page_id=1075)

[CAVE](https://cave.cs.columbia.edu/projects/categories/project?cid=Computational+Imaging&pid=Generalized+Assorted+Pixel+Camera)

[Harvard](https://vision.seas.harvard.edu/hyperspec/)

Usage:
-------  

Citation 
-------  

**Please consider cite MCArb for arbitrarily scaled hyperspectral image super-resolution if helpful.**

```
@ARTICLE{10902151,
  author={Zhang, Mingyang and Wu, Shuang and Wang, Xiangyu and Gong, Maoguo and Jiang, Fenlong and Zhou, Yu and Wu, Yue},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Meta-Collaborative Learning for Arbitrarily Scaled Hyperspectral Image Super-Resolution}, 
  year={2025},
  volume={63},
  number={},
  pages={1-18},
  keywords={Feature extraction;Hyperspectral imaging;Federated learning;Training;Image reconstruction;Superresolution;Spatial resolution;Metalearning;Optimization;Adaptation models;Collaborative learning;hyperspectral image;meta-learning;scale arbitrary;super-resolution (SR)},
  doi={10.1109/TGRS.2025.3544253}}
```

