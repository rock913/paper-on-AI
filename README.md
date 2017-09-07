# Paper-on-AI

## Topics

### Probabilistic programming language ([Note](doc/ppl.md))
 - Human-level concept learning through probabilistic program induction ([Note](doc/ppl_human_level_concep_learning.md))
 - Picture: A Probabilistic Programming Language for Scene Perception ([Note](doc/Picture_PPL_for_Scene_Perception.md))
 - [Deep API Programmer: Learning to Program with APIs](https://arxiv.org/pdf/1704.04327.pdf)
We present DAPIP, a **Programming-By-Example system** that learns to program with APIs to perform data transformation tasks. We design a **domain specific language** (DSL) that allows for arbitrary concatenations of API outputs and constant strings. The DSL consists of three family of APIs: regular expression-based APIs, lookup APIs, and transformation APIs. We then present a novel **neural synthesis algorithm** to search for programs in the DSL that are consistent with a given set of examples. The search algorithm uses recently introduced neural architectures to encode input-output examples and to model the program search in the DSL. We show that synthesis algorithm outperforms baseline methods for synthesizing programs on both synthetic and real-world benchmarks. <br />
[Neuro-Symbolic Program Synthesis](https://arxiv.org/abs/1611.01855)



 ### Reinforcement Learning
 - [Becca: a general learning program for use in any robot or embodied system](https://github.com/brohrer/becca)
 - [DEEP REINFORCEMENT LEARNING: AN OVERVIEW](https://arxiv.org/pdf/1701.07274.pdf)

### Imbalanced Classification ([Note](doc/imclfi.md))
 - Learning Deep Representation for Imbalanced Classification_cvpr2016 ([Note](doc/Learning_Deep_Representation_for_Imbalanced_Classification_cvpr2016.md))
 - Deep Over-sampling Framework for Classifying Imbalanced Data ([Note](doc/Deep_Over_sampling_Framework_for_Imbalanced_Data_2017.md))
 - [Metric Learning with Adaptive Density Discrimination](https://www.semanticscholar.org/paper/Metric-Learning-with-Adaptive-Density-Discriminati-Rippel-Paluri/bb818c11449768a43722f8087c7529d7875cfc35)
 - [Semi-supervised deep learning by metric embedding](https://www.semanticscholar.org/paper/Semi-supervised-deep-learning-by-metric-embedding-Hoffer-Ailon/0ad0518637d61e8f4b151657797b067ec74418e4)
 - [tf-magnet](https://github.com/pumpikano/tf-magnet-loss)
 - [Training Neural Networks with Very Little Data(data augumentatin)](https://arxiv.org/pdf/1708.04347v2.pdf)

### semi-supervised learning
 - [Semi-supervised deep learning by metric embedding](https://www.semanticscholar.org/paper/Semi-supervised-deep-learning-by-metric-embedding-Hoffer-Ailon/0ad0518637d61e8f4b151657797b067ec74418e4)
 - [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/pdf/1610.02242.pdf)
 - [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/pdf/1606.01583.pdf )
 - [Semi-Supervised Learning with Deep Generative Models **Github1**](https://github.com/dpkingma/nips14-ssl)
 - [Semi-Supervised Learning with Deep Generative Models **Github2**](https://github.com/saemundsson/semisupervised_vae)
 - [Learning Loss Functions for Semi-supervised Learning via Discriminative Adversarial Networks](https://arxiv.org/pdf/1707.02198.pdf)
 - [Auxiliary Deep Generative Models](https://arxiv.org/pdf/1602.05473.pdf)

### data ploting
 - [ImageTSNEViewer](http://ml4a.github.io/guides/ImageTSNEViewer/)
 - [plot_embedding](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html)
 - [plot_embedding2](https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/)


 ### vision
 [Hierarchical Surface Prediction for 3D Object Reconstruction](https://arxiv.org/pdf/1704.00710.pdf)

 ### health
 [Classification of Radiology Reports Using Neural Attention Models](http://xxx.lanl.gov/pdf/1708.06828)

 ### Garment
 - [Physics-driven Pattern Adjustment for Direct 3D Garment Editing](https://pdfs.semanticscholar.org/ad16/fec6f8d35c3cf12e944015466e3c9cba0b8a.pdf)
 - [Detailed Garment Recovery from a Single-View Image](https://arxiv.org/pdf/1608.01250.pdf)
	![ ](../fig/dtailedGRFASVI_Fig1.png)<br />
	Fig. 1. Garment recovery and re-purposing results. From left to right, we show an example of (a) the original image [Saaclothes 2015] c©, (b) the recovered dress and body shape from a single-view image, and (c)-(e) the recovered garment on another body of different poses and shapes/sizes [Hillsweddingdress 2015] c©.<br />
	![ ](../fig/dtailedGRFASVI_Fig2.png)<br />
	Fig. 2. The flowchart of our algorithm. We take a single-view image [ModCloth 2015] c©, a human-body dataset, and a garment-template database as input. We preprocess the input data by performing garment parsing, sizing and features estimation, and human-body reconstruction. Next, we recover an estimated garment described by the set of garment parameters, including fabric material, design pattern parameters, sizing and wrinkle density, as well as the registered garment dressed on the reconstructed body.<br />


 - [Styling Evolution for Tight-Fitting Garments](https://www.researchgate.net/profile/Tsz_Ho_Kwok/publication/280114266_Styling_Evolution_for_Tight-Fitting_Garments/links/55aabfa908aea99467241588.pdf)
 - [Parsing Sewing Patterns into 3D Garments](http://vis.berkeley.edu/papers/clopat/clopat.pdf)

 ### video game
 [The Game Imitation: Deep Supervised Convolutional Networks for Quick Video Game AI](https://arxiv.org/pdf/1702.05663.pdf)
 [ResearchDoom and CocoDoom: Learning Computer Vision with Games](https://arxiv.org/pdf/1610.02431.pdf)
 [ELF: An Extensive, Lightweight and Flexible Research Platform for Real-time Strategy Games](https://arxiv.org/pdf/1707.01067.pdf)
 [Beating the World’s Best at Super Smash Bros. Melee with Deep Reinforcement Learning](https://arxiv.org/pdf/1702.06230.pdf)
 [A Deep Hierarchical Approach to Lifelong Learning in Minecraft](https://arxiv.org/abs/1604.07255)
 [General Video Game AI: Competition, Challenges, and Opportunities](https://www.google.ca/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiL4oCAvfrVAhUk74MKHaOzDfIQFggpMAA&url=https%3A%2F%2Fwww.aaai.org%2Focs%2Findex.php%2FAAAI%2FAAAI16%2Fpaper%2Fdownload%2F11853%2F12281&usg=AFQjCNFBVI3AYrvL1Zpsdy4PREK9w3AbKw)
 [The General Video Game AI Competition - 2017](http://www.gvgai.net/index.php)


 ### Reservoir Computing
 - [website](http://reservoir-computing.org/)
 - [A Comparative Study of Reservoir Computing for Temporal Signal Processing](https://arxiv.org/pdf/1401.2224.pdf)
 - []()