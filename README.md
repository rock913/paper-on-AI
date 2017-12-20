# Paper-on-AI

## Topics

### Probabilistic programming language ([Note](doc/ppl.md))
 - Human-level concept learning through probabilistic program induction ([Note](doc/ppl_human_level_concep_learning.md))
 - Picture: A Probabilistic Programming Language for Scene Perception ([Note](doc/Picture_PPL_for_Scene_Perception.md))
 - [Deep API Programmer: Learning to Program with APIs](https://arxiv.org/pdf/1704.04327.pdf)
We present DAPIP, a **Programming-By-Example system** that learns to program with APIs to perform data transformation tasks. We design a **domain specific language** (DSL) that allows for arbitrary concatenations of API outputs and constant strings. The DSL consists of three family of APIs: regular expression-based APIs, lookup APIs, and transformation APIs. We then present a novel **neural synthesis algorithm** to search for programs in the DSL that are consistent with a given set of examples. The search algorithm uses recently introduced neural architectures to encode input-output examples and to model the program search in the DSL. We show that synthesis algorithm outperforms baseline methods for synthesizing programs on both synthetic and real-world benchmarks. <br />
[Neuro-Symbolic Program Synthesis](https://arxiv.org/abs/1611.01855)

### Bayesian Deep Learning
- [ZhuSuan: A Library for Bayesian Deep Learning](https://arxiv.org/pdf/1709.05870.pdf)
- [Robust RegBayes: Selectively Incorporating First-Order Logic Domain Knowledge into Bayesian Models](http://proceedings.mlr.press/v32/mei14.pdf)
- [Uncertainty in Deep Learning](mlg.eng.cam.ac.uk/yarin/blog_2248.html)

 ### Reinforcement Learning
 - [Becca: a general learning program for use in any robot or embodied system](https://github.com/brohrer/becca)
 - [DEEP REINFORCEMENT LEARNING: AN OVERVIEW](https://arxiv.org/pdf/1701.07274.pdf)
 - [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/pdf/1711.00832.pdf)

### Imbalanced Classification ([Note](doc/imclfi.md))
 - Learning Deep Representation for Imbalanced Classification_cvpr2016 ([Note](doc/Learning_Deep_Representation_for_Imbalanced_Classification_cvpr2016.md))
 - Deep Over-sampling Framework for Classifying Imbalanced Data ([Note](doc/Deep_Over_sampling_Framework_for_Imbalanced_Data_2017.md))
 - [Metric Learning with Adaptive Density Discrimination](https://www.semanticscholar.org/paper/Metric-Learning-with-Adaptive-Density-Discriminati-Rippel-Paluri/bb818c11449768a43722f8087c7529d7875cfc35)
 - [Semi-supervised deep learning by metric embedding](https://www.semanticscholar.org/paper/Semi-supervised-deep-learning-by-metric-embedding-Hoffer-Ailon/0ad0518637d61e8f4b151657797b067ec74418e4)
 - [tf-magnet](https://github.com/pumpikano/tf-magnet-loss)
 - [Training Neural Networks with Very Little Data(data augumentatin)](https://arxiv.org/pdf/1708.04347v2.pdf)

### semi-supervised learning
 - [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/pdf/1703.01780.pdf) [[code](https://github.com/CuriousAI/mean-teacher)]
 - [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/pdf/1610.02242.pdf)
 - [Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning](https://arxiv.org/abs/1704.03976)
 - [Self-ensembling for domain adaptation](https://arxiv.org/pdf/1706.05208.pdf)
 - [Semi-supervised deep learning by metric embedding](https://www.semanticscholar.org/paper/Semi-supervised-deep-learning-by-metric-embedding-Hoffer-Ailon/0ad0518637d61e8f4b151657797b067ec74418e4)
 - [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/pdf/1606.01583.pdf )
 - [Semi-Supervised Learning with Deep Generative Models **Github1**](https://github.com/dpkingma/nips14-ssl)
 - [Semi-Supervised Learning with Deep Generative Models **Github2**](https://github.com/saemundsson/semisupervised_vae)
 - [Learning Loss Functions for Semi-supervised Learning via Discriminative Adversarial Networks](https://arxiv.org/pdf/1707.02198.pdf)
 - [Auxiliary Deep Generative Models](https://arxiv.org/pdf/1602.05473.pdf)

### label noise
- [CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise](https://arxiv.org/pdf/1711.07131.pdf)

### data ploting
 - [ImageTSNEViewer](http://ml4a.github.io/guides/ImageTSNEViewer/)
 - [plot_embedding](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html)
 - [plot_embedding2](https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/)

### self play
- [Mastering the Game of Go without Human Knowledge](https://slack-files.com/files-pri-safe/T4P2UU3A8-F7MEP9K1D/2017_silver_alphagozero_unformatted_nature.pdf?c=1508355584-65a49fbd32f155b6e2422137c04eb2b566ac7cf1_)

 ### vision
 [Hierarchical Surface Prediction for 3D Object Reconstruction](https://arxiv.org/pdf/1704.00710.pdf)
 [3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction](https://arxiv.org/pdf/1604.00449.pdf)

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

### generate data from video game
- [Playing for Data: Ground Truth from Computer Games](download.visinf.tu-darmstadt.de/data/from_games/data/eccv-2016-richter-playing_for_data.pdf)
- [UnrealCV: Connecting Computer Vision to Unreal Engine](https://arxiv.org/pdf/1609.01326.pdf)
- [Augmented Reality Meets Computer Vision : Efficient Data Generation for Urban Driving Scenes](https://arxiv.org/pdf/1708.01566.pdf)
- [Model-driven Simulations for Computer Vision](https://pdfs.semanticscholar.org/44bb/6ccb3526bb38364550263bc608116910da32.pdf)
- [UE4Sim: A Photo-Realistic Simulator for Computer Vision Applications](https://arxiv.org/pdf/1708.05869.pdf)
- [Teaching UAVs to Race Using UE4Sim](https://arxiv.org/pdf/1708.05884.pdf)
- [Video Propagation Networks](https://arxiv.org/pdf/1612.05478.pdf)


### GANS([Note](doc/gans.md))
- [RENDERGAN: GENERATING REALISTIC LABELED DATA](https://arxiv.org/pdf/1611.01331.pdf)

### conditional GAN
- [Disentangled Variational Auto-Encoder for Semi-supervised Learning](https://arxiv.org/pdf/1709.05047.pdf)
- [Fader Networks: Manipulating Images by Sliding Attributes](https://arxiv.org/pdf/1706.00409.pdf)
	Distangle face attributes from laten variables by GAN
- [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/pdf/1706.02633.pdf)
- [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/pdf/1703.05192.pdf)
- [FACE AGING WITH CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1702.01983.pdf)
- [GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data](https://arxiv.org/pdf/1705.04932.pdf)
- [Invertible Conditional GANs for image editing](https://arxiv.org/pdf/1611.06355.pdf)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)

 ### RNN
 - [Training RNNs as Fast as CNNs](https://arxiv.org/pdf/1709.02755.pdf) [[code](https://github.com/taolei87/sru)]
 - [CreativeAI](www.creativeai.net/page/about)

 ### Reservoir Computing
 - [website](http://reservoir-computing.org/)
 - [A Comparative Study of Reservoir Computing for Temporal Signal Processing](https://arxiv.org/pdf/1401.2224.pdf)
 - [Samim](www.samim.io)
 - []()

 ### Design
 -[CreativeAI](https://medium.com/@creativeai/creativeai-9d4b2346faf3)

###sports
- [Data-Driven	Ghosting	using	Deep	Imitation	Learning](https://s3-us-west-1.amazonaws.com/disneyresearch/wp-content/uploads/20170228130457/Data-Driven-Ghosting-using-Deep-Imitation-Learning-Paper1.pdf)

### text summorization
- [GitXiv](www.gitxiv.com/page/about)

### web scraper
- [web scraper blog1](http://haiyuanzhang.pub/yi-bu-bu-li-yong-kai-yuan-xiang-mu-shi-xian-wang-luo-pa-chong-yi-zhua-qu-zheng-quan-ri-bao-xin-wen-wei-li/)
- [scrapper using GO](https://schier.co/blog/2015/04/26/a-simple-web-scraper-in-go.html)
- [A simple, higher level interface for Go web scraping](https://github.com/yhat/scrape)
- [Github scapper-python](https://github.com/sbaack/github-scraper)
- [Github scapper-javascript](https://github.com/nelsonic/github-scraper)

- [ How To Use node.js, request and cheerio to Set Up Simple Web-Scraping](https://www.digitalocean.com/community/tutorials/how-to-use-node-js-request-and-cheerio-to-set-up-simple-web-scraping)


### abductive resoning
- [Watson: Beyond Jeopardy!](https://ac.els-cdn.com/S0004370212000872/1-s2.0-S0004370212000872-main.pdf?_tid=b6496b4c-a852-11e7-a5e0-00000aacb35f&acdnat=1507046065_d0ba77d6b7037242cd8c248b39b735be)
it elaborates upon a vision for an evidence-based clinical decision support system, based on the DeepQA technology, that affords exploration of a broad range of hypotheses and their associated evidence, as well as uncovers missing information that can be used in mixed-initiative dialog.
- [Abduction in Machine Learning](https://www.researchgate.net/profile/Vincenzo_Cutello/publication/228933219_Abduction_in_Machine_Learning/links/0912f50ca4c5732926000000/Abduction-in-Machine-Learning.pdf)
- [Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge](https://arxiv.org/pdf/1606.04422.pdf)
- [Neural-Symbolic Computing, Deep logic networks and applications](www.staff.city.ac.uk/~aag/talks/dags2014.pdf)
- [Reasoning with Deep Learning: an Open Challenge](ceur-ws.org/Vol-1802/paper5.pdf)
- [3 Types Of Reasoning And AlphaGo](https://www.twinword.com/blog/3-types-of-reasoning-and-alphago/)
**inductive reasoning** He died and she died. Everyone died, so I will die
**deductive reasoning** If all humans die and I am a human, then I will die. 
**abductive reasoning** He died and the cat died, so he is a cat

### Financial
- [Deep Learning in Finance](https://arxiv.org/pdf/1602.06561.pdf)

### recursive network
- [Recursive Neural Networks](www.iro.umontreal.ca/~bengioy/talks/gss2012-YB6-NLP-recursive.pdf)

### EEG
- [Generative Adversarial Networks Conditioned by Brain Signals](https://pdfs.semanticscholar.org/dd3a/f1c0a31ac683d435f875362e2c432a8c7ada.pdf)
- [Deep adversarial neural decoding](https://arxiv.org/pdf/1705.07109.pdf)

### neuroscience
- [NeuCube: A spiking neural network architecture for mapping, learning and understanding of spatio-temporal brain data](https://pdfs.semanticscholar.org/64f2/e480e010368b1618bd31996397fd2287a404.pdf)

### segmentation
- [Speed/accuracy trade-offs for modern convolutional object detectors](openaccess.thecvf.com/content_cvpr_2017/papers/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf)
  Good paper summory on instance segmentation methods:the Faster R-CNN, R-FCN and SSD systems
- [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.

### pose estimation
- [Towards Accurate Multi-person Pose Estimation in the Wild](https://arxiv.org/pdf/1701.01779.pdf)
- [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields ](https://arxiv.org/pdf/1611.08050.pdf)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)[[code]](https://github.com/endernewton/tf-faster-rcnn) [[code2]](https://github.com/smallcorgi/Faster-RCNN_TF)
- [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://arxiv.org/pdf/1607.08128.pdf)


### Capsules
- [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
- [MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb)
- [Recurrent Neural Networks for Spatiotemporal Dynamics of Intrinsic Networks from fMRI Data](https://arxiv.org/pdf/1611.00864.pdf)

### active learning
- [libact: Pool-based Active Learning in Python](https://arxiv.org/pdf/1710.00379.pdf) [(Github)](https://github.com/ntucllab/libact)
- [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf)
- [Learning how to Active Learn: A Deep Reinforcement Learning Approach](people.eng.unimelb.edu.au/tcohn/papers/emnlp17pal.pdf)

### autoencoder
- [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf)

### graph convolutional recurrent networks
- [Structured sequence modeling with graph convolutional recurrent networks](https://arxiv.org/pdf/1612.07659.pdf)

### one shot learning
- [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
   Some kind of similar with KNN, need to read a paper on "attention lstm"

### Asynchronous Actor-Critic Agents
- [Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
- [A major achievement in reinforcement learning research](www.maluuba.com/hra)


### database
- [TensorLayer: A Versatile Library for Efficient Deep Learning Development](https://arxiv.org/pdf/1707.08551.pdf)
- [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://arxiv.org/pdf/1703.04046.pdf)


### automat learning process
- [Learning to learn by gradient descent](https://arxiv.org/pdf/1606.04474.pdf)

### Visualization
- [Plug-and-Play Interactive Deep Network Visualization](https://www.researchgate.net/profile/Gjorgji_Strezoski/publication/319932280_Plug-and-Play_Interactive_Deep_Network_Visualization/links/59c24aa9a6fdcc69b92fbf57/Plug-and-Play-Interactive-Deep-Network-Visualization.pdf)
- [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/abs/1711.02257)


### AI as servise
- [labellio](https://www.labell.io)
- [clarifai](https://www.clarifai.com/?mkt_tok=eyJpIjoiTkdZd056Y3daVEkyTVdWaiIsInQiOiJNZzBtaEFLWGFCTEs3amxkdTJQd1JYR1wvVFJhYUZIa1BQaWppOTJvOGpZUmNpRVZyVWg0NVZDQWJnVHhFZ1d0RkticWZoa0VVeHNsRUJwMTk2TDB4TU5QZmtLYlwvVTBcL3VndmpRd08yZGdYK1hYUGZOU1RManJkOWgwQVFoM0RITCJ9)
- [fastai](http://www.fast.ai/)


### Transfer and multi-task learning ([Note](doc/transfer_and_multi_task.md))

### transfer learning 
- [A survey of transfer learning](https://journalofbigdata.springeropen.com/track/pdf/10.1186/s40537-016-0043-6?site=journalofbigdata.springeropen.com)
- [Transfer learning and domain adaptation](https://drive.google.com/open?id=1fqdpd0V3FXMoAIGUc3aE9L853oqTY0c8)[second](https://drive.google.com/a/stradigi.ca/file/d/1xcbGAEivt7m_JDSExrbqiJcJtW7DgfdQ/view?usp=sharing)
frozone; fine-turn; domain; distillation; semi-supervised
- [Transfer Learning — The Next Frontier for ML](https://drive.google.com/file/d/1tpE7vcx8SuP5q5za-3_hZa7BDsOTW3-T/view)
- [Transfer Learning - Machine Learning's Next Frontier](ruder.io/transfer-learning/index.html)

### multi-task learning
- [An Overview of Multi-Task Learning in Deep Neural Networks](http://ruder.io/multi-task/)
- [Multi-task Self-Supervised Visual Learning](openaccess.thecvf.com/content_ICCV_2017/papers/Doersch_Multi-Task_Self-Supervised_Visual_ICCV_2017_paper.pdf)
- [HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition](https://arxiv.org/pdf/1603.01249.pdf)
- [Multi-task, Multi-lingual Learning](www.phontron.com/class/nn4nlp2017/assets/slides/nn4nlp-25-multitask.pdf)