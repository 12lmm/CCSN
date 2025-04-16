# Food image recognition method based on iterative clustering and confidence screening mechanism

Food image recognition is a crucial step for food detection, nutritional analysis, and dietary recommendations. However, this task is highly challenging due to the diversity of food appearances, such as variations in color, texture, and presentation. This paper proposes a novel Clustering and Confidence Screening Network(CCSN), which integrates a clustering method based on Kohonen Networks to fully leverage the inherent characteristics of food images. The model first groups similar features through clustering to capture the complex composition of food ingredients and employs a confidence screening mechanism to select high-confidence samples for subsequent training, ensuring the accuracy of clustering results. Then, an iterative strategy is applied in feature extraction to progressively refine image details, and feature representation is enhanced using a self-attention mechanism. Extensive experiments on multiple datasets, including ETH Food-101, Food11, Food new, and CAFD, demonstrate that the proposed model significantly improves recognition accuracy and exhibits strong generalization capabilities.

# Training
We provide scripts for training models using a multi-node GPU server (e.g., 8 NVIDIA GPUs).

1、Data preparation

Data:

a. [ETHZ Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

b. [food11](https://www.kaggle.com/datasets/vermaavi/food11)

c. [CAFD](https://github.com/IS2AI/Central-Asian-Food-Dataset)

d. [food new](https://www.kaggle.com/datasets/pranavkathar/foodnew)

or other data.

Download data to YOUR_PATH.

2、Training

Usage: me Config_file Model_name Dataset_name Img_size Remove_old_if_exist_0_or_1 Resume_or_not_if_exist Exp_name Tag Gpus Nb_gpus Workers Port

3、Validate

Usage: me model_name checkpoint_file dataset_name img_size gpus num_gpus


