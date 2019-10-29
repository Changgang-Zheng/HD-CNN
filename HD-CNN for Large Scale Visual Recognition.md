## HD-CNN: Hierarchical Deep Convolutional Neural Networks for Large Scale Visual Recognition
>This paper introduce hierarchical deep CNNs (HD-CNNs) by embedding deep CNNs into a two-level category hierarchy. An HD-CNN separates easy classes using a coarse category classifier while distinguishing difficult classes using fine category classifiers. During HD-CNN training, component-wise pretraining is followed by global fine-tuning with a multinomial logistic loss regularized by a coarse category consistency term. In addition, conditional executions of fine category classifiers and layer parameter compression make HD-CNNs scalable for large scale visual recognition. 

<center><img src="/Users/changgang/Documents/Study Notes/Data/Photos/Screen Shot 2019-01-10 at 2.54.56 PM.png"  alt="Transfer Learning" width="900"/></center>

* It include components:
	* shared layers
	* a single component B to handle coarse categories
	* multiple components for fine classification (one for each group) 
	* a single probabilistic averaging layer
* Shared layers receive raw image pixels as input and extract low-level features. The configuration of shared layers is set to be the same as the preceding layers in the building block CNN.
* The coarse category probabilities serve two purposes. 
	* First, they are used as weights for combining the predictions made by fine category components {Fk}Kk=1.
	*  Second, when thresholded, they enable conditional execution of fine category components whose corresponding coarse probabilities are sufficiently large.
* Independent layers of coarse category component B, which reuses the configuration of rear layers from the building block CNN and produces an intermediate fine prediction for an image. To produce a coarse category prediction, we append a fine-to-coarse aggregation layer (not shown in Fig), which reduces fine predictions into coarse using a mapping. The coarse category probabilities serve two purposes. First, they are used as weights for combining the predictions made by fine category components. Second, when thresholded, they enable conditional execution of fine category components whose corresponding coarse probabilities are sufficiently large.

First, we introduce HD-CNN, a novel hierarchical architecture for image classification. Second, we develop a scheme for learning the two-level organization of coarse and fine categories, and demonstrate that various components of an HD-CNN can be independently pretrained. The complete HD-CNN is further finetuned using a multinomial logistic loss regularized by a coarse category consistency term. Third, we make the HD-CNN scalable by compressing the layer parameters and conditionally executing the fine category classifiers. We demonstrate state-of-the-art performance on both CIFAR100 and ImageNet.
﻿﻿﻿﻿
## Steps
*  we initialize the coarse category component B with the weights of F_p.
*  Therefore, the pretraining of each Fk only uses images {xi |i ∈ Skc } from the coarse category.
	* shared preceding layers are already initialized and kept fixed in this stage. 
	* For each Fk, we initialize all the rear layers except the last convolutional layer by copying the learned parameters from the pretrained model Fp.
* After both coarse and fine category components are properly pretrained, we fine-tune the complete HD-CNN. 
	* Coarse category consistency. 
	* During fine-tuning, the semantics of coarse categories predicted by the coarse category component should be kept consistent with those associated with fine category components. Thus we add a coarse category consistency term to regularize the conven- tional multinomial logistic loss.
