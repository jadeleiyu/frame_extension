# Syntactic frame extension via multimodal chaining

This GitHub repository contains code implementation and data for the paper:

Yu, L. and Xu, Y. (2021) Predicting emergent linguistic compositions through time: Syntactic frame extension via multimodal chaining. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ (to appear).

## Dependencies
The following python libaraies/environments are required to run the code:
```
python (>= 3.6.1)
jupyter
numpy
pandas
torch
matplotlib
nltk
```
The training process in ```main.py``` can be significantly accelerated if GPU(s) with ```CUDA``` is available.
## Data
### Google Syntactic N-grams (GSN)
* Raw syntactic N-grams data can be downloaded at: http://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html. All frame usages are extracted from the 99 files under the "Verbargs" subcategory.
* Extracted syntactic N-grams for each decade can be found in ```data/gsn/```.

### ConceptNet 
To train ConceptNet embeddings with "anachronistic" concepts pruned out in historical decades:
1. Download ConceptNet assertion data from https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz.
2. Run the code in ```src/conceptnet.py``` to compute ConceptNet embeddings for each decade.

### ImageNet
To request for a full access to the ImageNet dataset, follow the instructions at https://image-net.org/download.php.
Pre-processed visual representations can be produced by running ```/src/imagenet.py```.

## Experiments
* To train and evaluate SFEM models, run ```jupyter main.ipynb```. Change configurations in the notebook to learn SFEMs with different modalities. Hyperparameters that yield results in the paper can also be found in the default configuration. 
* To reproduce analysis results (AUC scores and curves), run ```jupyter analysis.ipynb```.



