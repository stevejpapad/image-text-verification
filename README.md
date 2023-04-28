# figments-and-misalignments

Official repository for the "Figments and Misalignments: A Framework for Fine-grained Crossmodal Misinformation Detection" paper. You can read the pre-print here: https://doi.org/10.48550/arXiv.2304.14133

## Abstract
>*Multimedia content has become ubiquitous on social media platforms, leading to the rise of multimodal misinformation and the urgent need for effective strategies to detect and prevent its spread. This study focuses on CrossModal Misinformation (CMM) where image-caption pairs work together to spread falsehoods. We contrast CMM with Asymmetric Multimodal Misinformation (AMM), where one dominant modality propagates falsehoods while other modalities have little or no influence. We show that AMM adds noise to the training and evaluation process while exacerbating the unimodal bias, where text-only or image-only detectors can seemingly outperform their multimodal counterparts on an inherently multimodal task. To address this issue, we collect and curate FIGMENTS, a robust evaluation benchmark for CMM, which consists of real world cases of misinformation, excludes AMM and utilizes modality balancing to successfully alleviate unimodal bias. FIGMENTS also provides a first step towards fine-grained CMM detection by including three classes: truthful, out-of-context, and miscaptioned image-caption pairs. Furthermore, we introduce a method for generating realistic synthetic training data that maintains crossmodal relations between legitimate images and false human-written captions that we term Crossmodal HArd Synthetic MisAlignment (CHASMA). We conduct extensive comparative study using a Transformer-based architecture. Our results show that incorporating CHASMA in conjunction with other generated datasets consistently improved the overall performance on FIGMENTS in both binary (+6.26%) and multiclass settings (+15.8%).*

This repository also reproduces the methods presented in [Synthetic Misinformers: Generating and Combating Multimodal Misinformation](https://arxiv.org/abs/2303.01217).

If you find our work useful, please cite:
```
@article{papadopoulos2023figments,
  title={Figments and Misalignments: A Framework for Fine-grained Crossmodal Misinformation Detection},
  author={Papadopoulos, Stefanos-Iordanis and Koutlis, Christos and Papadopoulos, Symeon and Petrantonakis, Panagiotis C},
  journal={arXiv preprint arXiv:2304.14133},
  year={2023}
}

@article{papadopoulos2023synthetic,
  title={Synthetic Misinformers: Generating and Combating Multimodal Misinformation},
  author={Papadopoulos, Stefanos-Iordanis and Koutlis, Christos and Papadopoulos, Symeon and Petrantonakis, Panagiotis C},
  journal={arXiv preprint arXiv:2303.01217},
  year={2023}
}
```

## Preparation

- Clone this repo: 
```
git clone https://github.com/stevejpapad/figments-and-misalignments
cd figments-and-misalignments
```

- Create a python (>= 3.8) environment (Anaconda is recommended) 

- Install all dependencies with: `conda install --file requirements.txt` and follow the [instructions](https://github.com/openai/CLIP) to install CLIP.

## FIGMENTS Benchmark

FIGMENTS is a benchmark dataset designed for evaluating fine-grained crossmodal misinformation detection models. This dataset consists of real-world instances of misinformation collected from Snopes and Reuters, and it addresses unimodal bias by excluding asymmetric misinformation and employing modality balancing. Modality balancing denotes that images and captions will appear twice, once in their truthful and once in their misleading pairs to ensure that the model considers both modalities when distinguishing between truth and misinformation.

![Screenshot](FIGMENTS/figments.png)

The images are sourced from within the articles of Snopes and Reuters, as well as Google Images. We do not provide the images, only their URLs. 
FIGMENTS supports multiclass classification of three categories: Truthful, Out-of-context, and Miscaptioned image-caption pairs but it can also be used for binary classification. 
We collected 260 articles from Snopes and 78 from Reuters that met our criteria which translates to 338 Truthful, 338 Miscaptioned and 324 Out-of-Context pairs. 
Please note that this dataset is intended solely for research purposes.

- If you are only interested in the FIGMENTS benchmark, we provide the processed dataset and the visual and textual features from CLIP ViT-L/14 in `/FIGMENTS`. 
- If you also want to download the images from the provided URLs, you can run the following code:
```python
from prepare_datasets import prepare_figments
prepare_FIGMENTS(download_images=True)
```
- To extract visual and textual features of FIGMENTS with the use of CLIP ViT-L/14 run the following code: 
```python
from extract_features import extract_CLIP_features
extract_CLIP_features(data_path='FIGMENTS/', output_path='FIGMENTS/FIGMENTS_')
```

Samples from the FIGMENTS benchmark:
```python
{'caption': 'Image shows a damaged railway bridge collapsed into a body of water in June 2020 in Murmansk, Russia.',
 'image_path': 'images/true_239.jpg',
 'label': 'true'}
{'caption': 'Image shows a damaged railway bridge collapsed into a body of water in 2022 during the Russia-Ukraine war.',
 'image_path': 'images/true_239.jpg',
 'label': 'miscaptioned'}
{'caption': 'Image shows a damaged railway bridge collapsed into a body of water in June 2020 in Murmansk, Russia.',
 'image_path': 'images/false_239.jpg',
 'label': 'out-of-context'}
```
If you encounter any problems when downloading and preparing FIGMENTS (e.g dead image URLs) please contact with stefpapad@iti.gr

## Prerequisite
If you want to reproduce the experiments on the paper it is necassary to first download the following datasets and save them in their respective folder: 
- COSMOS test set -> https://github.com/shivangi-aneja/COSMOS -> `/COSMOS`
- Fakeddit dataset -> https://github.com/entitize/Fakeddit -> `/Fakeddit`
- MEIR -> https://github.com/Ekraam/MEIR -> `/MEIR`
- VisualNews -> https://github.com/FuxiaoLiu/VisualNews-Repository -> `/VisualNews`
- NewsCLIPings -> https://github.com/g-luo/news_clippings -> `/news_clippings`
- Image-verification-corpus (Twitter dataset) -> https://github.com/MKLab-ITI/image-verification-corpus -> `/Twitter`

## Reproducibility
All experiments from the paper can be re-created by running 
```python main.py``` 
to prepare the datasets, extract the CLIP features and reproduce all experiments. 

## Usage
- To extract visual and/or textual features with the use of CLIP ViT-L14 run:
```python
from extract_features import extract_CLIP_features
extract_CLIP_features(data_path=INPUT_PATH, output_path=OUTPUT_PATH) 
```

- To prepare the *CHASMA* dataset run the following: 
```python
from prepare_datasets import prepare_Misalign
prepare_Misalign(CLIP_VERSION="ViT-L/14", choose_gpu=0)
```

- To train the DT-Transformer (e.g) for 30 epochs, on binary classification, using the *CHASMA* dataset while utilizing both multimodal and unimodal (text-only, image-only) inputs with a learning_rate of 5e-5, 4 transformer layers, 8 attention heads of dimensionality 1024, run the following code: 
```python
run_experiment(
    dataset_methods_list = [
        'Misalign', 
    ],
    modality_options = [
        ["images", "texts"],
        ["texts"], 
        ["images"]
    ],
    epochs=30,
    seed_options = [0],
    lr_options = [5e-5],
    batch_size_options = [512],
    tf_layers_options = [4],
    tf_head_options = [8],
    tf_dim_options = [1024],
    use_multiclass = False, 
    balancing_method = None,
    choose_gpu = 0, 
)
```

- Similarly, for finegrained detection using with the *CHASMA + R-NESt + NC-t2t* run the following: 
```python
run_experiment(
    dataset_methods_list = [
        'EntitySwaps_random_topicXMisalignXnews_clippings_txt2txt',
    ],
    epochs=30,
    use_multiclass = True,
    balancing_method = 'downsample',
)
```

## Acknowledgements
This work is partially funded by the project "vera.ai: VERification Assisted by Artificial Intelligence" under grant agreement no. 101070093.

## Licence
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/stevejpapad/figments-and-misalignments/blob/master/LICENSE) file for more details.

## Contact
Stefanos-Iordanis Papadopoulos (stefpapad@iti.gr)
