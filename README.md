# Visual Representation Learning with Self-Supervised Attention for _Low-Label_ _High-Data_ Regime
Created by [Prarthana Bhattacharyya](https://ca.linkedin.com/in/prarthana-bhattacharyya-44582a79).

This repository contains PyTorch implementation for paper **Visual Representation Learning with Self-Supervised Attention for _Low-Label_ _High-Data_ Regime**.

Self-supervision has shown outstanding results for natural language processing, and more recently, for image recognition. Simultaneously, vision transformers and its variants have emerged as a promising and scalable alternative to convolutions on various computer vision tasks. In this paper, we are the first to question if self-supervised vision transformers (SSL-ViTs) can be adapted to two important computer vision tasks in the low-label, high-data regime: few-shot image classification and zero-shot image retrieval. The motivation is to reduce the number of manual annotations required to train a visual embedder, and to produce generalizable, semantically meaningful and robust embeddings. 

</br>

<div align="center">
    <img src="figs/intro.png" width="700">
</div>

## Results
- SSL-ViT + few-shot image classification:
<div align="center">
    <img src="figs/fewshottable.png" width="700">
</div>

- Qualitative analysis for base-classes chosen by supervised CNN and SSL-ViT for few-shot distribution calibration:
<div align="center">
    <img src="figs/qualitative.png" width="600">
</div>

- SSL-ViT + zero-shot image retrieval:
<div align="center">
    <img src="figs/retrievaltable.png" width="700">
</div>

## Pretraining Self-Supervised ViT
- Run DINO with ViT-small network on a single node with 4 GPUs for 100 epochs with the following command. 
```python
cd dino/
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```
- For mini-ImageNet pretraining, we use the classes listed in: ```ssl-vit-fewshot/data/ImageNetSSLTrainingSplit_mini.txt```
For tiered-ImageNet pretraining, we use the classes listed in: ```ssl-vit-fewshot/data/ImageNetSSLTrainingSplit_tiered.txt```
- For CUB-200, Cars-196 and SOP, we use the pretrained model from:
```python
import torch
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
```

## Usage
### Dataset Preparation
Please follow the instruction in [FRN](https://github.com/Tsingularity/FRN) for few-shot image classification and [RevisitDML](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch) for zero-shot image retrieval to download the datasets and put all the datasets in `data` folder.

### Training and Evaluation for few-shot image classification
- The first step is to extract features for base and novel classes using the pretrained SSL-ViT.
- ```get_dino_miniimagenet_feats.ipynb``` extracts SSL-ViT features for the base and novel classes.
- Change the hyper-parameter ```data_path``` to use CUB or tiered-ImageNet.
- The SSL-ViT checkpoints for the various datasets are provided below (Note: this has only been trained without labels). We also provide the extracted features which need to be stored in ```ssl-vit-fewshot/dino_features_data/```. 

<table> 
    <tr>
        <th>arch</th>
        <th>dataset</th>
        <th >download</th>
        <th >extracted-train</th>
        <th >extracted-test</th>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>mini-ImageNet</td>
        <td><a href="https://drive.google.com/file/d/1kXdqIR15fYKANx97blXVm8Fbwiz-FTvF/view?usp=sharing">mini_imagenet_checkpoint.pth</a></td>
        <td><a href="https://drive.google.com/file/d/1CcIF4NnDoAB12FCq4Q2Q0jrTxCk_wVfM/view?usp=sharing">train.p</a></td>
        <td><a href="https://drive.google.com/file/d/1KpQkdOwFr_jM-t4t-gA9cMPWGFuqyV2Y/view?usp=sharing">test.p</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>tiered-ImageNet</td>
        <td><a href="https://drive.google.com/file/d/1ObdJQozacGBcAmh8iWBrXv6dG6-OaC1i/view?usp=sharing">tiered_imagenet_checkpoint.pth</a></td>
        <td><a href="https://drive.google.com/file/d/1a7EdInD6JBWA3ppOWfhtVEBqNZWnq_13/view?usp=sharing">train.p</a></td>
        <td><a href="https://drive.google.com/file/d/13WAjvQAE3Lwtj83wvHLfVKYWvaw_90Q6/view?usp=sharing">test.p</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>CUB</td>
        <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth">cub_checkpoint.pth</a></td>
        <td><a href="https://drive.google.com/file/d/1KfXbBCxDQ3CchFQVUSmTz2bjSfQKCmkW/view?usp=sharing">train.p</a></td>
        <td><a href="https://drive.google.com/file/d/1pU21XZ2r8cDkFhwWwEwNcRB1FIIREDC_/view?usp=sharing">test.p</a></td>
    </tr>
</table>    

- For _n_-way-_k_-shot evaluation, we provide ```miniimagenet_evaluate_dinoDC.ipynb```.  
