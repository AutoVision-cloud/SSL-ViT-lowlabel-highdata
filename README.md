# Visual Representation Learning with Self-Supervised Attention for _Low-Label_ _High-Data_ Regime
Created by [Prarthana Bhattacharyya](https://ca.linkedin.com/in/prarthana-bhattacharyya-44582a79), [Chenge Li](https://www.linkedin.com/in/chengeli), [Xiaonan Zhao](https://www.linkedin.com/in/zhaoxiaonan), [István Fehérvári](https://ca.linkedin.com/in/istvanfehervari).

This repository contains PyTorch implementation for paper **Visual Representation Learning with Self-Supervised Attention for _Low-Label_ _High-Data_ Regime**.

Self-supervision has shown outstanding results for natural language processing, and more recently, for image recognition. Simultaneously, vision transformers and its variants have emerged as a promising and scalable alternative to convolutions on various computer vision tasks. In this paper, we are the first to question if self-supervised vision transformers (SSL-ViTs) can be adapted to two important computer vision tasks in the low-label, high-data regime: few-shot image classification and zero-shot image retrieval. The motivation is to reduce the number of manual annotations required to train a visual embedder, and to produce generalizable, semantically meaningful and robust embeddings. 

For few-shot image classification we train SSL-ViTs without any supervision, on external data, and use this trained embedder to adapt quickly to novel classes with limited number of labels. For zero-shot image retrieval, we use SSL-ViTs pre-trained on a large dataset without any labels and fine-tune them with several metric learning objectives. Our self-supervised attention representations outperforms the state-of-the-art on several public benchmarks for both tasks, namely miniImageNet and CUB200 for few-shot image classification by up-to 6%-10%, and Stanford Online Products, Cars196 and CUB200 for zero-shot image retrieval by up-to 4%-11%.

</br>

<div align="center">
    <img src="figs/intro.png" width="740">
</div>

