# SDGAN: Semantic-aware De-identification Generative Adversarial Networks for Identity Anonymization

Abstract
---
Privacy protection in the computer vision field has attracted increasing attention. Generative adversarial network-based methods have been explored for identity anonymization, but they do not take into consideration semantic information of images, which may result in unrealistic or flawed facial results. In this paper, we propose a Semantic-aware De-identification Generative Adversarial Network (SDGAN) model for identity anonymization. To retain the facial expression effectively, we extract the facial semantic image using the edge-aware graph representation network to constraint the position, shape and relationship of generated facial key features. Then the semantic image is injected into the generator together with the randomly selected identity information for de-Identification. To ensure the generation quality and realistic-looking results, we adopt the SPADE architecture to improve the generation ability of conditional GAN. Meanwhile, we design a hybrid identity discriminator composed of an image quality analysis module, a VGG-based perceptual loss function, and a contrastive identity loss to enhance both the generation quality and ID anonymization. A comparison with the state-of-the-art baselines demonstrates that our model achieves significantly improved de-identification (De-ID) performance and provides more reliable and realistic-looking generated faces. 

---

Our research has experimented on various angles, occlusion, facial expressions, gender, ages, skin colors etc.
To the best of our knowledge, our research has been tested under the most diverse conditions.

<p align="center"> <b>Front Face Images
<p align="center"><img src="https://user-images.githubusercontent.com/41537576/165922091-0b3d52c9-b960-435f-b4cb-27df9bd49b6b.png" width="700" height="370">
<p align="center"> <b>Side Face Images
<p align="center"><img src="https://user-images.githubusercontent.com/41537576/165922457-f74b7a77-e2c1-4af7-9338-8ef6e17f53be.png" width="700" height="370">
<p align="center"> <b>Occlusion Face Images
<p align="center"><img src="https://user-images.githubusercontent.com/41537576/165922755-de4f8b34-a45a-401a-aaa6-3382886e39d2.png" width="700" height="370">


** Now, we are preparing to share the code. **
  
  
## Installation


Please download the code:

To use our code, first download the repository:
````
git clone https://github.com/kimhyeongbok/SDGAN.git
````

To install the dependencies:

````
pip install -r requirements.txt
````


## Citation

If you find this code useful, please consider citing the following paper:

````
@article{kim2023semantic,
  title={Semantic-aware deidentification generative adversarial networks for identity anonymization},
  author={Kim, Hyeongbok and Pang, Zhiqi and Zhao, Lingling and Su, Xiaohong and Lee, Jin Suk},
  journal={Multimedia Tools and Applications},
  volume={82},
  number={10},
  pages={15535--15551},
  year={2023},
  publisher={Springer}
}
````
