# SDGAN: Semantic-aware De-identification Generative Adversarial Networks for Identity Anonymization

Our research has experimented on various angles, occlusion, facial expressions, gender, ages, skin colors etc.

To the best of our knowledge, our study is an experimental study case on the most diverse conditions.

<p align="center"> <b>Front Face Images
<p align="center"><img src="https://user-images.githubusercontent.com/41537576/165922091-0b3d52c9-b960-435f-b4cb-27df9bd49b6b.png" width="700" height="370">
<p align="center"> <b>Side Face Images
<p align="center"><img src="https://user-images.githubusercontent.com/41537576/165922457-f74b7a77-e2c1-4af7-9338-8ef6e17f53be.png" width="700" height="370">
<p align="center"> <b>Occlusion Face Images
<p align="center"><img src="https://user-images.githubusercontent.com/41537576/165922755-de4f8b34-a45a-401a-aaa6-3382886e39d2.png" width="700" height="370">

**Our research is currently under review.
We will officially open this Github after it is published.**
  
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

## Training

In order to train a SDGAN model, run the following command:

````
python run_training.py
````

We provided an example of our dataset that contains 5 identity folders from celebA dataset in the dataset folder. To train with full celebA dataset (or your own dataset), please setup the data in the same format. For the results generated in our paper, we trained the network using 1200 identities (each of them having at least 30 images) from celebA dataset. The identities can be found in: 

````
dataset/celeba/legit_indices.npy
````

You can download pre-trained model [here (google drive)](https://drive.google.com/drive/folders/1RcIntjkg6PgsijBilNLUyo17zfs7ERei?usp=sharing).


We provide example of inference code in test.py file:

````
python test.py --model [path to the model and its name] --data [path to the data (optional)] -out [path to the output directory (optional)]
````


To process landmarks you can use code in process_data.py:
````
python process_data.py --input [path to a directory with raw data] --output [path to the output directory] -dlib [path to the dlib shape detector model(optional)]
````



## Citation

If you find this code useful, please consider citing the following paper:

````
````
