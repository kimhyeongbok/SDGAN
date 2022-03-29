# SDGAN: Semantic-aware De-Identity Generative Adversarial Networks for Identity Anonymization
 
Official PyTorch implementation of SDGAN

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

You can download pre-trained model [here (google drive)](https://drive.google.com/file/d/1j5iT-SvvbC-JRy7qvY-eEP4sLzvoh8Ut/view?usp=sharing).


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
