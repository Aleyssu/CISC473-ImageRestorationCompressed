# CISC 473 - ImageRestorationCompressed
An experiment in compressing image restoration NN models using pruning and quantization.

# To Run
To prune and quantize the baseline model, first acquire the weights of the baseline model from https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR and place the `.pth` in `models/`.

In `model.ipynb`, you can adjust the pruning percentage and whether or not quantization is applied under the "Establish model compression parameters section". Set `apply_quantization` to true to apply quantization and `prune_amount` to a value between 0 and 1 to set the percentage of weights in the convolutional layers which will be pruned. 

Run all to load and process the model. Its weights will be saved in `models/`.

# Evaluating Models
The validation dataset is acquired from [Div2K](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images?resource=download)

Just extract the downloaded zip into `datasets/` and run the entirety of `dataprep.ipynb` to prepare the images.

In line 112 of `evaluate_div2k.py`, add the paths to the model weights which you want to evaluate. Make sure to comment out the models which aren't present.

Run `python evaluate_div2k.py` to evaluate the models. The output will be saved in `div2k_results.csv`.
