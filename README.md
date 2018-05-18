## Adversarial Regression

This project is designed to create adversarial noise for various autoencoders. It uses convex optimization to create the ideal input for a maximum output perturbation.

### Requirements

```
python 3.6
tensorflow 1.7
```

### Datasets

Larger datasets are currently not included in the repository. To run existing models you will need to add the necessary datasets to `/datasets/data`.

Your directory structure should look similar to this after adding CIFAR, STL10 and MNIST.

```
/datasets/data/set14
/datasets/data/stl10
/datasets/data/cifar-10-batches-py
/datasets/data/t10k-images-idx3-ubyte.gz
/datasets/data/train-images-idx3-ubyte.gz
```

### Available Models

fcnn (MNIST)
fcnn2 (MNIST)
fcnn3 (CIFAR)
aen_stl10 (STL10)
koala (SET14)
c_dcscn (SET14)

### Run Models

Pretrained models are available through the parser class. Default models can be added to the `/models/models.json` file.

```
python adversarial.py --model fcnn --norm l2
python adversarial.py --model fcnn2 --norm linf
python adversarial.py --model aen_stl10 --norm linf
python adversarial.py --model koala

python adv_2.py --model c_dcscn
```

`adv.py` currently supports RGB, Greyscale and YCbCr autoencoders as well as colorization. The loss function for colorization is adjusted through the description attribute in `models.json` file.

### Images & Figures

Images and figures are available through `/results/images/`.
