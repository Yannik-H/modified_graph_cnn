# A generalization of Convolutional Neural Networks to Graph-Structured Data(Modified Version)

This is updated code to [https://github.com/hechtlinger/graph_cnn]() . Because the code from the link requires environment using "theano" as backend, which is super unconvinent for reproducing, I modifiy the code in function `call(self, x)` in graph_convolution.py and try to make it easier for deployment.

------------------

### Basic example
```python
cd /code/
python DPP4_graph_conv.py
```

------------------

### Merck Dataset
The DPP4 dataset is part of the Merck Molecular Activity Challenge, a previous [Kaggle](https://www.kaggle.com/c/MerckActivity) competition.

------------------

### Dependencies
```
conda create -n py35 python=3.5 && conda activate py35
```

```
conda install requirement.txt
```

```
pip install ipykernel
```


