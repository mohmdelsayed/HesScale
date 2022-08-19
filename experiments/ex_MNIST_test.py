from experiments.approximation_quality.data_generator import InfiniteMNIST

for (x, y) in InfiniteMNIST():
    print(x.shape, y)