from hess_bench.data_generator import InfiniteMNIST

for (x, y) in InfiniteMNIST():
    print(x.shape, y)