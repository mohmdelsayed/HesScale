import matplotlib.pyplot as plt
import numpy as np

methods = [
"Adam",
"SGD",
"BL89",
"AdaHessian",
"GGNMC",
"AdaHesScaleLM",
"AdaHesScale",
]

problems = ["mnist" ,
            "cifar10_3c3d",
            "cifar100_3c3d",
            "cifar100_all_cnn",
]

scalings = [40, 14, 14, 14] # adjust to match with the real computational time 

n_epochs = 100
batch_size = 128
train_size = 50000
test_size = 10000
means = np.load('data/ex_optimizer_times/update_times.npy')


for (problem, scaling) in zip(problems, scalings):
        comp_time = {}
        for method, mean in zip(methods, means.T):
                comp_time[method] = mean[3] * (train_size/batch_size) * n_epochs / scaling

        data = np.load(f'data/ex_optimizer_times/{problem}.npy', allow_pickle=True)
        for i, optimizer in enumerate(data):
                for setting in data[i]:
                        if setting["metric"] == "test_accuracies":
                                plt.plot([i*(comp_time[setting["optimizer_name"]]) for i, _ in enumerate(setting["center"])], setting["center"], label=setting["optimizer_name"])

        plt.legend()
        plt.xlabel("Time in seconds")
        plt.ylabel("Accuracy")
        plt.gcf().set_size_inches(10, 5)
        plt.savefig(f"data/ex_optimizer_times/{problem}.pdf")
        plt.clf()