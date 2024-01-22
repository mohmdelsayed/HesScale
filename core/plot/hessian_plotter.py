import json
import matplotlib.pyplot as plt
from core.best_config import BestConfig
import os
import numpy as np
import matplotlib
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

class HessianPlotter:
    def __init__(self, best_runs_path, task_name):
        self.best_runs_path = best_runs_path
        self.task_name = task_name

    def compute_diag_dominance(self, hessian_matrix):
        return np.linalg.norm(np.diag(np.diag(hessian_matrix))) / np.linalg.norm(hessian_matrix)

    def hessian_heatmap(self):
        layer = 0
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            Hs_before = []
            Hs_after = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    Hs_before.append(data["H_before"])
                    Hs_after.append(data["H_after"])

            Hs_before_avg = np.abs(np.array(Hs_before))[:,layer,:,:].mean(axis=0)
            diag_dominance_before = self.compute_diag_dominance(Hs_before_avg)
            Hs_after_avg = np.abs(np.array(Hs_after))[:,layer,:,:].mean(axis=0)
            diag_dominance_after = self.compute_diag_dominance(Hs_after_avg)
            cmap = plt.cm.get_cmap("RdBu")
            plt.imshow(
                Hs_before_avg,
                vmin=0.0,
                vmax=np.max(Hs_before_avg),
                cmap=cmap,
            )
            plt.colorbar()
            plt.title("Magitudes of Hessian matrix before training")
            plt.savefig(f"H_before{layer}.pdf", bbox_inches="tight", dpi=600)
            print("Diag Dominance Before: ", diag_dominance_before)
            plt.clf()

            plt.imshow(
                Hs_after_avg,
                vmin=0.0,
                vmax=np.max(Hs_after_avg),
                cmap=cmap,
            )
            plt.title("Magitudes of Hessian matrix after training")
            plt.colorbar()
            plt.savefig(f"H_after{layer}.pdf", bbox_inches="tight", dpi=600)
            print("Diag Dominance After: ", diag_dominance_after)

            rand_dominance = self.compute_diag_dominance(np.random.randn(*Hs_after_avg.shape))
            print("Random Dominance: ", rand_dominance)
if __name__ == "__main__":
    best_runs = BestConfig("exp5/stationary_emnist", "fcn_relu_single_hidden_layer", ["adam"]).get_best_run(measure="accuracies")
    print(best_runs)
    plotter = HessianPlotter(best_runs, task_name="Hessian on MNIST")
    plotter.hessian_heatmap()
