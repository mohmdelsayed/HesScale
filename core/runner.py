import os

class Runner:
    def __init__(self, run, learner, task, grid_search, exp_name):
        self.grid_search = grid_search
        self.exp_name = exp_name
        self.learner = learner
        self.task = task
        self.run = run

    def get_combinations(self):
        return self.grid_search.get_permutations()

    def write_cmd(self, directory, mode='w'):
        cmd = ""
        for permutation in self.get_combinations():
            cmd += f"python3 core/run/{self.run}.py --task {self.task} --learner {self.learner} --exp_name {self.exp_name}"
            keys, values = zip(*permutation.items())
            for key, value in zip(keys, values):
                cmd += f" --{key} {value}"
            cmd += "\n"

        dir = directory / self.exp_name
        os.makedirs(dir, exist_ok=True)

        with open(dir / f"{self.learner}.txt", mode) as f:
            f.write(cmd)

        return len(self.get_combinations())
