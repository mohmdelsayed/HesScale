import unittest, os, json, shutil
from core.logger import Logger

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger("test_logs")

    def test_log(self):
        # Define test data
        kwargs = {
            "task": "dummy_task",
            "network": "dummy_network",
            "exp_name": "dummy_exp_name",
            "learner": "adam",
            "seed": 0,
            "optimizer_hps": {
                "lr": 0.01,
                "beta1": 0.9,
                "beta2": 0.9,
            }
        }

        # Call log method to save the data
        self.logger.log(**kwargs)

        # Check if the file exists
        file_dir = f"{self.logger.log_dir}/{kwargs['exp_name']}/{kwargs['task']}/{kwargs['learner']}/{kwargs['network']}/lr_{kwargs['optimizer_hps']['lr']}_beta1_{kwargs['optimizer_hps']['beta1']}_beta2_{kwargs['optimizer_hps']['beta2']}/{kwargs['seed']}.json"
        self.assertTrue(os.path.exists(file_dir))

        # Check if the data is saved correctly
        with open(file_dir) as f:
            data = json.load(f)
        self.assertEqual(data['task'], kwargs['task'])
        self.assertEqual(data['learner'], kwargs['learner'])
        self.assertEqual(data['network'], kwargs['network'])
        self.assertEqual(data['seed'], kwargs['seed'])
        self.assertEqual(data['optimizer_hps'], kwargs['optimizer_hps'])

        # remove artifcats
        shutil.rmtree(self.logger.log_dir)	

if __name__ == '__main__':
    unittest.main()