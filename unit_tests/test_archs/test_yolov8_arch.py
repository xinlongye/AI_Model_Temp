import sys
sys.path.append(r"D:\Code\Projects\python\AI_Model_Temp\archs")

import unittest
import yaml
import torch
from archs import build_network
import torchinfo

class MyTestCase(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_yolov8_net(self):
        config_path = r"D:\Code\Projects\python\AI_Model_Temp\options\seg_u2net_duts_option.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.option_data = yaml.load(f, Loader=yaml.FullLoader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_g = build_network(self.option_data['network'])
        self.net_g = self.net_g.to(self.device)
        torchinfo.summary(self.net_g, input_size=(1, 3, 640, 640), device="cpu")


if __name__ == '__main__':
    unittest.main()
