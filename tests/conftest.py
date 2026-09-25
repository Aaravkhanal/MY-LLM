import torch


def pytest_configure(config):
    torch.manual_seed(0)
