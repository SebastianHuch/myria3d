import torch
import random

class RandomBrightness:
    def __init__(self, delta=0.15): self.delta = delta
    def __call__(self, data):
        if random.random() < 0.5:
            shift = (random.uniform(-self.delta, self.delta))
            data.x[:, :3] = (data.x[:, :3] + shift).clamp(0, 1)
        return data

class RandomHue:
    def __init__(self, delta=0.10): self.delta = delta
    def __call__(self, data):
        if random.random() < 0.5:
            h = torch.rand(1).item()*self.delta*2 - self.delta
            data.x[:, :3] = (data.x[:, :3].roll(shifts=1, dims=1) + h).clamp(0, 1)
        return data
