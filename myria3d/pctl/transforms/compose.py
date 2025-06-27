from typing import Callable, List

from torch_geometric.transforms import BaseTransform, GridSampling
from myria3d.utils.utils import calc_avg_point_density


class CustomCompose(BaseTransform):
    """
    Composes several transforms together.
    Edited to bypass downstream transforms if None is returned by a transform.
    Args:
        transforms (List[Callable]): List of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data):
        # avg_density = calc_avg_point_density(data["pos"].numpy())
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
                data = [d for d in data if d is not None and d.num_nodes != 0]
                if len(data) == 0:
                    return None
            else:
                num_pts = data["pos"].shape[0]
                data = transform(data)
                # if type(transform) == GridSampling:
                #     avg_density_aft = calc_avg_point_density(data["pos"].numpy())
                #     print(f"Average point density: {avg_density:.2f} -> {avg_density_aft:.2f} points/m²")
                    print(f"Points: {num_pts} -> {data['pos'].shape[0]}")
                if data is None or data.num_nodes == 0:
                    return None
        return data
