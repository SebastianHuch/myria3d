import os
import os.path as osp
import sys
import copy
import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from tqdm import tqdm

from myria3d.models.model import Model

sys.path.append(osp.dirname(osp.dirname(__file__)))
from myria3d.models.interpolation import Interpolator  # noqa
from myria3d.utils import utils  # noqa

log = utils.get_logger(__name__)


@utils.eval_time
def predict(config: DictConfig) -> str:
    """
    Inference pipeline.

    A lightning datamodule splits a single point cloud of arbitrary size (typically: 1km * 1km) into subtiles
    (typically 50m * 50m), which are grouped into batches that are fed to a trained neural network embedded into a lightning Module.

    Predictions happen on a subsampled version of each subtile, which needs to be propagated back to the complete
    point cloud via an Interpolator. This Interpolator also includes the creation of a new LAS file with additional
    dimensions, including predicted classification, entropy, and (optionnaly) predicted probability for each class.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        str: path to ouptut LAS.

    """

    # Those are the 2 needed inputs, in addition to the hydra config.
    assert os.path.exists(config.predict.ckpt_path)
    assert os.path.exists(config.predict.src_las)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_predict_data(config.predict.src_las)

    # Do not require gradient for faster predictions
    torch.set_grad_enabled(False)
    model = Model.load_from_checkpoint(config.predict.ckpt_path)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    # TODO: Interpolator could be instantiated directly via hydra.
    itp = Interpolator(
        interpolation_k=config.predict.interpolator.interpolation_k,
        classification_dict=config.dataset_description.get("classification_dict"),
        probas_to_save=config.predict.interpolator.probas_to_save,
        predicted_classification_channel=config.predict.interpolator.get(
            "predicted_classification_channel", "PredictedClassification"
        ),
        entropy_channel=config.predict.interpolator.get("entropy_channel", "entropy"),
        weighted=config.predict.interpolator.get("weighted", True),
    )
    use_tta = bool(config.predict.get("use_tta", False))
    AUGS = [
        {"flip_x": False, "flip_y": False},
        {"flip_x": True,  "flip_y": False},
        {"flip_x": False, "flip_y": True},
        {"flip_x": True,  "flip_y": True},
    ]

    for batch in tqdm(datamodule.predict_dataloader()):
        batch.to(device)

        if use_tta:
            logits_acc = None
            for a in AUGS:
                b = copy.copy(batch)
                b.pos = batch.pos.clone()
                if a["flip_x"]:
                    b.pos[:, 0] *= -1
                if a["flip_y"]:
                    b.pos[:, 1] *= -1
                l = model.predict_step(b)["logits"]
                logits_acc = l if logits_acc is None else logits_acc + l
            logits_out = logits_acc / len(AUGS)
        else:
            logits_out = model.predict_step(batch)["logits"]

        itp.store_predictions(logits_out, batch.idx_in_original_cloud)

    grid_m = config.datamodule.transforms.preparations.predict.GridSampling._args_[0]
    grid_mm = int(grid_m * 1000)
    overlap_m = int(config.predict.subtile_overlap)
    max_nodes = int(config.datamodule.transforms.preparations.predict.MaximumNumNodes._args_[0])

    out_f = itp.reduce_predictions_and_save(
        config.predict.src_las,
        config.predict.output_dir,
        config.datamodule.get("epsg"),
        grid_resolution_mm=grid_mm,
        subtile_overlap_m=overlap_m,
        max_points=max_nodes,
    )
    return out_f
