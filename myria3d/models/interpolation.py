import logging
import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pdal
import torch
from torch.distributions import Categorical
from torch_scatter import scatter_sum
from torch_cluster import knn

from pdaltools import las_info

from myria3d.pctl.dataset.utils import get_pdal_info_metadata, get_pdal_reader

log = logging.getLogger(__name__)


class Interpolator:
    """A class to load, update with classification, update with probas (optionnal), and save a LAS."""

    def __init__(
        self,
        interpolation_k: int = 10,
        classification_dict: Dict[int, str] = {},
        probas_to_save: Union[List[str], Literal["all"]] = "all",
        predicted_classification_channel: Optional[str] = "PredictedClassification",
        entropy_channel: Optional[str] = "entropy",
        weighted: bool = True,
    ):
        """Initialization method.
        Args:
            interpolation_k (int, optional): Number of Nearest-Neighboors for inverse-distance averaging of logits. Defaults 10.
            classification_dict (Dict[int, str], optional): Mapper from classification code to class name (e.g. {6:building}). Defaults {}.
            probas_to_save (List[str] or "all", optional): Specific probabilities to save as new LAS dimensions.
            Override with None for no saving of probabilities. Defaults to "all".


        """

        self.k = interpolation_k
        self.weighted = weighted
        self.classification_dict = classification_dict
        self.predicted_classification_channel = predicted_classification_channel
        self.entropy_channel = entropy_channel

        if probas_to_save == "all":
            self.probas_to_save = list(classification_dict.values())
        elif probas_to_save is None:
            self.probas_to_save = []
        else:
            self.probas_to_save = probas_to_save

        # Maps ascending index (0,1,2,...) back to conventionnal LAS classification codes (6=buildings, etc.)
        self.reverse_mapper: Dict[int, int] = {
            class_index: class_code
            for class_index, class_code in enumerate(classification_dict.keys())
        }

        self.logits: List[torch.Tensor] = []
        self.idx_in_full_cloud_list: List[np.ndarray] = []

    def load_full_las_for_update(self, src_las: str, epsg: str) -> Tuple[np.ndarray, Dict]:
        """Loads a LAS and adds necessary extradim.

        Args:
            filepath (str): Path to LAS for which predictions are made.
            epsg (str): epsg to force the reading with
        """
        # We do not reset the dims we create channel.
        # Slight risk of interaction with previous values, but it is expected that all non-artefacts values are updated.
        pipeline = pdal.Pipeline() | get_pdal_reader(src_las, epsg)
        for proba_channel_to_create in self.probas_to_save:
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{proba_channel_to_create}")
            pipeline |= pdal.Filter.assign(value=f"{proba_channel_to_create}=0")

        if self.predicted_classification_channel:
            # Copy from Classification to preserve data type
            # Also preserves values of artefacts.
            if self.predicted_classification_channel != "Classification":
                pipeline |= pdal.Filter.ferry(
                    dimensions=f"Classification=>{self.predicted_classification_channel}"
                )

        if self.entropy_channel:
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{self.entropy_channel}")
            pipeline |= pdal.Filter.assign(value=f"{self.entropy_channel}=0")

        pipeline.execute()
        writer_params = las_info.get_writer_parameters_from_reader_metadata(
            pipeline.metadata, a_srs=f"EPSG:{epsg}" if str(epsg).isdigit() else epsg
        )
        return pipeline.arrays[0], writer_params

    def store_predictions(self, logits, idx_in_original_cloud) -> None:
        """Keep a list of predictions made so far."""
        self.logits += [logits]
        self.idx_in_full_cloud_list += idx_in_original_cloud

    @torch.no_grad()
    def reduce_predicted_logits(self, nb_points) -> Tuple[torch.Tensor, np.ndarray]:
        """Interpolate logits to points without predictions using an inverse-distance weightning scheme.

        Returns:
            torch.Tensor, torch.Tensor: interpolated logits classification

        """

        # Concatenate elements from different batches
        logits: torch.Tensor = torch.cat(self.logits).cpu()
        idx_in_full_cloud: np.ndarray = np.concatenate(self.idx_in_full_cloud_list)
        self.logits, self.idx_in_full_cloud_list = [], []        # free mem

        # We scatter_sum logits based on idx, in case there are multiple predictions for a point.
        # scatter_sum reorders logits based on index,they therefore match las order.
        reduced_logits = torch.zeros((nb_points, logits.size(1)))
        scatter_sum(logits, torch.from_numpy(idx_in_full_cloud), out=reduced_logits, dim=0)
        # reduced_logits contains logits ordered by their idx in original cloud !
        # We need to select the points for which we have a prediction via idx_in_full_cloud.
        # NB1 : some points may not contain any predictions if they were in small areas.

        return reduced_logits[idx_in_full_cloud], idx_in_full_cloud

    @torch.no_grad()
    def reduce_predictions_and_save(
        self,
        raw_path: str,
        output_dir: str,
        epsg: str,
        grid_resolution_mm: Optional[int] = None,
        subtile_overlap_m: Optional[int] = None,
        max_points: Optional[int] = None,
    ) -> str:
        """Interpolate all predicted probabilites to their original points in LAS file, and save.

        Args:
            raw_path (str): Path of input LAS.
            output_dir (str): Directory to save output LAS with new predicted classification, entropy,
            and probabilities.
            epsg (str): epsg to force the reading with
            grid_resolution_mm (int, optional): grid cell size in millimeters for filename suffix.
            subtile_overlap_m (int, optional): subtile overlap in meters for filename suffix.
            max_points (int, optional): MaximumNumNodes value for filename suffix.
        Returns:
            str: path of the updated, saved LAS file.

        """
        basename = os.path.basename(raw_path)
        # Read number of points only from las metadata in order to minimize memory usage
        nb_points = get_pdal_info_metadata(raw_path)["count"]

        logits_reduced, idx_in_full_cloud = self.reduce_predicted_logits(nb_points)

        # ---------- distance‑weighted k‑NN interpolation --------------------
        full_idx = torch.arange(nb_points, dtype=torch.long)
        known_mask = torch.zeros(nb_points, dtype=torch.bool)
        known_mask[idx_in_full_cloud] = True

        if (~known_mask).any() and self.k > 0:
            # Read XYZ early for k‑NN
            las, writer_params = self.load_full_las_for_update(raw_path, epsg)
            xyz = np.vstack((las["X"], las["Y"], las["Z"])).T.astype(np.float32)
            xyz = torch.from_numpy(xyz)

            src_pos = xyz[known_mask]          # N_known × 3
            dst_pos = xyz[~known_mask]         # N_missing × 3

            # Build logits tensor aligned with src_pos order
            full_logits = torch.zeros_like(logits_reduced)
            full_logits[idx_in_full_cloud] = logits_reduced[idx_in_full_cloud]
            src_logits = full_logits[known_mask]

            # k‑NN (src ↔ dst)
            row, col = knn(src_pos, dst_pos, self.k)       # src index, dst index
            d2 = (dst_pos[col] - src_pos[row]).pow(2).sum(1).clamp_min(1e-12)
            
            if self.weighted:
                w = 1.0 / d2 # inverse‑distance weights
            else:
                w = torch.ones_like(d2)

            miss_logits = torch.zeros(dst_pos.size(0), logits_reduced.size(1))
            miss_logits.index_add_(0, col, src_logits[row] * w.unsqueeze(1))
            norm = torch.zeros(dst_pos.size(0), 1)
            norm.index_add_(0, col, w.unsqueeze(1))
            miss_logits = miss_logits / norm

            # Insert back
            full_logits[~known_mask] = miss_logits
            logits_reduced = full_logits
            idx_in_full_cloud = full_idx.numpy()

        else:
            # We already loaded LAS if interpolation was skipped
            las, writer_params = self.load_full_las_for_update(raw_path, epsg)
        # ---------------------------------------------------------------------

        probas = torch.nn.Softmax(dim=1)(logits_reduced)

        if self.predicted_classification_channel:
            preds = torch.argmax(logits_reduced, dim=1)
            preds = np.vectorize(self.reverse_mapper.get)(preds)

        # ------------ write to LAS -------------------------------------------
        for idx, class_name in enumerate(self.classification_dict.values()):
            if class_name in self.probas_to_save:
                las[class_name][:] = probas[:, idx].numpy()

        if self.predicted_classification_channel:
            las[self.predicted_classification_channel][:] = preds
            log.info(
                f"Saving predicted classes to channel {self.predicted_classification_channel}."
                "Channel name can be changed by setting `predict.interpolator.predicted_classification_channel`."
            )
            del preds

        if self.entropy_channel:
            las[self.entropy_channel][:] = Categorical(probs=probas).entropy().numpy()
            log.info(
                f"Saving Shannon entropy of probabilities to channel {self.entropy_channel}."
                "Channel name can be changed by setting `predict.interpolator.entropy_channel`"
            )

        os.makedirs(output_dir, exist_ok=True)
        name, ext = os.path.splitext(basename)
        if grid_resolution_mm is not None and subtile_overlap_m is not None:
            suffix = f"_grid{grid_resolution_mm:04d}_tile{subtile_overlap_m:04d}"
            if max_points is not None:
                suffix += f"_maxpoints{max_points:06d}"
            new_basename = f"{name}{suffix}{ext}"
        else:
            new_basename = basename

        out_f = os.path.abspath(os.path.join(output_dir, new_basename))
        log.info(f"Updated LAS ({new_basename}) will be saved to: \n {output_dir}\n")
        log.info("Saving...")
        writer_params["extra_dims"] = "all"
        pipeline = pdal.Writer.las(filename=out_f, **writer_params).pipeline(las)
        pipeline.execute()
        log.info("Saved.")

        return out_f
