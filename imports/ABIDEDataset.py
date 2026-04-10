import os
import json
import glob
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data


# Number of ROIs for supported dataset names
_NROI_MAP = {
    'ABIDE': 182,
    'ADHD':  137,
}

# Pearson correlation threshold for edge construction
_DEFAULT_THRESHOLD = 0.2


class ABIDEDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for ABIDE / ADHD fMRI graph classification.

    Expects data to be organised in BrainGNN-style directories::

        root/
          <sub_id>/
            rois_cc200.1D   (or rois_aal.1D / *.1D)
          phenotypic.csv    # columns: SUB_ID, DX_GROUP  (0 or 1)

    Graph construction
    ------------------
    * Nodes  : ROIs
    * Node features (x) : Pearson FC row vector  [N_ROI, N_ROI]
    * Edges  : |FC[i,j]| > pearson_threshold  (no self-loops)
    * Edge weights (edge_attr) : FC[i,j]
    * pos    : 3-D MNI coordinates from ``cc200_roi_coords_182.json``
               (only for ABIDE/182-ROI datasets; zeros otherwise)

    Parameters
    ----------
    root : str
        Path to data directory (containing subject sub-dirs and phenotypic.csv).
    name : str
        Dataset identifier – ``'ABIDE'`` or ``'ADHD'``.
    pearson_threshold : float, optional
        Threshold for binarising the FC matrix into an adjacency matrix.
    transform, pre_transform : optional
        Standard PyG transforms.
    """

    def __init__(self, root, name,
                 pearson_threshold=_DEFAULT_THRESHOLD,
                 transform=None, pre_transform=None):
        self.name = name
        self.pearson_threshold = pearson_threshold
        self.nroi = _NROI_MAP.get(name, 182)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    # PyG directory / file-name properties
    # ------------------------------------------------------------------

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return os.path.join(self.root, f'processed_{self.name}')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    # ------------------------------------------------------------------
    # ROI coordinates
    # ------------------------------------------------------------------

    def _load_roi_coords(self):
        """Return [N_ROI, 3] tensor of MNI coordinates, or zeros."""
        # Look for the coord file next to this module or in the repo root
        search_dirs = [
            os.path.dirname(os.path.abspath(__file__)),   # imports/
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
            self.root,
        ]
        for d in search_dirs:
            fpath = os.path.join(d, 'cc200_roi_coords_182.json')
            if os.path.isfile(fpath):
                with open(fpath) as f:
                    coords = json.load(f)
                arr = np.array(coords, dtype=np.float32)
                if arr.shape[0] == self.nroi:
                    arr_min, arr_max = arr.min(0), arr.max(0)
                    rng = arr_max - arr_min
                    rng[rng == 0] = 1.0
                    arr = (arr - arr_min) / rng
                    return torch.tensor(arr, dtype=torch.float)
        # Fall back to zero vectors
        return torch.zeros(self.nroi, 3, dtype=torch.float)

    # ------------------------------------------------------------------
    # Label loading
    # ------------------------------------------------------------------

    def _load_labels(self):
        """Return dict {sub_id_str: int_label}."""
        label_map = {}
        csv_candidates = [
            os.path.join(self.root, 'phenotypic.csv'),
            os.path.join(self.root, f'phenotypic_{self.name}.csv'),
        ]
        for csv_path in csv_candidates:
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path) as f:
                header = None
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if header is None:
                        header = [p.strip().strip('"') for p in parts]
                        # find column indices
                        id_col = next(
                            (i for i, h in enumerate(header)
                             if h.upper() in ('SUB_ID', 'SUBJECT', 'SUBJECT_ID', 'ID')),
                            0)
                        dx_col = next(
                            (i for i, h in enumerate(header)
                             if h.upper() in ('DX_GROUP', 'DX', 'LABEL', 'GROUP', 'DIAGNOSIS')),
                            1)
                        continue
                    if len(parts) <= max(id_col, dx_col):
                        continue
                    sid = parts[id_col].strip().strip('"')
                    try:
                        lbl = int(float(parts[dx_col].strip().strip('"')))
                    except ValueError:
                        continue
                    label_map[sid] = lbl
            if label_map:
                return label_map
        return label_map

    # ------------------------------------------------------------------
    # Find subject data files
    # ------------------------------------------------------------------

    def _find_subjects(self, label_map):
        """
        Return list of (sub_id, data_file_path, label).

        Searches for .1D files either:
          * root/<sub_id>/rois_cc200.1D  (per-subject sub-directories)
          * root/<sub_id>_*.1D            (flat directory)
        """
        subjects = []

        # --- strategy 1: per-subject sub-directories -----------------
        for sub_id, lbl in label_map.items():
            sub_dir = os.path.join(self.root, str(sub_id))
            if not os.path.isdir(sub_dir):
                continue
            # prefer rois_cc200, then any .1D file
            candidates = sorted(glob.glob(os.path.join(sub_dir, '*.1D')))
            if not candidates:
                continue
            preferred = [c for c in candidates if 'cc200' in c or 'aal' in c]
            fpath = preferred[0] if preferred else candidates[0]
            subjects.append((sub_id, fpath, lbl))

        if subjects:
            return subjects

        # --- strategy 2: flat directory – files named <sub_id>_*.1D --
        all_1d = glob.glob(os.path.join(self.root, '*.1D'))
        for fpath in sorted(all_1d):
            fname = os.path.basename(fpath)
            # try to match a label_map key
            for sub_id, lbl in label_map.items():
                if fname.startswith(str(sub_id)):
                    subjects.append((sub_id, fpath, lbl))
                    break

        return subjects

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self):
        pos = self._load_roi_coords()           # [N_ROI, 3]
        label_map = self._load_labels()

        if not label_map:
            raise RuntimeError(
                f"No labels loaded for dataset '{self.name}'. "
                f"Expected phenotypic.csv in '{self.root}'."
            )

        subjects = self._find_subjects(label_map)

        if not subjects:
            raise RuntimeError(
                f"No subject data files found in '{self.root}'. "
                "Expected per-subject sub-directories with .1D files "
                "or flat directory layout."
            )

        data_list = []
        for sub_id, fpath, lbl in subjects:
            try:
                ts = np.loadtxt(fpath)          # [T, N_ROI]
            except Exception:
                continue

            if ts.ndim == 1:
                ts = ts.reshape(-1, 1)

            # Truncate / pad ROI dimension
            n_roi = self.nroi
            if ts.shape[1] < n_roi:
                pad = np.zeros((ts.shape[0], n_roi - ts.shape[1]), dtype=np.float32)
                ts = np.hstack([ts, pad])
            elif ts.shape[1] > n_roi:
                ts = ts[:, :n_roi]

            # Pearson correlation matrix [N_ROI, N_ROI]
            with np.errstate(invalid='ignore', divide='ignore'):
                fc = np.corrcoef(ts.T)
            fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

            fc_t = torch.tensor(fc, dtype=torch.float)

            # Node features: each node's feature = its FC row
            x = fc_t.clone()
            x[x == float('inf')] = 0.0
            x = torch.nan_to_num(x, nan=0.0)

            # Edge construction: |FC[i,j]| > threshold, no self-loops
            mask = (torch.abs(fc_t) > self.pearson_threshold)
            mask.fill_diagonal_(False)
            edge_index = mask.nonzero(as_tuple=False).t().contiguous()   # [2, E]
            edge_attr  = fc_t[edge_index[0], edge_index[1]]              # [E]

            y = torch.tensor([lbl], dtype=torch.long)

            # pos: per-node coordinates (broadcast to all nodes)
            node_pos = pos  # [N_ROI, 3]  (already loaded above)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                pos=node_pos,
            )
            data_list.append(data)

        if not data_list:
            raise RuntimeError(
                f"Could not construct any graphs for dataset '{self.name}'. "
                "Check that .1D files are readable and contain valid timeseries."
            )

        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
