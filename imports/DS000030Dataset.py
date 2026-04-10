import os
import re
import glob
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data


_DEFAULT_THRESHOLD = 0.2
_N_ROI = 189


class DS000030Dataset(InMemoryDataset):
    """
    PyG InMemoryDataset for the OpenNeuro ds000030 fMRI dataset.

    Expected directory layout::

        root/
          sub-XXXXX_task-rest_atlas-cc200_timeseries.1D  (space-separated)
          ...
          Phenotypic_V1_0b_preprocessed1_ds000030.csv

    CSV format::

        SUB_ID,DX_GROUP
        sub-10159,0
        sub-10171,1
        ...

    Timeseries files:  shape = [T, 189]  (189 ROI columns, space-separated)

    Graph construction
    ------------------
    * Nodes      : ROIs  (189)
    * Node features (x) : Pearson FC row  [189, 189]
    * Edges      : |FC[i,j]| > pearson_threshold  (no self-loops)
    * edge_attr  : FC[i,j]
    * pos        : zero vectors [189, 3]  (no MNI coords available for 189-ROI atlas)

    Parameters
    ----------
    root : str
        Directory containing .1D files and the phenotypic CSV.
    name : str, optional
        Dataset identifier tag; controls the processed sub-directory name.
    pearson_threshold : float, optional
        Threshold for edge construction (default 0.2).
    """

    def __init__(self, root, name='DS000030',
                 pearson_threshold=_DEFAULT_THRESHOLD,
                 transform=None, pre_transform=None):
        self.name = name
        self.pearson_threshold = pearson_threshold
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
    # Label loading
    # ------------------------------------------------------------------

    def _load_labels(self):
        """Return dict {sub_id_str: int_label} from the phenotypic CSV."""
        label_map = {}
        # Try common CSV file names
        candidates = sorted(glob.glob(os.path.join(self.root, '*.csv')))
        # Prefer the official ds000030 phenotypic file
        preferred = [c for c in candidates
                     if 'ds000030' in os.path.basename(c).lower()
                     or 'phenotypic' in os.path.basename(c).lower()]
        search_order = preferred + [c for c in candidates if c not in preferred]

        for csv_path in search_order:
            with open(csv_path, encoding='utf-8') as f:
                header = None
                id_col, dx_col = 0, 1
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip().strip('"') for p in line.split(',')]
                    if header is None:
                        header = [p.upper() for p in parts]
                        id_col = next(
                            (i for i, h in enumerate(header)
                             if h in ('SUB_ID', 'SUBJECT', 'SUBJECT_ID', 'ID')),
                            0)
                        dx_col = next(
                            (i for i, h in enumerate(header)
                             if h in ('DX_GROUP', 'DX', 'LABEL', 'GROUP', 'DIAGNOSIS')),
                            1)
                        continue
                    if len(parts) <= max(id_col, dx_col):
                        continue
                    sid = parts[id_col]
                    try:
                        lbl = int(float(parts[dx_col]))
                    except ValueError:
                        continue
                    label_map[sid] = lbl
            if label_map:
                return label_map

        return label_map

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self):
        label_map = self._load_labels()

        if not label_map:
            raise RuntimeError(
                f"No labels loaded for DS000030. "
                f"Expected a CSV with SUB_ID and DX_GROUP columns in '{self.root}'."
            )

        # Collect all .1D files and extract sub-IDs
        all_1d = sorted(glob.glob(os.path.join(self.root, '*.1D')))

        # Match by sub-XXXXX pattern in filename
        _sub_re = re.compile(r'(sub-\d+)')

        pos = torch.zeros(_N_ROI, 3, dtype=torch.float)  # no MNI coords for 189-ROI atlas

        data_list = []
        for fpath in all_1d:
            fname = os.path.basename(fpath)
            m = _sub_re.search(fname)
            if m is None:
                continue
            sub_id = m.group(1)
            if sub_id not in label_map:
                continue
            lbl = label_map[sub_id]

            try:
                ts = np.loadtxt(fpath)          # [T, N_ROI]
            except Exception:
                continue

            if ts.ndim == 1:
                ts = ts.reshape(1, -1)

            # Ensure correct number of ROI columns
            if ts.shape[1] < _N_ROI:
                pad = np.zeros((ts.shape[0], _N_ROI - ts.shape[1]), dtype=np.float32)
                ts = np.hstack([ts, pad])
            elif ts.shape[1] > _N_ROI:
                ts = ts[:, :_N_ROI]

            # Pearson correlation matrix [N_ROI, N_ROI]
            with np.errstate(invalid='ignore', divide='ignore'):
                fc = np.corrcoef(ts.T)
            fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

            fc_t = torch.tensor(fc, dtype=torch.float)

            # Node features: each node's FC row vector
            x = fc_t.clone()
            x[x == float('inf')] = 0.0
            x = torch.nan_to_num(x, nan=0.0)

            # Edge construction: |FC[i,j]| > threshold, no self-loops
            mask = (torch.abs(fc_t) > self.pearson_threshold)
            mask.fill_diagonal_(False)
            edge_index = mask.nonzero(as_tuple=False).t().contiguous()   # [2, E]
            edge_attr  = fc_t[edge_index[0], edge_index[1]]              # [E]

            y = torch.tensor([lbl], dtype=torch.long)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                pos=pos,
            )
            data_list.append(data)

        if not data_list:
            raise RuntimeError(
                f"No graphs constructed for DS000030. "
                f"Check that .1D files exist in '{self.root}' and "
                f"subject IDs match the phenotypic CSV."
            )

        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
