# PRD-08: Dataset Loading & Processor Pipeline

> **Status:** TODO
> **Priority:** P1 — Required for real data training
> **Dependencies:** PRD-02 (tensor_ops)
> **Estimated LOC:** ~500
> **Phase:** 3 (Data Pipeline)

---

## Objective

Port the LeRobotDataset loading and processor pipeline to return `mx.array` instead of `torch.Tensor`. Keep the Parquet + video (MP4) format untouched — only the tensor conversion layer changes.

---

## Architecture

```
HuggingFace Hub / Local Files
  ├── Parquet files (tabular: states, actions, timestamps)
  ├── MP4 videos or JPEG images
  └── metadata.json

        ↓ LeRobotDataset

  ├── Read Parquet → pandas/arrow → numpy → mx.array
  ├── Decode video → OpenCV/PIL → numpy → mx.array
  └── Return dict of mx.array tensors

        ↓ Processor Pipeline

  ├── Normalize (mean/std)
  ├── Delta actions (action[t] - action[t-1])
  ├── Image transforms (resize, crop, normalize)
  └── Batch collation
```

---

## What Changes vs Upstream

| Upstream | Our Version | Change |
|----------|-------------|--------|
| `torch.tensor(data)` | `mx.array(data)` | Tensor creation |
| `torch.stack(tensors)` | `mx.stack(tensors)` | Batch collation |
| `torch.from_numpy(arr)` | `mx.array(arr)` | Numpy conversion |
| `torchvision.transforms` | PIL/OpenCV + mx.array | Image transforms |
| `torch.utils.data.Dataset` | Python `__getitem__` | No DataLoader needed |
| `torch.utils.data.DataLoader` | Simple Python iterator | Batching |

## What Does NOT Change

- Parquet reading (arrow/pandas — no torch)
- Video decoding (OpenCV/torchcodec → frames as numpy)
- HuggingFace Hub download (huggingface_hub — no torch)
- Dataset metadata format (JSON)
- Episode structure (episode_index, frame_index)

---

## Deliverables

### 1. `datasets/lerobot_dataset.py` — Minimal Port

```python
class LeRobotDatasetMLX:
    """Wrapper around upstream LeRobotDataset that returns mx.array."""

    def __init__(self, repo_id, **kwargs):
        # Use upstream's data loading (Parquet + video)
        # Only intercept the tensor conversion at the end
        ...

    def __getitem__(self, idx):
        """Return dict of mx.array instead of torch.Tensor."""
        item = self._load_item(idx)
        return {k: mx.array(np.asarray(v)) for k, v in item.items()}

    def __len__(self):
        return self._num_frames
```

### 2. `processor/processor_mlx.py` — Processing Pipeline

```python
class ProcessorMLX:
    """Pre/post processing pipeline using mx.array."""

    def normalize(self, tensor, mean, std):
        return (tensor - mx.array(mean)) / mx.array(std)

    def unnormalize(self, tensor, mean, std):
        return tensor * mx.array(std) + mx.array(mean)

    def compute_delta(self, actions):
        """action[t] - action[t-1] for delta action space."""
        return actions[1:] - actions[:-1]

    def process_images(self, images):
        """Resize + normalize images for policy input."""
        # images: (B, H, W, C) numpy uint8
        # output: (B, C, H, W) mx.array float32 normalized
        ...
```

### 3. `datasets/dataloader.py` — Simple Batching

```python
class SimpleDataLoader:
    """Minimal dataloader — no multiprocessing, no prefetch.

    MLX unified memory + lazy eval makes simple iteration efficient enough.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            items = [self.dataset[j] for j in batch_indices]
            yield _collate(items)

def _collate(items):
    """Stack list of dicts into batched dict of mx.arrays."""
    batch = {}
    for key in items[0]:
        batch[key] = mx.stack([item[key] for item in items])
    return batch
```

---

## Acceptance Criteria

1. `LeRobotDatasetMLX` loads a real dataset from HF Hub
2. `dataset[0]` returns dict of `mx.array` (not torch.Tensor)
3. `SimpleDataLoader` iterates and produces batched mx.arrays
4. Normalization roundtrip: `unnormalize(normalize(x)) ≈ x`
5. Delta action computation is correct
6. Image processing produces correct shape/dtype
7. 15+ tests passing

---

## Notes

- We CAN use upstream's Parquet/video loading code unchanged (no torch in data reading)
- The key insight: only the **output conversion** (numpy → mx.array) needs to change
- No need for multiprocessing DataLoader — MLX lazy eval + unified memory is fast enough
- Video decoding stays in OpenCV/numpy — convert to mx.array at the batch level
- For Phase 1 (synthetic data), we can skip real dataset loading entirely
