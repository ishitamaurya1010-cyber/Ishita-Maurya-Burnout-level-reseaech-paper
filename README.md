# Ishita Maurya — Agentic AI (Dataset 2)

Notebook: [Conferencedt2_agenticAI.ipynb](Conferencedt2_agenticAI.ipynb)

Generates BERT (`bert-base-uncased`) `[CLS]` embeddings for the *Mental Distress Dataset*, plus NLTK sentence/word tokenization on the same text column.

The embedding pipeline is designed to be **memory-safe** — it feeds a `torch.utils.data.DataLoader` with a tokenizing `collate_fn` (padding is bounded per mini-batch), runs in `torch.inference_mode()`, moves work to GPU when available, and streams results to disk via `numpy.memmap` so RAM usage stays flat regardless of dataset size.

---

## 1. Requirements

- Python 3.10+
- ~2 GB free RAM (with default settings) and ~1 GB free disk for the embedding file
- Optional but recommended: an NVIDIA GPU with CUDA (Colab T4 works well)

### Python packages

```bash
pip install "transformers>=4.40" "torch>=2.2" numpy pandas matplotlib seaborn nltk psutil
```

### Dataset

Place `Mental Distress Dataset-original.csv` next to the notebook. In Colab, the notebook uploads it interactively via `google.colab.files.upload()`.

---

## 2. Running the notebook

### Option A — Google Colab (recommended)

1. Open [Conferencedt2_agenticAI.ipynb](Conferencedt2_agenticAI.ipynb) in Colab.
2. **Runtime → Change runtime type → GPU (T4)** — big speedup for the BERT cell.
3. Run cells top to bottom. When prompted, upload `Mental Distress Dataset-original.csv`.
4. The embedding cell writes `bert_cls_embeddings.npy` to the Colab working directory.

### Option B — Local (VS Code / Jupyter)

1. Create and activate a virtual environment, install the packages above.
2. Put `Mental Distress Dataset-original.csv` in the repo root.
3. Skip / remove the two Colab-only cells:
   - `!free -h` (Linux only)
   - `from google.colab import files; files.upload()`
4. Run cells top to bottom.

---

## 3. Tuning the embedding cell

The batched BERT cell exposes five knobs at the top. Tune these **before** trying anything more exotic:

| Setting | Default | When to change |
|---|---|---|
| `MODEL_NAME` | `bert-base-uncased` | Switch to `sentence-transformers/all-MiniLM-L6-v2` for ~5× speed and ~3× less memory (384-dim). |
| `BATCH_SIZE` | `16` | Lower to `8` or `4` if you hit OOM. Raise to `32` on a GPU with ≥12 GB. |
| `MAX_LENGTH` | `128` | Cap on tokens per row. `64` is fine for short posts; only raise if your text is long. |
| `OUTPUT_PATH` | `bert_cls_embeddings.npy` | Where the embeddings are streamed. |
| `NUM_WORKERS` | `0` | Raise to `2`–`4` **on Linux** if the GPU is idle waiting for tokenization. Keep `0` on Windows/Colab notebooks — worker processes don't play well there. |

The cell logs RAM (via `psutil`) and GPU memory (`torch.cuda.memory_allocated`) every `LOG_EVERY` batches so you can watch usage stay flat.

The final embeddings are reloaded as a **read-only `numpy.memmap`**:

```python
import numpy as np
embedding = np.load("bert_cls_embeddings.npy", mmap_mode="r")
print(embedding.shape)   # (n_rows, 768)
```

Downstream code can slice `embedding[i:j]` without pulling the whole file into RAM.

---

## 4. Troubleshooting

### `Session crashed after using all available RAM.`

Almost always the embedding step. Apply in this order:

1. Lower `BATCH_SIZE` (16 → 8 → 4).
2. Lower `MAX_LENGTH` (128 → 64).
3. Set `MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"`.
4. Save as `float16` to halve memory:
   ```python
   embeddings = np.lib.format.open_memmap(
       OUTPUT_PATH, mode="w+", dtype=np.float16, shape=(n_rows, hidden_size)
   )
   ```
5. Confirm you are **not** calling `tokenizer(df['text'].tolist(), ...)` on the whole column — that's the original bug (`batch_size=` is silently ignored by the HF tokenizer).

### `CUDA out of memory`

- Lower `BATCH_SIZE`.
- Add mixed precision around the forward pass:
  ```python
  with torch.autocast(device_type="cuda", dtype=torch.float16):
      outputs = model(**enc)
  ```
- Confirm nothing else is holding GPU memory: `torch.cuda.empty_cache()` and restart the runtime if a previous cell leaked tensors.

### `AttributeError: 'float' object has no attribute 'lower' / .translate`

`NaN` rows in the `text` column. Guard every string op:

```python
df['text'] = df['text'].fillna('').astype(str).str.lower()
```

### `LookupError: Resource punkt_tab not found` (or `punkt` not found)

NLTK resource missing. The tokenization cells attempt both names; if you're running an older NLTK manually:

```python
import nltk
nltk.download('punkt')       # NLTK < 3.8.2
nltk.download('punkt_tab')   # NLTK ≥ 3.8.2
```

### Warnings about `cls.seq_relationship.*` / `cls.predictions.*`

**Safe to ignore.** Those are the BERT pre-training heads (NSP + MLM). `AutoModel` loads only the encoder, so they are correctly discarded — nothing is broken.

### Notebook frontend hangs after a tokenization cell

The old versions of the sentence/word tokenization cells ended in a bare `sent_tokens` / `word_tokens`, which asked Jupyter to render the entire list as HTML and could freeze the browser. The current cells print a small preview instead. If you customize them, always slice before rendering:

```python
df['word_tokens'].head()
df['word_tokens'].iloc[0][:20]
```

### `FileNotFoundError: Mental Distress Dataset-original.csv`

Upload it via the Colab cell, or place it next to the notebook when running locally. The path is relative to the notebook's working directory.

### `!free -h` fails on Windows

`free` is a Linux command. Either remove that cell locally or replace it with:

```python
import psutil
print(psutil.virtual_memory())
```

---

## 5. Repository layout

```
.
├── Conferencedt2_agenticAI.ipynb   # main notebook
├── README.md                        # this file
└── (generated at runtime)
    └── bert_cls_embeddings.npy      # streamed embedding matrix
```

---

## 6. Notes on the pipeline

- A `torch.utils.data.Dataset` wraps the raw text list; a `DataLoader` with a `collate_fn` runs the tokenizer **per mini-batch**, so padding is bounded by the batch (not the corpus).
- The `Dataset` returns `(index, text)` and the loop writes each result into `embeddings[indices]` — order is preserved regardless of loader behavior.
- The embedding cell uses `torch.inference_mode()` (stronger than `torch.no_grad()` — also disables version counters) and `model.eval()` to skip dropout.
- `pin_memory=True` is enabled automatically on CUDA for faster host→GPU copies.
- Intermediate tensors (`enc`, `outputs`, `cls`) are `del`-ed each iteration and `torch.cuda.empty_cache()` is called on GPU runs.
- The `[CLS]` vector is the default sentence representation. The single-sentence demo cell also shows **mask-aware mean pooling**, which is usually a stronger alternative — swap it in if you want to.
