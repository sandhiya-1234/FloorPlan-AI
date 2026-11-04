# dataset/

This directory holds **no data** and is intentionally almost empty.

The learned pipeline writes its prepared data to `data/seg/` (git-ignored) from the official CubiCasa5K archive; see
[`docs/dataset.md`](../docs/dataset.md) for provenance, the source-to-target class mapping, split and duplicate policy, and what is
still unverified. How to produce it:

```bash
# on Google Colab, via notebooks/colab_segmentation.ipynb (Stage A), or by hand:
python -m training.prepare_dataset inspect --zip cubicasa5k.zip --n 60
python -m training.prepare_dataset prepare --zip cubicasa5k.zip --out data/seg
```

- Dataset: CubiCasa5K, https://zenodo.org/records/2613548 (DOI 10.5281/zenodo.2613548), 5.5 GB zip, md5 `0ce0b203d1e3c125b51087b219bd23b9`.
- Licence: CC BY-NC-SA 4.0 on Zenodo (the official repository says CC BY-NC 4.0): non-commercial, share-alike. Do not redistribute the data.
- The old 15-file synthetic set was moved to `docs/audit/invalid_artifacts/synthetic_fixture_dataset/`. It is not real data and is not used.
- `preprocessing/download_dataset.py` generates synthetic fixtures and is deprecated. Do not use it to create data.
- `dataset/raw`, `dataset/processed`, `dataset/train`, `dataset/validation`, `dataset/test` belong to the earlier Qwen plan and stay empty.
