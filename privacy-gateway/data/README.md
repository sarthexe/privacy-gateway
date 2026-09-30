# Nemotron-PII data

`nvidia/Nemotron-PII` is not committed to this repository.  Although it is a
synthetic dataset, it is large and contains PII-shaped text.  Keeping it in the
Hugging Face cache (or an explicitly chosen external cache) prevents accidental
publication of data artifacts.

From the service directory, load and inspect it with:

```bash
python scripts/download_nemotron_pii.py
```

By default, Hugging Face uses its normal local cache (typically
`~/.cache/huggingface` on Linux/macOS or the Hugging Face cache under the user
profile on Windows).  To select a different cache outside this repository:

```bash
python scripts/download_nemotron_pii.py --cache-dir /path/to/hf-cache
```

To normalize records, choose an external destination.  The command refuses a
path inside this Git repository:

```bash
python scripts/prepare_nemotron_pii.py --output-dir /path/to/nemotron-normalized
```

The normalized files are JSONL, one record per line, containing only `text` and
character-offset entities.  They remain untracked by design.

## Presidio baseline evaluation

Install the offline evaluation extra (never needed by the gateway runtime):

```bash
pip install -e ".[dev,evaluation]"
```

The evaluator streams `test.jsonl` in batches and writes an atomic checkpoint
after every batch.  The default checkpoint is
`<data-dir>/checkpoints/presidio_test.json`; a checkpoint path inside this
repository is refused.  Checkpoints hold only run identity (dataset SHA-256,
ontology hash, detector/spaCy/Presidio configuration), progress counters, and
aggregate TP/FP/FN — never source text or entity values.

```bash
# Pilot on the first 1,000 examples; compare worker counts.
python scripts/evaluate_presidio.py --data-dir /path/to/nemotron-normalized \
  --checkpoint /path/to/pilot-w4.json --report-dir /path/to/pilot-report \
  --limit 1000 --workers 4 --restart

# Full split.  If interrupted, rerun the same command with --resume.
python scripts/evaluate_presidio.py --data-dir /path/to/nemotron-normalized --workers 8
python scripts/evaluate_presidio.py --data-dir /path/to/nemotron-normalized --workers 8 --resume
```

- An existing checkpoint is never overwritten implicitly: pass `--resume` to
  continue it or `--restart` to discard it.
- `--resume` rejects corrupted checkpoints (integrity hash, structure, count
  consistency, record-boundary offset) and incompatible ones (different dataset
  file, ontology, detector configuration, or matching strategy).
- Worker count and batch size may change between sessions; they do not affect
  metrics.  A pilot checkpoint can be resumed without `--limit` to extend it to
  the full split.

## GLiNER-PII baseline evaluation

Benchmarks the pretrained `nvidia/gliner-PII` (pinned revision in
`configs/gliner_label_map.yaml`) on the same normalized `test.jsonl` with the same
strict scoring as the Presidio baseline.  Nothing is trained or fine-tuned, and the
gateway runtime is untouched.  Use a separate environment: install a CUDA build of
`torch` from the PyTorch index, then `pip install -e ".[dev,evaluation,gliner-evaluation]"`.
Model weights stay in the Hugging Face cache.

Inference and scoring are separate steps.  `gliner_predict.py` writes an external,
append-only predictions file (example index, native label, offsets, score; never
text or values) plus a `.runtime.json` log, and resumes with `--resume`.
`evaluate_gliner.py` reads the normalized split, the cached raw parquet (native
labels, uid, document format), and the predictions in lockstep, verifies they
describe the same records, and scores them through the existing harness.

```bash
D=/path/to/nemotron-normalized
# 1. Inference-configuration pilot (device / dtype / batch size) on dev records only.
python scripts/gliner_benchmark.py --data-dir $D --output $D/gliner/benchmark.json \
  --records 64 --configs cuda:fp32:4 cuda:fp16:4 cpu:fp32:4
# 2. Threshold selection on a deterministic 2,000-record subset of TRAIN (never test).
python scripts/gliner_predict.py --data-dir $D --source dev --threshold 0.2 \
  --dtype fp16 --batch-size 4 --output $D/gliner/dev_t0.2.jsonl
python scripts/select_gliner_threshold.py --data-dir $D \
  --predictions $D/gliner/dev_t0.2.jsonl --output $D/gliner/threshold_selection.json
# 3. Test split once at the selected threshold: 1,000-example pilot, then resume.
python scripts/gliner_predict.py --data-dir $D --source test --threshold T \
  --dtype fp16 --batch-size 4 --output $D/gliner/test.jsonl --limit 1000
python scripts/gliner_predict.py --data-dir $D --source test --threshold T \
  --dtype fp16 --batch-size 4 --output $D/gliner/test.jsonl --resume
# 4. Reports (aggregate only) and the Presidio comparison.
python scripts/evaluate_gliner.py --data-dir $D --predictions $D/gliner/test.jsonl \
  --selection $D/gliner/threshold_selection.json
python scripts/compare_pii_models.py
```

GLiNER-PII was trained on the Nemotron-PII train split, so the development subset
is in-sample for the model; its scores are only used to rank thresholds.
