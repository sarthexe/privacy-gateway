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
