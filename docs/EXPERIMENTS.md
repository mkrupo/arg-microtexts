# Experiment Policy

## Systems

The primary German system is the released German-only `eduseg_de` model. The
multilingual SeCoRel segmenter is a secondary comparison. The bilingual
EduSeg model may be added later as a documented ablation; it is not silently
substituted for the primary model.

## Required conditions

Each model produces a raw complete-document prediction. Gold ADU starts are
then unioned with that raw set to create a separate constrained proposal.
EduSeg is additionally run independently per ADU as a context-sensitivity
ablation.

## Primary EduSeg run

The frozen complete-document configuration is
`experiments/configs/eduseg_de_document_v1.toml`. The runner takes the model
directory explicitly so weights and machine-local paths stay outside the
repository:

```bash
python tools/run_eduseg_de.py \
  --model-dir /path/to/eduseg_de/model \
  --output-dir work/runs/eduseg_de_document_v1
```

The model directory must contain the exact six files hash-pinned in the
configuration. The runner loads them offline, checks every document against
the 512-subword limit without truncation, and maps model-token starts back to
canonical raw-text character offsets. A predicted `B` label inside a subword
is rejected as an invalid textual boundary and counted in the summary.

Outputs include raw and ADU-constrained EDU-per-line files, parallel boundary
inventories, raw softmax probabilities for every eligible text-start token,
pre-gold diagnostics, and a manifest with hashes of every artifact. Softmax
scores are uncalibrated model confidence, not empirical correctness
probabilities. When multiple SentencePiece tokens share a character start,
boundary metadata uses the highest boundary probability at that position;
the score table retains every model token and its index. The run directory is
ignored until the result has been checked and deliberately promoted as a
released automatic layer.

After reviewing a completed run, publish the versioned automatic layer with:

```bash
python tools/publish_eduseg_run.py
```

This verifies all run hashes and text reconstructions before copying the raw
and constrained EDU files into `derived/edu/de/automatic/`, the boundary and
score inventories into their canonical `derived/` directories, and compact
review tables plus provenance into `experiments/results/`. A recursive clone
can validate the published layer without model weights or the ignored run:

```bash
python tools/publish_eduseg_run.py --check
```

## Primary EduSeg result

The published `eduseg_de_document_v1` run processed all 112 complete German
documents with no truncation (maximum 190 of 512 model tokens) and no rejected
subword boundary predictions. It produced:

- 691 raw EDUs;
- 461/464 recovered non-initial gold ADU starts (99.35% recall);
- 118 candidate boundaries inside ADUs; and
- 694 ADU-constrained EDUs after restoring the three missed locked starts.

ADU recall is a conservative diagnostic of boundary detection, not overall
EDU accuracy. Likewise, the 118 additions cannot be scored as false positives
until German fine-grained gold annotation exists. Of these additions, 117 are
sentence-internal and one follows terminal punctuation. The latter separates
the two questions inside `micro_b014/a4` and is a plausible refinement of a
coarse ADU.

The project discussion example in `micro_b001/a5` illustrates the opposite
risk: the model assigns 0.8091 boundary probability before *und Vorreiter im
Mülltrennen werden!*, although the shared subject and modal construction make
one EDU the current expert working analysis. This is a likely oversegmentation
candidate, not an adjudicated gold error. The full review table preserves both
segments, the containing ADU, offset, and score so such hypotheses can be
checked systematically without altering the automatic layer.

## Per-ADU context ablation

`experiments/configs/eduseg_de_adu_context_v1.toml` freezes the secondary
condition in which each of the 576 gold ADUs is presented independently to the
same model. Run it with:

```bash
python tools/run_eduseg_de_adu_ablation.py \
  --model-dir /path/to/eduseg_de/model
```

This is not an alternative primary segmentation. Every ADU start is an
artificial sequence start and is therefore inserted structurally; only
strictly within-ADU predictions are compared with the complete-document run.
The comparison reports exact character-offset overlap, context-only
boundaries, and agreement without interpreting either condition as gold.

## Reproducibility record

Every run manifest contains:

- input manifest hash;
- repository and inference-code commits;
- model path or public identifier and immutable revision/hash;
- tokenizer revision;
- label mapping and decoding policy;
- full-document/chunking configuration;
- device and relevant library versions;
- raw boundary scores;
- start and completion timestamps; and
- hashes of exported predictions.

Model weights, caches, virtual environments, and raw runtime logs are not
committed. Small curated prediction layers, configurations, manifests, and
summary tables are committed when they form part of a documented experiment.

## Pre-gold reporting

Before adjudicated German EDUs exist, report:

- recall at known ADU boundaries;
- missed ADU boundaries;
- added internal boundaries;
- sentence-final versus sentence-internal proposals;
- complete-document versus per-ADU agreement;
- EduSeg versus SeCoRel agreement; and
- confidence and linguistic error categories.

Do not report model precision using ADUs as negative evidence.
