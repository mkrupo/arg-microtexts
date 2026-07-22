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
