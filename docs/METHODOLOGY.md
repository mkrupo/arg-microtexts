# Methodology

## Refinement assumption

Let `A` be the set of original ADU starts, `E` the intended EDU starts, and
`M` a model's predicted starts. The multilayer annotation methodology assumes
that every ADU boundary is also an EDU boundary, so:

```text
A ⊆ E
```

This determines how automatic predictions are interpreted:

- `A - M` are conservative model misses.
- `M - A` are candidate internal EDU starts, not automatic errors.
- `A ∪ M` is an ADU-preserving automatic proposal, not a gold segmentation.

Raw predictions and constrained proposals are always stored separately. The
constrained proposal must never be reported as raw model performance at ADU
boundaries because those boundaries are inserted by construction.

## Language-specific status

The English multilayer corpus already supplies expert gold RST EDUs. This
project imports and validates the main RST segmentation; it does not recreate
it automatically. The refined argument layer is consulted only to associate
each RST EDU with its original ADU.

German currently has gold ADU starts but no gold internal EDU starts. EduSeg
and SeCoRel outputs therefore remain model proposals until independent human
annotation and adjudication are complete.

## Primary inference context

Complete-document inference is primary because it retains syntactic and
discourse context. Independent per-ADU inference is an ablation measuring the
effect of context loss and artificial sequence resets. It is not the default
way to construct the German proposal layer.

## Canonical positions

Raw Unicode character offsets are the primary boundary coordinates. Model
token indices are secondary metadata. This allows systems with different
tokenizers to be compared at the same positions and makes text reconstruction
independent of any model environment.

## Evaluation before German gold

Before human gold exists, valid diagnostics include ADU-boundary recall,
internal proposal counts, model agreement, context sensitivity, confidence,
and linguistic analysis. Precision against the ADU layer is not meaningful,
because correct finer EDU boundaries would be counted as false positives.

## Human annotation

Annotators will see complete German documents with ADU boundaries locked, but
will not see model predictions or English EDU boundaries. Two independent
layers will be retained before adjudication. If the annotations are later used
for model training, a held-out evaluation subset must be frozen before any
fine-tuning.
