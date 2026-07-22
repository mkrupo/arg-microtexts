# Methodological Decisions

## 2026-07-22: Multilayer source as a submodule

The public upstream `arg-microtexts-multilayer` repository is pinned under
`external/` rather than copied into this fork. This preserves provenance and
allows deliberate upstream updates.

## 2026-07-22: Character offsets are canonical

Raw-text Unicode character offsets are the primary boundary coordinates.
Token indices are optional secondary metadata tied to a named tokenizer.

## 2026-07-22: ADU-preserving refinement

Original ADU starts are locked EDU starts. Automatic complete-document
predictions are stored raw and as a separate union with the ADU starts.

## 2026-07-22: English and German annotation status

The main 680-unit English RST segmentation is imported as gold. The refined
argument layer supplies ADU membership; SDRT is outside this workflow. German
automatic outputs are proposals until independent human annotation and
adjudication are complete.

## 2026-07-22: Same-Unit handling

The main multilayer segmentation is the primary English reference. The nine
documents with alternative no-Same-Unit RST trees are flagged for sensitivity
analysis rather than treated as a second complete corpus version.
