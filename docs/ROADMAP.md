# EDU Refinement Roadmap

This roadmap separates trusted source preparation, automatic proposals, human
annotation, and final release. A milestone is complete only after its listed
validation gate passes.

## M1: Repository and source foundation

**Status: complete.**

- Pin `arg-microtexts-multilayer` as a public submodule.
- Establish the tracked/ignored artifact policy.
- Document methodology, formats, experiments, and annotation plans.
- Audit all original and multilayer representations.

**Gate:** a recursive clone passes the source audit without local paths or
untracked dependencies.

## M2: Canonical English gold

**Status: complete.**

- Extract the English EDU segmentation from the main multilayer RST trees.
- Cross-check the RST sequence against the refined argument layer to recover
  original ADU membership.
- Export one EDU per line plus a character-offset boundary inventory.
- Flag the nine documents with alternative no-Same-Unit RST trees.

**Gate:** all 112 files reconstruct the raw English texts and total 680 EDUs;
all 576 original ADU starts remain boundaries.

## M3: Canonical German ADU reference

**Status: complete.**

- Extract German raw-text character spans for all 576 ADUs.
- Publish a canonical character-offset boundary inventory for model comparison.
- Preserve document, ADU, and bilingual correspondence identifiers.

**Gate:** all spans are exhaustive, ordered, non-overlapping, reconstruct the
raw text, and align one-to-one across the German and English ADU layers.

## M4: `eduseg_de` experiments

**Status: in progress.**

- Run the released German-only model on complete documents.
- Retain raw boundary probabilities and full provenance.
- Produce raw and ADU-constrained outputs.
- Run per-ADU inference only as a context-sensitivity ablation.

**Gate:** every output reconstructs its input, no document is truncated, and
the run can be reproduced from a committed configuration and model revision.

## M5: SeCoRel comparison

- Run SeCoRel on the same canonical token stream.
- Preserve raw and ADU-constrained outputs.
- Compare ADU recall, added boundaries, model agreement, and context effects.

**Gate:** both systems are compared at identical raw-text boundary positions,
not through model-specific token indices.

## M6: Human German gold

- Train annotators on a versioned guideline and expert practice set.
- Annotate complete documents independently with ADU starts locked.
- Measure agreement, revise the guideline if necessary, and adjudicate.
- Preserve individual and adjudicated layers.

**Gate:** all disagreements are resolved or explicitly marked ambiguous, and
the adjudicated layer reconstructs every source text.

## M7: Evaluation and bilingual alignment

- Evaluate both raw and constrained systems against German gold.
- Report all-boundary and internal-only scores with document-level uncertainty.
- Align German and English EDUs within their already aligned ADUs.
- Record one-to-one, one-to-many, many-to-one, and unaligned cases.

**Gate:** reported tables are generated from committed configurations and
released boundary inventories.

## Release targets

- `v0.2.0`: audited sources, canonical English gold, and German automatic
  proposals.
- `v0.3.0`: annotation guideline, training material, and pilot agreement.
- `v1.0.0`: adjudicated German gold, final evaluation, and bilingual EDU
  alignment.
