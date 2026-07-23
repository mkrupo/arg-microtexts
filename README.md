# Argumentative Microtexts with EDU Refinement

This repository is a research fork of the original **Argumentative Microtext
Corpus**. It preserves the original German texts, their professional English
translations, and their argumentation structures, while adding a reproducible
workflow for finer Elementary Discourse Unit (EDU) segmentation.

The project has two complementary goals:

1. expose the existing English gold EDU segmentation from the multilayer
   corpus in a simple, verified format; and
2. create high-quality German EDU segmentation through documented automatic
   experiments followed by independent human annotation and adjudication.

Automatic German segmentations are proposals, not gold annotations. Existing
Argumentative Discourse Unit (ADU) boundaries are treated as locked EDU
boundaries; models may add finer boundaries inside ADUs.

## Data sources

- `corpus/de/` and `corpus/en/` contain the original corpus release.
- `external/arg-microtexts-multilayer/` is a pinned Git submodule containing
  the finer English RST EDU segmentation and its refined argumentation layer.
- `derived/` contains verified, reproducible exports and later automatic or
  adjudicated EDU layers. Every derived dataset has provenance metadata.

Clone with the multilayer source initialized:

```bash
git clone --recurse-submodules git@github.com:mkrupo/arg-microtexts.git
```

For an existing clone:

```bash
git submodule update --init --recursive
```

## Current status

The original corpus contains 112 German microtexts and aligned English
translations with 576 coarse ADUs. The English multilayer RST resource refines
these into 680 EDUs: 83 ADUs are split, introducing 104 internal EDU
boundaries. Nine RST documents additionally have alternative trees without
Same-Unit segmentation.

The first German complete-document `eduseg_de` experiment is also published as
an explicitly automatic proposal layer. Its raw output contains 691 EDUs and
recovers 461 of 464 non-initial ADU starts; the ADU-constrained copy contains
694 EDUs after restoring the three locked starts. The model adds 118 candidate
within-ADU boundaries. These counts are diagnostics, not German gold results.

The implementation roadmap and validation gates are documented in
[`docs/ROADMAP.md`](docs/ROADMAP.md). The canonical boundary representation is
defined in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

## Reproducible checks

The core audit and English export use only the Python standard library:

```bash
python3 tools/audit_sources.py
python3 tools/export_english_edus.py --check
python3 tools/export_german_adus.py --check
python3 tools/publish_eduseg_run.py --check
python3 tools/publish_eduseg_adu_ablation.py --check
python3 tools/publish_secorel_run.py --check
python3 tools/build_edu_explorer_data.py --check
python3 -m unittest discover -s tests
```

The committed source audit is `derived/source_audit.manifest.json`. English
gold files are under `derived/edu/en/multilayer_gold/`, with canonical starts
in `derived/boundaries/en_multilayer_gold.tsv`. The locked German ADU reference
is under `derived/adu/de/original_gold/`; its bilingual ADU mapping is
`derived/alignments/adu_de_en.tsv`.

The German automatic layer is under
`derived/edu/de/automatic/eduseg_de_document_v1/`. Its result manifest and
review tables are under `experiments/results/eduseg_de_document_v1/`.
The secondary per-ADU context ablation uses the parallel
`eduseg_de_adu_context_v1` directories.
The SeCoRel comparison uses `secorel_disrpt_sentence_chunks_v1` and preserves
its separate tokenization/chunking provenance.

Model inference is deliberately separate from the source audit. It will use
optional environments for `eduseg_de` and SeCoRel; model weights, caches, raw
runs, and machine-local paths are not committed.

## Local interactive result explorer

The repository includes a locally served website for reading the aggregate
findings and inspecting exact model boundaries inside every source text. It
combines the German ADUs, all three published German prediction conditions,
their raw scores at displayed positions, and aligned English RST gold EDUs.
The browser dataset is a deterministic derived artifact:

```bash
python3 tools/build_edu_explorer_data.py
python3 tools/build_edu_explorer_data.py --check
```

For local website development:

```bash
nvm install
nvm use
npm install
npm run dev
```

Then open the local URL printed by vinext (normally
`http://localhost:3000`). Node.js 22 or newer is required; `.nvmrc` records
that version for `nvm`.

The explorer is not hosted by this project. Anyone who clones the repository
can build or serve it locally from the committed source and browser dataset.

The interface labels automatic German boundaries as proposals throughout.
It does not turn system agreement or English translation alignment into German
gold evidence. See [`docs/EXPLORER.md`](docs/EXPLORER.md) for the visual
encoding, review filters, provenance, and interpretation limits.

The primary German complete-document EduSeg experiment is frozen in
`experiments/configs/eduseg_de_document_v1.toml`. After installing PyTorch and
Transformers in an external environment and obtaining the pinned model files,
run it with:

```bash
python tools/run_eduseg_de.py --model-dir /path/to/eduseg_de/model
```

The runner verifies every required model file against its committed SHA-256,
refuses truncation, and writes raw predictions, ADU-constrained proposals,
character-level boundary scores, diagnostics, and a portable provenance
manifest under the ignored `work/runs/` directory. See
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) before promoting outputs into a
versioned derived layer.

## Repository policy

- Original files under `corpus/` are preserved unless an upstream correction
  is intentionally incorporated and documented.
- The multilayer source is never edited in place; updates occur by changing
  the pinned submodule commit.
- Curated code, configurations, summaries, manifests, gold data, and released
  automatic proposals are versioned.
- Virtual environments, model weights, logs, caches, and scratch outputs are
  ignored.
- A generated EDU file is committed only after its text reconstructs the
  corresponding raw document under the documented whitespace policy.

## Original corpus

The corpus contains 112 short argumentative texts. All texts were originally
written in German and professionally translated into English. Texts `b001`–
`b064` and `k001`–`k031` were collected in a controlled text-generation
experiment from 23 participants. Texts `d01`–`d23` were written by Andreas
Peldszus for teaching and analysis.

The original argumentation annotation follows Peldszus and Stede (2013). The
corpus is released under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International license; see
[`LICENSE`](LICENSE).

When using the original corpus, cite:

> Andreas Peldszus and Manfred Stede. 2015. *An Annotated Corpus of
> Argumentative Microtexts*. First European Conference on Argumentation.

When using the English multilayer EDU annotation, also cite:

> Manfred Stede, Stergos Afantenos, Andreas Peldszus, Nicholas Asher, and
> Jérémy Perret. 2016. *Parallel Discourse Annotations on a Corpus of Short
> Texts*. LREC 2016.

Automatic German segmentation experiments using `eduseg_de` should also cite:

> Steffen Frenzel, Maximilian Krupop, and Manfred Stede. 2026. *Discourse
> Segmentation of German Text with Pretrained Language Models*. JLCL 39(1).
