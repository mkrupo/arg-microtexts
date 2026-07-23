# EDU Explorer

The repository website is a reader for the verified corpus exports and the
three published German segmentation conditions. It is designed for qualitative
error analysis before German fine-grained gold EDU annotation exists.

## What the interface shows

The overview reports only quantities that can be established from committed
artifacts:

- the audited bilingual corpus totals;
- the number of within-ADU proposals made by each condition;
- exact-character-offset agreement between the two EduSeg contexts;
- exact-character-offset agreement between document-context EduSeg and
  SeCoRel; and
- three documented cases illustrating plausible refinement,
  likely oversegmentation, and a missed locked boundary.

The document explorer preserves the complete original German text. Gold ADU
starts delimit visible blocks. Inside each block, colored markers show automatic
within-ADU proposals:

| Marker | Layer | Input condition |
| --- | --- | --- |
| `D` | EduSeg document | Complete German microtext |
| `A` | EduSeg per ADU | One locked ADU at a time |
| `S` | SeCoRel | DISRPT-like sentence chunks |

Selecting a marker displays the exact raw-text offset, local context, and the
published softmax score for each condition. A score is model evidence, not a
calibrated probability that the boundary is correct.

The optional English column shows aligned ADUs with the existing English RST
gold EDU boundaries. It is a reference layer for later alignment analysis. The
viewer does not project those positions onto German or call them German gold.

## Filters and review use

Documents can be searched in either language and filtered for:

- any disagreement among the three German conditions;
- at least one boundary shared by all three;
- an ADU start missed by document-context EduSeg or SeCoRel;
- an English ADU refined into multiple RST EDUs; or
- membership in the nine-document Same-Unit alternative set.

“Review priority” is a navigation heuristic based on disagreements, missed
locked starts, and shared proposals. It is not a quality score.

## Reproducible data build

`tools/build_edu_explorer_data.py` reads the canonical raw texts, audited
English segmentation, published boundary inventories, score tables, summaries,
and run manifests. It emits:

```text
public/data/edu-explorer.json
```

The script uses the standard library and asserts the three published proposal
totals. The output is deterministic:

```bash
python3 tools/build_edu_explorer_data.py
python3 tools/build_edu_explorer_data.py --check
python3 -m unittest tests.test_edu_explorer_data
```

No model weights, inference environment, runtime database, or machine-local
paths are needed by the website.

## Running the local website

The interface is implemented with React and vinext. Node.js 22 or newer is
required:

```bash
nvm install
nvm use
npm install
npm run data:check
npm run check
npm run build
npm run dev
```

For ordinary exploration, only `nvm use`, `npm install`, and `npm run dev` are
needed. Open the URL printed by the development server, normally
`http://localhost:3000`.

Generated build directories and dependency installations are ignored. Website
source, the package lock, and the deterministic browser dataset are versioned.
The project does not publish or require a hosted deployment.

## Interpretation boundary

The 118 EduSeg document additions and 133 SeCoRel additions are not false
positives merely because the original German annotation has no boundary there:
ADUs are coarser units. Conversely, a position proposed by all systems is not
automatically a correct EDU boundary. The coordination split in
`micro_b001/a5` demonstrates why model agreement is useful for review ordering
but insufficient for annotation. Human double annotation and adjudication
remain the path to German gold EDUs.
