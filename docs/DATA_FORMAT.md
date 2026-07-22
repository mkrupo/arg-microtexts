# Canonical EDU Data Format

## Text policy

The raw `.txt` file is authoritative. XML segment text is located sequentially
inside that raw string. Whitespace between adjacent source units belongs to no
unit; the current corpus uses a single ASCII space there. EDU-per-line exports
strip only leading and trailing whitespace from each source unit. Joining the
exported lines with one ASCII space must recreate the raw document exactly.

No Unicode normalization, punctuation normalization, or internal whitespace
rewriting is permitted.

One known upstream mismatch is recorded explicitly: the refined argument XML
for `micro_d14` omits the final period of its last unit, while the raw text and
RST segment contain it. The RST/raw version is exported.

## EDU-per-line files

Files under `derived/edu/` contain one non-empty EDU per line in document
order. Their basenames match the source document IDs, for example:

```text
derived/edu/en/multilayer_gold/micro_b001.edus
```

Every collection is accompanied by a manifest with source commits, extraction
policy, counts, and SHA-256 hashes.

## Boundary inventory

Canonical TSV inventories use these columns:

| Column | Meaning |
| --- | --- |
| `doc_id` | Stable document ID such as `micro_b001` |
| `language` | ISO 639-1 code (`de` or `en`) |
| `char_offset` | Zero-based boundary offset in the raw Python Unicode string |
| `token_index` | Optional zero-based index in a documented reference token stream |
| `adu_id` | Original aligned ADU identifier |
| `edu_id` | EDU identifier within the source layer |
| `boundary_class` | `document_start`, `adu`, or `internal_edu` |
| `source` | Corpus layer or model that supplied the boundary |
| `status` | `gold`, `automatic`, `forced`, `adjudicated`, or `ambiguous` |
| `confidence` | Optional model probability or calibrated score |
| `model` | Optional exact model identifier |
| `run_id` | Optional experiment identifier |
| `sameunit_affected` | Whether the document has an alternative no-Same-Unit RST tree |

Document starts are recorded but excluded from segmentation metrics. For the
English multilayer gold, the first EDU has class `document_start`; the first
EDU of each later ADU has class `adu`; and additional EDUs inside an ADU have
class `internal_edu`.

## Provenance manifests

A committed derived collection manifest records:

- source repository and commit;
- tool and repository commit when available;
- document and unit counts;
- normalization and reconstruction policy;
- source and output hashes;
- special cases such as Same-Unit alternatives; and
- model configuration for automatic layers.
