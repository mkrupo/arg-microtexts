# German EDU Annotation Plan

German gold annotation is planned after the automatic proposal experiments.
The model outputs will help identify phenomena for training material, but they
will remain hidden during annotation.

## Annotation setup

- Annotators work on complete German documents.
- Existing ADU boundaries are visible and locked.
- Annotators may add only within-ADU EDU boundaries.
- Model outputs and English EDU boundaries are not displayed.
- Every token must belong to exactly one EDU; EDUs may not overlap.
- The guideline version and annotation-tool version are recorded.

## Training sequence

1. Introduce the EDU definition and its relation to the locked ADUs.
2. Work through expert examples of coordination, subordinate and relative
   clauses, parentheticals, rhetorical PPs, ellipsis, fragments, and embedded
   units.
3. Annotate an expert practice set and discuss every disagreement.
4. Double-annotate a pilot batch and calculate boundary agreement.
5. Revise the guideline before full production if systematic ambiguity
   remains.

## Production and adjudication

Two assistants independently annotate all documents. Individual layers are
preserved. An adjudicator reviews every boundary disagreement using the
versioned guideline and may mark genuinely unresolved cases as ambiguous.

Agreement is reported as boundary precision/recall/F1 between annotators,
supplemented by document-level results and a qualitative disagreement
inventory. A prevalence-sensitive statistic may be reported additionally but
does not replace direct boundary agreement.
