# Scientific branch structure

## 1. GSE275562 — discovery branch (required)

Scripts **01–12** reproduce the original discovery analysis. Several stages are intentionally diagnostic and print results rather than writing derived files; the master runner preserves their complete stdout/stderr under `logs/`.

The branch ends with the frozen discovery interpretation: the common trajectory is robust, but the simple universal ATAC-before-RNA priming model is not supported by the event-region timing and cross-modal alignment analyses.

Scripts **40–42** are publication-only post-processing stages generated from the frozen discovery evidence.

## 2. GSE205117 — trajectory-suitability / negative-control branch (valid, independent)

Scripts **13–21** are retained because they are scientifically valid and document why this paired multiome dataset is **not suitable for downstream GDIS timing validation**.

The corrected chain is:

`F6a acquisition -> corrected F6b audit -> F7/F7b pairing validation -> final GEO-native F8 reconstruction -> F8b technical rescue -> F8c primary arm freeze -> F9 trajectory test -> F9b topology diagnostic`

F9/F9b show that the required common continuous developmental clock is not defensible. The branch therefore terminates **before GDIS by design**. This is a valid negative/suitability result, not a pipeline failure.

Script **43** generates the publication-only trajectory-suitability summary figure from this frozen endpoint.

## 3. GSE140203 SHARE-seq — positive external validation (required)

The canonical branch begins with the new acquisition-only F10 script (`22_phase_f10_shareseq_acquisition_canonical.py`) and then jumps intentionally to **F10c / script 24**. Scripts 22-original and 23 were development-stage audits with parsing/namespace assumptions that were superseded.

Scripts **24–37** reproduce pairing, modality-specific representations, the RNA-only common clock, GDIS/transition-energy topology, prospectively frozen event pairing, bootstrap inference, alignment null, feature extraction, mm10 annotation, motif analysis, and final evidence freeze.

Script **39** generates the F23b publication figure set. Scripts **44–48** add later manuscript-ready publication figures and supplementary tables from frozen F11–F21 evidence. Script 38 remains intentionally excluded because F23b superseded it without changing any scientific result.
