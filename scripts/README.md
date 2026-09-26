# Canonical script map

Run the workflow through `run_pipeline.py` unless reproducing or debugging a specific phase. Numbering gaps correspond to superseded development stages.

| Order | Script | Branch | Stage | Purpose |
|---:|---|---|---|---|
| `00` | `00_preflight.py` | global | preflight | Verify Python/dependencies; enforce pygdis==1.0.0; warn on disk space |
| `01` | `01_download_gse275562.py` | GSE275562 | acquisition | Download processed paired RNA+ATAC MuData discovery object |
| `02` | `02_phase_f0_audit.py` | GSE275562 | validation | Feasibility, modality, pairing, metadata and lineage audit |
| `03` | `03_phase_f1_batch_representation_qc.py` | GSE275562 | validation | Audit stored RNA PCA / ATAC LSI for technical batch structure |
| `04` | `04_phase_f1b_representation_rescue_qc.py` | GSE275562 | cleaning/rescue | Evaluate/freeze modality-specific representation rescue |
| `05` | `05_phase_f2_common_trajectory.py` | GSE275562 | trajectory | Construct and validate RNA-derived common trajectory with ATAC posthoc coherence |
| `06` | `06_phase_f3_gdis_rna_atac.py` | GSE275562 | analysis | Compute RNA/ATAC GDIS profiles on one common ordering |
| `07` | `07_phase_f3b_peak_topology_qc.py` | GSE275562 | validation | Peak-topology QC across window designs |
| `08` | `08_phase_f3c_transition_anchored_pairing_qc.py` | GSE275562 | analysis freeze | Prospectively choose event-specific RNA/ATAC pairing before bootstrap |
| `09` | `09_phase_f4_event_specific_cmil_bootstrap.py` | GSE275562 | inference | Paired-cell bootstrap of frozen event-specific timing |
| `10` | `10_phase_f4b_peak_stability_diagnostics.py` | GSE275562 | diagnostic | Peak-location stability diagnostic |
| `11` | `11_phase_f4c_event_region_timing_diagnostics.py` | GSE275562 | diagnostic | Event-region timing diagnostic |
| `12` | `12_phase_f5_cross_modal_alignment_null.py` | GSE275562 | null test | Cross-modal alignment null; freezes discovery interpretation |
| `13` | `13_phase_f6a_external_metadata_acquisition.py` | GSE205117 | acquisition | Download official metadata and, in full runner, the ~32.5GB RAW archive |
| `14` | `14_phase_f6b_external_dataset_audit_corrected.py` | GSE205117 | validation | Corrected canonical-field structural audit |
| `15` | `15_phase_f7_external_archive_pairing_qc.py` | GSE205117 | validation | Archive inventory and sample+barcode RNA/ATAC pairing QC |
| `16` | `16_phase_f7b_external_barcode_universe_diagnostic.py` | GSE205117 | validation fix | Correct barcode-universe definition and full target ATAC presence scan |
| `17` | `17_phase_f8_geo_native_reconstruction_qc_dynamicbins.py` | GSE205117 | cleaning/representation | Reconstruct modality-specific RNA PCA and ATAC LSI from GEO raw data using final fixed-bin implementation |
| `18` | `18_phase_f8b_external_representation_rescue_qc.py` | GSE205117 | cleaning/rescue | RNA sample-centering and ATAC depth-axis filtering QC |
| `19` | `19_phase_f8c_external_primary_arm_freeze_qc.py` | GSE205117 | analysis freeze | Freeze ordinary WT developmental arm and final matched 10D representations |
| `20` | `20_phase_f9_external_common_trajectory.py` | GSE205117 | trajectory | Attempt/validate RNA-only common developmental trajectory |
| `21` | `21_phase_f9b_external_trajectory_topology_diagnostic.py` | GSE205117 | suitability endpoint | Topology diagnostic; freezes conclusion that continuous common clock is not defensible, so no GDIS |
| `22` | `22_phase_f10_shareseq_acquisition_canonical.py` | GSE140203 | acquisition | Download six frozen SHARE-seq late-anagen files; verify frozen SHA256 and gzip |
| `24` | `24_phase_f10c_shareseq_namespace_normalized_audit.py` | GSE140203 | validation | Final barcode namespace-normalized same-cell RNA/ATAC audit; full fragment scan in runner |
| `25` | `25_phase_f11_shareseq_representations_branch_continuity.py` | GSE140203 | representation/branch QC | Construct RNA PCA and ATAC TF-IDF/LSI; evaluate candidate hair branches |
| `26` | `26_phase_f12_shareseq_irs_freeze_rna_technical_rescue.py` | GSE140203 | cleaning/freeze | Freeze TAC1→TAC2→IRS branch; remove technical RNA PC1; freeze 10D modality representations |
| `27` | `27_phase_f13_shareseq_irs_common_trajectory.py` | GSE140203 | trajectory | Validate/freeze RNA-only DPT common clock; ATAC excluded from clock construction |
| `28` | `28_phase_f14_shareseq_irs_gdis_primary_profiles.py` | GSE140203 | analysis | Compute primary/sensitivity GDIS and transition-energy profiles with frozen pyGDIS configuration |
| `29` | `29_phase_f15_shareseq_irs_peak_topology_qc.py` | GSE140203 | validation | Peak-topology QC; detects unsuitable sustained-GDIS peak pairing |
| `30` | `30_phase_f15b_shareseq_irs_sustained_plateau_transition_energy_qc.py` | GSE140203 | validation | Separate sustained profile architecture from localized transition-energy families |
| `31` | `31_phase_f16_shareseq_irs_transition_energy_pair_freeze.py` | GSE140203 | analysis freeze | Independently freeze three-window RNA/ATAC transition-energy family pair before inference |
| `32` | `32_phase_f17_shareseq_irs_frozen_pair_bootstrap.py` | GSE140203 | inference | 500 paired pseudotime-stratified bootstraps of frozen primary 400/100 timing estimator |
| `33` | `33_phase_f18_shareseq_irs_cross_modal_alignment_null.py` | GSE140203 | null test | Primary circular-shift alignment null + block-permutation sensitivity |
| `34` | `34_phase_f19_shareseq_irs_frozen_event_feature_extraction.py` | GSE140203 | mechanistic extraction | Extract descriptive RNA/ATAC effects from frozen nonoverlapping event windows |
| `35` | `35_phase_f20_shareseq_irs_mm10_peak_gene_annotation.py` | GSE140203 | annotation | mm10 nearest-TSS annotation and prevalence-aware cross-modal candidates |
| `36` | `36_phase_f21_shareseq_irs_motif_tf_regulatory_plausibility.py` | GSE140203 | mechanistic test | Matched-background JASPAR2026 motif/TF plausibility analysis; preserves valid negative result |
| `37` | `37_phase_f22_shareseq_irs_external_validation_evidence_freeze.py` | GSE140203 | evidence freeze | Consolidate F13-F21 evidence and freeze supported/non-supported claims without new inference |
| `39` | `39_phase_f23b_shareseq_irs_publication_figure_revision.py` | GSE140203 | figures | Generate final 600-dpi publication figures from frozen evidence only |
| `40` | `40_phase_f24_discovery_publication_figure.py` | GSE275562 | publication figure | Generate final discovery Figure 2 from frozen discovery analysis; no new inference |
| `41` | `41_supplementary_figure_s1_discovery_diagnostics.py` | GSE275562 | supplementary figure | Generate frozen discovery bootstrap-stability and regional-timing diagnostics |
| `42` | `42_supplementary_figure_s2_discovery_alignment_null.py` | GSE275562 | supplementary figure | Generate discovery cross-modal alignment-null figure from frozen F5 analysis |
| `43` | `43_supplementary_figure_s3_gse205117_trajectory_suitability.py` | GSE205117 | supplementary figure | Generate trajectory-suitability figure documenting the prespecified GSE205117 rejection |
| `44` | `44_supplementary_figure_s4_tf_motif_enrichment.py` | GSE140203 | supplementary figure | Generate TF-motif enrichment diagnostic from frozen F21 results |
| `45` | `45_main_figure_5_exact_circular_shift_null.py` | GSE140203 | publication figure | Render exact F18 circular-shift null values and validate frozen p-values |
| `46` | `46_supplementary_figure_s5_shareseq_sustained_architecture.py` | GSE140203 | supplementary figure | Visualize sustained-instability architecture and near-maximum envelopes without new inference |
| `47` | `47_supplementary_table_s4_shareseq_event_family_selection.py` | GSE140203 | supplementary table | Create auditable frozen event-family selection and window-sensitivity tables |
| `48` | `48_supplementary_table_s5_shareseq_branch_selection_qc.py` | GSE140203 | supplementary table | Create RNA-only branch-selection QC and chronology tables from frozen F11/F12 records |
