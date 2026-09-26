# Excluded / superseded development scripts

These files were present in the uploaded development archive but are intentionally **not** part of the canonical pipeline.

| Script | Reason |
|---|---|
| `13_phase_f6_external_dataset_audit.py` | Preliminary all-in-one F6 script; superseded by explicit F6a acquisition + corrected F6b audit. |
| `14_phase_f6b_external_dataset_audit.py` | Incorrect heuristic metadata-field selection; it could interpret `sample` as stage and `genotype` as cell type. |
| `17_phase_f8_external_representation_acquisition_qc.py` | Depended on author-hosted FTP representations; outbound FTP was unavailable on the target HPC. |
| `17_phase_f8_external_representation_acquisition_qc_v2.py` | Byte-identical duplicate of the same inaccessible-FTP script. |
| `17_phase_f8_geo_native_reconstruction_qc.py` | Initial GEO-native reconstruction; superseded by later contig/bin fixes. |
| `17_phase_f8_geo_native_reconstruction_qc_contigfix.py` | Intermediate contig-fix implementation; superseded by the final dynamic-bin/fixed-bin implementation. |
| `22_phase_f10_shareseq_acquisition_audit.py` | Useful download code, but the structural audit made incorrect raw-format assumptions and generated a false review. Replaced with acquisition-only canonical F10. |
| `23_phase_f10b_shareseq_format_corrected_audit.py` | Intermediate parsing correction; barcode namespaces were not yet normalized correctly. |
| `38_phase_f23_shareseq_irs_publication_figures.py` | Superseded by F23b/script 39, which uses the actual primary 400/100 frozen event positions and improved publication labeling. |

The accidental filename `8_phase_f8b_external_representation_rescue_qc.py` **is valid code**. In this package it has been normalized to `18_phase_f8b_external_representation_rescue_qc.py` so the canonical order is unambiguous.


## Additional publication-script version consolidation

The following source variants from `additional.zip` were superseded. For families whose old unversioned source name is also the clean canonical name, the final package contains that filename with the newest selected content.

| Source variant | Reason |
|---|---|
| `40_phase_f24_discovery_publication_figure.py` | Superseded source content; replaced by v3 content under this clean canonical filename. |
| `40_phase_f24_discovery_publication_figure_v2.py` | Superseded by v3. |
| `41_supplementary_figure_s1_discovery_diagnostics.py` | Superseded source content; replaced by v4 content under this clean canonical filename. |
| `41_supplementary_figure_s1_discovery_diagnostics_v2.py` | Superseded by v4. |
| `41_supplementary_figure_s1_discovery_diagnostics_v3.py` | Superseded by v4. |
| `42_supplementary_figure_s2_discovery_alignment_null.py` | Superseded source content; replaced by v2 content under this clean canonical filename. |
