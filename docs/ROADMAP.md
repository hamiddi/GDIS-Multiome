# Roadmap

Exact manuscript replication remains separate from future extensions.

## Near-term repository improvements

- Add the final paper DOI, journal citation, and release metadata.
- Create a tagged archival release linked to a permanent software archive.
- Add lightweight unit tests for helper functions that do not require full datasets.
- Add machine-readable snapshots/hashes for key reference outputs.
- Add a simpler command-line interface around validated workflow components.
- Evaluate container recipes after native Conda replication is verified across systems.

## Scientific extensions

### Broader benchmarking
Apply GDIS-Multiome to additional paired datasets and classify transitions as ATAC-leading, RNA-leading, approximately synchronous, multimodal, weakly coupled, or modality-specific.

### Branch-aware trajectories
Extend suitability and instability analysis to lineage bifurcations while preserving branch-specific validation.

### Additional paired modalities
Potential targets include RNA–protein, RNA–methylation, accessibility–histone modification, spatial multimodal measurements, and perturbation-linked single-cell data.

### Stronger downstream regulatory integration
After timing is frozen, integrate enhancer–gene linking, co-accessibility networks, chromatin-conformation data, TF occupancy, and perturbational evidence.

### Explicit chronological time
Apply the framework to real-time or perturbation-time datasets to compare pseudotime-derived ordering with direct temporal measurements.

## Long-term software direction

A reusable GDIS-Multiome interface could expose validated components for trajectory suitability, matched windows, modality-specific GDIS, event-family freezing, CMIL estimation, paired bootstrap inference, dependence-preserving nulls, and standardized reporting.

A future reusable API should preserve a strict `reproduce-paper` mode in which manuscript parameters remain immutable.
