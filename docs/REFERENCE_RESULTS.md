# Reference results for replication auditing

These are **reference checkpoints**, not optimization targets.

## GSE275562 discovery

- Analyzed endocrine trajectory: **8,868 paired cells**
- RNA-derived clock: robust across prespecified neighborhood settings
- Prospectively defined comparison: approximately ATAC 0.228 → RNA 0.288
- Paired-event recovery: **73.8%**
- Bootstrap median separation: **0.0509**
- 95% bootstrap CI: **−0.0108 to 0.1372**
- Regional median separation: approximately **−0.0039**
- Interpretation: no robust universal ATAC-before-RNA lead; no significant broad alignment under the primary discovery null

## GSE205117 suitability control

- Primary WT arm: **5,344 paired cells**
- Stage-restricted trajectory subset: **4,859 paired cells**
- Primary medians: NMP **0.166**, Somitic **0.508**, Paraxial **0.770**
- Observed ordering: NMP < Somitic < Paraxial
- Expected linear progression: NMP < Paraxial < Somitic
- Paraxial RNA-neighbor fractions: NMP ~**0.007**, self ~**0.959**, Somitic ~**0.035**
- Interpretation: trajectory-suitability criteria fail; no downstream GDIS timing

## GSE140203 SHARE-seq

### RNA-only clock
- TAC-1 → TAC-2 → IRS: **5,050 paired cells**
- TAC-1: **3,370**
- TAC-2: **1,008**
- IRS: **672**
- Median pseudotimes: TAC-1 **0.1483**, TAC-2 **0.1994**, IRS **0.4772**
- Cross-neighborhood Spearman correlations: **>0.997**

### Sustained architecture
- Median 95% near-maximum width: RNA **0.491**, ATAC **0.027**
- Approximate width ratio: **18-fold**
- Median global maximum: RNA **0.246**, ATAC **0.084**

### Localized timing
- Frozen family centers: RNA **0.1508**, ATAC **0.1208**
- Primary events: RNA **0.1541**, ATAC **0.1235**
- Observed CMIL: **0.0306**
- Window sensitivity CMIL: **0.0285**, **0.0306**, **0.0409**

### Paired bootstrap
- Replicates: **500**
- Paired recovery: **422/500 = 84.4%**
- Median CMIL: **0.0371**
- 95% CI: **0.0111–0.0528**
- `P_boot(CMIL > 0) = 0.9953`

### Broad alignment
- Absolute Q50 separation: **0.0051**
- Spearman correlation: ~**0.6613**
- Jensen–Shannon divergence: ~**0.0438**
- Exact circular-shift p-values: Q50 **0.2195**, Spearman **0.0244**, JS **0.1463**
- Interpretation: only one of three prespecified metrics is significant; overall primary evidence remains limited/no evidence

### Mechanistic follow-up
- ATAC-opening candidate peaks: **310**
- Later RNA-increase genes: **301**
- Proximity-supported genes: **11**
- Eligible TF motifs: **326**
- Motifs satisfying all prespecified criteria: **0**

The mechanistic analysis does not establish direct peak-to-gene causality or universal chromatin priming.
