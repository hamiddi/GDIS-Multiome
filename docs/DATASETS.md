# Datasets

## GSE275562 — pancreatic endocrine development

- Technology: 10x Genomics Multiome RNA + ATAC
- System: mouse embryonic pancreatic endocrinogenesis
- Processed paired cells: 22,604
- Primary analyzed endocrine lineage: 8,868 cells
- States: Ngn3 low → Ngn3 high → Fev+ → Fev+ Beta → Beta
- Role: discovery

## GSE205117 — early organogenesis

- Technology: 10x Genomics Multiome RNA + ATAC
- System: mouse early organogenesis
- Samples: 11 spanning E7.5–E8.75, including 10 WT/control and one Brachyury/T knockout sample
- Primary WT analysis arm: 5,344 paired cells
- Stage-restricted trajectory subset: 4,859 paired cells
- Candidate states: NMP → Paraxial mesoderm → Somitic mesoderm
- Role: trajectory-suitability / negative-control branch

The intended continuous three-state biological topology does not satisfy the prespecified suitability gate, so downstream GDIS timing is not performed.

## GSE140203 — SHARE-seq mouse skin

- Technology: SHARE-seq paired RNA + chromatin accessibility
- System: late-anagen mouse skin / hair follicle
- Paired annotation universe: 34,774 cells
- Final validation trajectory: 5,050 cells
- States: TAC-1 → TAC-2 → IRS
- Role: independent external validation

## Data policy

Large public datasets are not committed. Canonical scripts download required public resources and verify integrity where frozen hashes are available. Generated data, large matrices, downloaded references, and logs are ignored by Git by default.
