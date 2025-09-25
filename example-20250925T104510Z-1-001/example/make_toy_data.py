
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
rng = np.random.default_rng(42)

outdir = Path(__file__).parent

n_cells = 1500
subjects = np.repeat(["S1","S2","S3"], n_cells//3)
cell_types = rng.choice(["Tcell","Mono","Fibro"], size=n_cells, p=[0.4,0.4,0.2])
batches = rng.choice(["b1","b2"], size=n_cells)

genes = [f"G{i}" for i in range(980)] + [f"MT-{i}" for i in range(20)]
ng = len(genes)

X = rng.poisson(lam=1.0, size=(n_cells, ng)).astype(float)
mt_idx = [i for i,g in enumerate(genes) if g.startswith("MT-")]
X[subjects=="S2"][:, mt_idx] += rng.poisson(lam=1.5, size=(sum(subjects=='S2'), len(mt_idx)))

fusion_genes = ["MFN1","MFN2","OPA1","IMMT","PHB1","PHB2"]
fission_genes = ["DNM1L","FIS1","MFF","MIEF1","MIEF2"]
mitophagy_genes = ["PINK1","PRKN","SQSTM1","OPTN","BNIP3","BNIP3L","FUNDC1"]
biogenesis_genes = ["PPARGC1A","PPARGC1B","TFAM","NRF1","NRF2","PPARA","PPARD"]

for g in fusion_genes+fission_genes+mitophagy_genes+biogenesis_genes:
    if g not in genes:
        genes.append(g)
        add = rng.poisson(lam=1.0, size=(n_cells,1)).astype(float)
        X = np.hstack([X, add])

def boost(cols, mask, lam):
    X[mask][:, cols] += rng.poisson(lam=lam, size=(mask.sum(), len(cols)))

g2i = {g:i for i,g in enumerate(genes)}
boost([g2i[g] for g in fusion_genes if g in g2i], subjects=="S1", 2.0)
boost([g2i[g] for g in fission_genes if g in g2i], subjects=="S3", 1.5)
boost([g2i[g] for g in mitophagy_genes if g in g2i], subjects=="S2", 2.5)

adata = ad.AnnData(X=X)
adata.obs["subject_id"] = subjects
adata.obs["cell_type"] = cell_types
adata.obs["batch"] = batches
adata.var.index = genes
adata.write_h5ad(outdir/"scrna_toy.h5ad")

all_prots = [f"P{i}" for i in range(180)] + fusion_genes + fission_genes + mitophagy_genes + biogenesis_genes
rows = []
for s in ["S1","S2","S3"]:
    for p in all_prots:
        base = np.exp(rng.normal(loc=0.0, scale=0.5))
        if (s=="S2") and (p in mitophagy_genes):
            base *= 2.0
        rows.append((s,p,float(base)))
dfp = pd.DataFrame(rows, columns=["subject_id","protein","abundance"])
dfp.to_csv(outdir/"ev_proteomics_toy.csv", index=False)

img = pd.DataFrame({
    "subject_id": ["S1","S2","S3"],
    "mito_count": [1000, 1200, 900],
    "mean_length": [1.5, 1.2, 1.8],
    "mean_branching": [2.2, 1.9, 2.0],
    "fragmentation_index": [0.4, 0.6, 0.5],
})
img.to_csv(outdir/"imaging_toy.csv", index=False)

print("Wrote toy data to", outdir)
