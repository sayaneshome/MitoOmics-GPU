# Fallback small defaults (used only if MitoCarta pathways are empty)
FUSION_GENES = ["MFN1","MFN2","OPA1","IMMT","PHB1","PHB2"]
FISSION_GENES = ["DNM1L","FIS1","MFF","MIEF1","MIEF2"]
MITOPHAGY_GENES = ["PINK1","PRKN","SQSTM1","OPTN","BNIP3","BNIP3L","FUNDC1"]
BIOGENESIS_GENES = ["PPARGC1A","PPARGC1B","TFAM","NRF1","NRF2","PPARA","PPARD"]

ALL_PATHWAYS = {
    "fusion": FUSION_GENES,
    "fission": FISSION_GENES,
    "mitophagy": MITOPHAGY_GENES,
    "biogenesis": BIOGENESIS_GENES,
}
