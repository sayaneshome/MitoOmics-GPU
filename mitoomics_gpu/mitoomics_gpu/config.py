
from dataclasses import dataclass

@dataclass
class Weights:
    copy_number: float = 0.35
    fusion_fission: float = 0.25
    mitophagy: float = 0.25
    heterogeneity: float = 0.15

DEFAULT_WEIGHTS = Weights()
BOOTSTRAP_N = 200
RANDOM_SEED = 1337
