from dataclasses import dataclass
from typing import List, Dict, Set, Optional

@dataclass
class Element:
    kind: str          # 'R', 'C', 'L', 'V', 'I', 'E', 'O', 'T'
    name: str          # 'R1'
    n1: str
    n2: str
    value: str         # keep as string for symbolic (e.g. 'R', '10k', '1e-6')
    extra: Optional[str] = None  # for sources (DC/SIN/PULSE...), optional

@dataclass
class Circuit:
    elements: List[Element]
    nodes: Set[str]

    def node_set(self) -> Set[str]:
        return set(self.nodes)
