import re
from typing import List
from .circuit import Circuit, Element

_UNIT_SUFFIX = {
    # SPICE-ish suffixes (approx); keep simple for now
    "t": "1e12",
    "g": "1e9",
    "meg": "1e6",
    "k": "1e3",
    "m": "1e-3",
    "u": "1e-6",
    "n": "1e-9",
    "p": "1e-12",
    "f": "1e-15",
}

def _strip_comment(line: str) -> str:
    # remove inline comments starting with ';'
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()

def _is_comment_or_empty(line: str) -> bool:
    return (not line) or line.startswith("*")

def _normalize_value(val: str) -> str:
    """
    Keep symbolic values as-is (R, C, L, etc).
    Convert common suffix forms like 10k, 15u, 1meg to scientific notation strings.
    """
    v = val.strip()
    # if it contains letters and is not a suffix pattern, leave symbolic
    # e.g. 'R', 'C1' shouldn't appear here, but keep safe
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-zA-Z]+)", v)
    if not m:
        return v

    num, suf = m.group(1), m.group(2).lower()
    if suf in _UNIT_SUFFIX:
        return f"({num}*{_UNIT_SUFFIX[suf]})"
    # handle 'meg' already; any unknown suffix -> leave as is
    return v

def parse_spice(text: str) -> Circuit:
    elements: List[Element] = []
    nodes = set()

    for raw in text.splitlines():
        line = _strip_comment(raw)
        if _is_comment_or_empty(line):
            continue
        if line.lower().startswith(".end"):
            break
        if line.startswith("."):
            # ignore other directives for v1 (we can add later)
            continue

        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"SPICE line too short: {raw}")

        name = parts[0]
        kind = name[0].upper()

        if kind not in {"R", "C", "L", "V", "I", "E", "O", "T"}:
            raise ValueError(f"Unsupported element '{name}' in line: {raw}")

        n1, n2 = parts[1], parts[2]

        # value + optional extra for sources
        # value + optional extra for sources / dependent sources
        if kind in {"R", "C", "L"}:
            value = _normalize_value(parts[3])
            extra = None

        elif kind in {"V", "I"}:
            # V/I source: allow 'DC 5', 'SIN(...)', 'PULSE(...)', or just a number
            if len(parts) == 4:
                value = _normalize_value(parts[3])
                extra = None
            else:
                tail = " ".join(parts[3:])
                if parts[3].upper() == "DC" and len(parts) >= 5:
                    value = _normalize_value(parts[4])
                    extra = "DC"
                else:
                    value = parts[3]  # best effort
                    extra = tail

        elif kind == "E":
            # VCVS: Ename nout nref ncp ncn gain
            if len(parts) < 6:
                raise ValueError(f"VCVS (E) expects: Ename nout nref ncp ncn gain. Line: {raw}")

            ncp = parts[3]
            ncn = parts[4]
            gain_raw = parts[5]
            extra = f"{ncp} {ncn}"

            if gain_raw.upper() == "INF":
                value = f"A_{name}"
            else:
                value = _normalize_value(gain_raw)

        elif kind == "O":
            # Ideal op-amp:
            # 1) Oname nout nref nplus nminus
            # 2) Oname nout nplus nminus   (nref defaults to 0)
            if len(parts) == 5:
                n1, n2 = parts[1], parts[2]
                nplus, nminus = parts[3], parts[4]
            elif len(parts) == 4:
                n1, n2 = parts[1], "0"
                nplus, nminus = parts[2], parts[3]
            else:
                raise ValueError(
                    f"Ideal op-amp (O) expects: Oname nout nref nplus nminus "
                    f"or Oname nout nplus nminus. Line: {raw}"
                )
            value = "IDEAL"
            extra = f"{nplus} {nminus}"

        elif kind == "T":
            # Ideal transmission line:
            # Tname n1 n2 n3 n4 Zc tau
            # Port1: n1(+), n2(-), Port2: n3(+), n4(-)
            if len(parts) < 7:
                raise ValueError(
                    f"Transmission line (T) expects: Tname n1 n2 n3 n4 Zc tau. Line: {raw}"
                )
            n1, n2 = parts[1], parts[2]
            n3, n4 = parts[3], parts[4]
            zc = _normalize_value(parts[5])
            tau = _normalize_value(parts[6])
            value = zc
            extra = f"{n3} {n4} {tau}"

        elements.append(Element(kind=kind, name=name, n1=n1, n2=n2, value=value, extra=extra))
        nodes.add(n1); nodes.add(n2)
        if kind in {"E", "O"} and extra is not None:
            ncp, ncn = extra.split()[:2]
            nodes.add(ncp)
            nodes.add(ncn)
        if kind == "T" and extra is not None:
            n3, n4, _tau = extra.split()[:3]
            nodes.add(n3)
            nodes.add(n4)

    return Circuit(elements=elements, nodes=nodes)
