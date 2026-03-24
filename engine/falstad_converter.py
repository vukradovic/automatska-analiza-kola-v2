from __future__ import annotations

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class FalstadElement:
    kind: str
    tokens: list[str]


@dataclass(frozen=True)
class WireSegment:
    a: tuple[int, int]
    b: tuple[int, int]


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _coord(x: str, y: str) -> str:
    return f"{x},{y}"


def _coord_int(x: int, y: int) -> str:
    return f"{x},{y}"


def _is_int_token(token: str) -> bool:
    if not token:
        return False
    if token[0] in "+-":
        token = token[1:]
    return token.isdigit()


def _split_lines(text: str) -> list[tuple[int, list[str]]]:
    parsed: list[tuple[int, list[str]]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("$"):
            continue
        parts = line.split()
        parsed.append((lineno, parts))
    return parsed


def _parse_elements(lines: list[tuple[int, list[str]]]) -> tuple[list[FalstadElement], _UnionFind, set[str], list[WireSegment], set[str]]:
    elements: list[FalstadElement] = []
    uf = _UnionFind()
    grounded_points: set[str] = set()
    wire_segments: list[WireSegment] = []
    known_points: set[str] = set()

    def register_point(x: str, y: str) -> None:
        p = _coord(x, y)
        known_points.add(p)
        uf.add(p)

    for lineno, parts in lines:
        kind = parts[0].lower()
        if kind in {"h", "%", "?"}:
            raise ValueError(f"Unsupported Falstad line type '{parts[0]}' at line {lineno}")

        if kind == "o":
            if len(parts) < 3:
                raise ValueError(f"Analog output line too short at line {lineno}")
            probe_point = _coord(parts[1], parts[2])
            known_points.add(probe_point)
            uf.add(probe_point)
            elements.append(FalstadElement(kind=kind, tokens=parts))
            continue

        if kind == "x":
            continue

        if kind == "w":
            if len(parts) < 5:
                raise ValueError(f"Wire line too short at line {lineno}")
            register_point(parts[1], parts[2])
            register_point(parts[3], parts[4])
            a = (int(parts[1]), int(parts[2]))
            b = (int(parts[3]), int(parts[4]))
            wire_segments.append(WireSegment(a=a, b=b))
            uf.union(_coord_int(*a), _coord_int(*b))
            continue

        if kind == "g":
            if len(parts) < 3:
                raise ValueError(f"Ground line too short at line {lineno}")
            register_point(parts[1], parts[2])
            grounded_points.add(_coord(parts[1], parts[2]))
            continue

        if len(parts) >= 5 and all(_is_int_token(tok) for tok in parts[1:5]):
            register_point(parts[1], parts[2])
            register_point(parts[3], parts[4])
            if kind == "a":
                plus_point, minus_point = _opamp_input_points(
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                )
                known_points.add(_coord_int(*plus_point))
                known_points.add(_coord_int(*minus_point))
                uf.add(_coord_int(*plus_point))
                uf.add(_coord_int(*minus_point))
            elements.append(FalstadElement(kind=kind, tokens=parts))
            continue

        raise ValueError(f"Unsupported or malformed Falstad line at line {lineno}: {' '.join(parts)}")

    return elements, uf, grounded_points, wire_segments, known_points


def _point_on_segment(point: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if cross != 0:
        return False
    return (
        min(ax, bx) <= px <= max(ax, bx)
        and min(ay, by) <= py <= max(ay, by)
    )


def _connect_points_on_wires(uf: _UnionFind, known_points: set[str], wire_segments: list[WireSegment]) -> None:
    points = []
    for raw in known_points:
        xs, ys = raw.split(",", 1)
        points.append((int(xs), int(ys)))

    for wire in wire_segments:
        anchor = _coord_int(*wire.a)
        for point in points:
            if _point_on_segment(point, wire.a, wire.b):
                uf.union(anchor, _coord_int(*point))


def _assign_nodes(uf: _UnionFind, grounded_points: set[str], elements: list[FalstadElement]) -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    for point in uf.parent:
        root = uf.find(point)
        groups.setdefault(root, []).append(point)

    ground_roots = {uf.find(point) for point in grounded_points}
    node_map: dict[str, str] = {}
    next_idx = 1

    for root in sorted(groups):
        if root in ground_roots:
            name = "0"
        else:
            name = f"n{next_idx}"
            next_idx += 1
        for point in groups[root]:
            node_map[point] = name

    for point in grounded_points:
        node_map[point] = "0"
    return node_map


def _extract_output_nodes(
    uf: _UnionFind,
    node_map: dict[str, str],
    elements: list[FalstadElement],
) -> list[str]:
    outputs: list[str] = []
    for elem in elements:
        if elem.kind != "o":
            continue
        probe_point = _coord(elem.tokens[1], elem.tokens[2])
        node_name = node_map.get(probe_point)
        if node_name and node_name not in outputs:
            outputs.append(node_name)
    return outputs


def _value_at(tokens: list[str], index: int, default: str | None = None) -> str:
    if len(tokens) > index:
        return tokens[index]
    if default is None:
        raise ValueError(f"Missing value in Falstad line: {' '.join(tokens)}")
    return default


def _opamp_input_points(x1: int, y1: int, x2: int, y2: int) -> tuple[tuple[int, int], tuple[int, int]]:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("Falstad op-amp has zero-length geometry.")

    perp_x = int(round((-dy / length) * 16.0))
    perp_y = int(round((dx / length) * 16.0))
    plus_point = (x1 + perp_x, y1 + perp_y)
    minus_point = (x1 - perp_x, y1 - perp_y)
    return plus_point, minus_point


def convert_falstad_to_spice(text: str) -> str:
    lines = _split_lines(text)
    if not lines:
        raise ValueError("Falstad netlist is empty.")

    elements, uf, grounded_points, wire_segments, known_points = _parse_elements(lines)
    _connect_points_on_wires(uf, known_points, wire_segments)
    node_map = _assign_nodes(uf, grounded_points, elements)
    output_nodes = _extract_output_nodes(uf, node_map, elements)
    output: list[str] = ["* Auto-generated from Falstad export"]
    if len(output_nodes) == 1:
        output.append(f"* OUT_NODE: {output_nodes[0]}")
    elif len(output_nodes) > 1:
        output.append(f"* OUT_NODES: {', '.join(output_nodes)}")
    counters: dict[str, int] = {}

    def next_name(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}{counters[prefix]}"

    for elem in elements:
        tokens = elem.tokens
        kind = elem.kind

        if kind == "o":
            continue

        n1 = node_map[_coord(tokens[1], tokens[2])]
        n2 = node_map[_coord(tokens[3], tokens[4])]

        if kind == "r":
            output.append(f"{next_name('R')} {n1} {n2} {_value_at(tokens, 6)}")
        elif kind == "c":
            output.append(f"{next_name('C')} {n1} {n2} {_value_at(tokens, 6)}")
        elif kind == "l":
            output.append(f"{next_name('L')} {n1} {n2} {_value_at(tokens, 6)}")
        elif kind == "v":
            dc_value = _value_at(tokens, 8, _value_at(tokens, 6, "1"))
            output.append(f"{next_name('V')} {n1} {n2} DC {dc_value}")
        elif kind == "i":
            dc_value = _value_at(tokens, 8, _value_at(tokens, 6, "1"))
            output.append(f"{next_name('I')} {n1} {n2} DC {dc_value}")
        elif kind == "a":
            plus_point, minus_point = _opamp_input_points(
                int(tokens[1]),
                int(tokens[2]),
                int(tokens[3]),
                int(tokens[4]),
            )
            nout = n2
            nplus = node_map[_coord_int(*plus_point)]
            nminus = node_map[_coord_int(*minus_point)]
            output.append(f"{next_name('OOP')} {nout} 0 {nplus} {nminus}")
        else:
            raise ValueError(
                f"Falstad element '{tokens[0]}' is not supported yet. "
                "Supported now: wires, ground, R, C, L, V, I, ideal op-amp."
            )

    output.append(".end")
    return "\n".join(output)


_NUMERIC_VALUE_RE = re.compile(
    r"^\(?[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\)?$"
)


def _is_numeric_value(value: str) -> bool:
    v = value.strip()
    return bool(_NUMERIC_VALUE_RE.fullmatch(v))


def netlist_to_symbolic(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            lines.append(raw)
            continue

        parts = raw.split()
        if not parts:
            lines.append(raw)
            continue

        kind = parts[0][0].upper()
        if kind == "R" and len(parts) >= 4 and _is_numeric_value(parts[3]):
            parts[3] = "R"
        elif kind == "C" and len(parts) >= 4 and _is_numeric_value(parts[3]):
            parts[3] = "C"
        elif kind == "L" and len(parts) >= 4 and _is_numeric_value(parts[3]):
            parts[3] = "L"
        elif kind == "V":
            if len(parts) >= 5 and parts[3].upper() == "DC" and _is_numeric_value(parts[4]):
                parts[4] = "Ug"
            elif len(parts) >= 4 and _is_numeric_value(parts[3]):
                parts[3] = "Ug"
        elif kind == "I":
            if len(parts) >= 5 and parts[3].upper() == "DC" and _is_numeric_value(parts[4]):
                parts[4] = "Ig"
            elif len(parts) >= 4 and _is_numeric_value(parts[3]):
                parts[3] = "Ig"
        elif kind == "E" and len(parts) >= 6 and _is_numeric_value(parts[5]):
            parts[5] = "A"
        elif kind == "T":
            if len(parts) >= 6 and _is_numeric_value(parts[5]):
                parts[5] = "Zc"
            if len(parts) >= 7 and _is_numeric_value(parts[6]):
                parts[6] = "tau"

        lines.append(" ".join(parts))

    return "\n".join(lines)
