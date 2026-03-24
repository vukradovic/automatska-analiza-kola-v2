from __future__ import annotations

import base64
import io
import json
import mimetypes
import re
import sys
import imghdr
from pathlib import Path
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sympy as sp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is importable regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.spice_parser import parse_spice
from engine.falstad_converter import convert_falstad_to_spice, netlist_to_symbolic
from engine.analysis import (
    transfer_function,
    system_order,
    zeros_poles_symbolic,
    amplitude_response_symbolic,
    symbolic_3db_band,
    amplitude_curve_from_transfer,
    amplitude_markers_numeric,
    evaluate_measure,
    inverse_laplace_with_delay,
)

WEB_ROOT = PROJECT_ROOT / "web"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
IMAGES_DIR = PROJECT_ROOT / "assets" / "circuit_images"


def _compact_expr(expr):
    try:
        return sp.simplify(sp.cancel(sp.together(expr), extension=True))
    except Exception:
        return sp.simplify(expr)


def _default_numeric_substitutions(*exprs):
    s_sym = sp.Symbol("s")
    defaults_by_name = {
        "Ug": 1.0,
        "A": 1e6,
    }

    syms = set()
    for e in exprs:
        syms.update(e.free_symbols)

    syms = sorted((sym for sym in syms if sym != s_sym), key=lambda x: str(x))
    subs = {}
    for sym in syms:
        name = str(sym)
        if name in defaults_by_name:
            subs[sym] = defaults_by_name[name]
        elif name.startswith("A"):
            subs[sym] = 1e6
        else:
            subs[sym] = 1.0
    return subs


def _resolve_netlist(path_value: str) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        fallback = EXAMPLES_DIR / path_value
        if fallback.exists():
            p = fallback
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Netlist not found: {path_value}")
    if PROJECT_ROOT not in p.parents and p != PROJECT_ROOT:
        raise ValueError("Netlist path outside project is not allowed.")
    return p


def _extract_image_hint(netlist_path: Path) -> Path | None:
    pattern = re.compile(r"IMG\s*:\s*(.+)$", re.IGNORECASE)
    for raw in netlist_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*") or line.startswith(";"):
            m = pattern.search(line)
            if not m:
                continue
            img_raw = m.group(1).strip()
            raw_path = Path(img_raw)
            candidates = []
            if raw_path.is_absolute():
                candidates.append(raw_path.resolve())
            else:
                # 1) Relative to netlist directory
                candidates.append((netlist_path.parent / raw_path).resolve())
                # 2) Relative to project root (more convenient for shared assets/)
                candidates.append((PROJECT_ROOT / raw_path).resolve())

            for cand in candidates:
                if cand.exists() and cand.is_file():
                    return cand
    return None


def _extract_node_hint(netlist_path: Path, key: str) -> str | None:
    pattern = re.compile(rf"^\*\s*{re.escape(key)}\s*:\s*(.+?)\s*$", re.IGNORECASE)
    for raw in netlist_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(raw.strip())
        if m:
            return m.group(1).strip()
    return None


def _file_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _latex_roots(roots):
    if not roots:
        return r"\varnothing"
    return r"\left[" + ", ".join(_latex_pretty(r) for r in roots) + r"\right]"


def _latex_pretty(expr):
    """
    UI display mapping:
    Om -> Omega
    """
    e = sp.sympify(expr)
    # Replace any symbol named 'Om' regardless of assumptions/signature.
    repl = {}
    for sym in e.free_symbols:
        if str(sym) == "Om":
            repl[sym] = sp.Symbol("Omega", real=True, positive=True)
    if repl:
        e = e.xreplace(repl)
    return sp.latex(e)


def _amplitude_plot_base64(omega, A, markers):
    fig = plt.figure(figsize=(8.0, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(omega, A, label="A(omega)")

    wp = markers.get("omega_peak")
    Ap = markers.get("A_peak")
    w1 = markers.get("omega_3db_low")
    w2 = markers.get("omega_3db_high")
    A3 = markers.get("A_3db")

    if wp is not None and Ap is not None:
        ax.plot([wp], [Ap], marker="o")
        ax.axvline(wp, linestyle="--", alpha=0.6)

    if A3 is not None:
        ax.axhline(A3, linestyle="--", alpha=0.6)
    if w1 is not None:
        ax.axvline(w1, linestyle="--", alpha=0.6)
    if w2 is not None:
        ax.axvline(w2, linestyle="--", alpha=0.6)
    if w1 is not None and w2 is not None and A3 is not None:
        ax.fill_between(omega, 0.0, A, where=(omega >= w1) & (omega <= w2), alpha=0.12)

    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("A(omega)")
    ax.set_title("Amplitude characteristic A(omega)")
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _expr_time_plot_base64(expr_t, t_max=6.0, t_pre=1.0, n=1000, title="measure(t)"):
    t = sp.Symbol("t")
    heaviside = lambda x, y=0.0: np.heaviside(x, y)
    f = sp.lambdify(t, expr_t, modules=[{"Heaviside": heaviside}, "numpy"])

    ts = np.linspace(-float(t_pre), float(t_max), int(n))
    ys = f(ts)
    if np.isscalar(ys):
        ys = np.full_like(ts, float(ys), dtype=float)
    else:
        ys = np.asarray(ys, dtype=float).reshape(-1)
        if ys.size == 1:
            ys = np.full_like(ts, float(ys[0]), dtype=float)

    fig = plt.figure(figsize=(8.0, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ts, ys)
    ax.axvline(0.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(True)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def analyze_payload(payload: dict) -> dict:
    netlist_path = _resolve_netlist(payload.get("netlist", "examples/opamp.cir"))
    in_node = str(payload.get("in_node", "in"))
    out_node = str(payload.get("out_node", "out"))
    analyses = payload.get("analyses", {})
    measures = payload.get("measures", []) or []
    measure_time = bool(payload.get("measure_time", False))
    measure_plot = bool(payload.get("measure_plot", False))

    text = netlist_path.read_text(encoding="utf-8")
    ckt = parse_spice(text)

    H, Vout, Vin = transfer_function(ckt, out_node=out_node, in_node=in_node)

    Vin_c = _compact_expr(Vin)
    Vout_c = _compact_expr(Vout)
    H_c = _compact_expr(H)

    result = {
        "ok": True,
        "netlist": str(netlist_path.relative_to(PROJECT_ROOT)),
        "raw": {
            "vin": str(Vin),
            "vout": str(Vout),
            "h": str(H),
        },
        "compact": {
            "vin": str(Vin_c),
            "vout": str(Vout_c),
            "h": str(H_c),
        },
        "latex": {
            "vin": _latex_pretty(Vin_c),
            "vout": _latex_pretty(Vout_c),
            "h": _latex_pretty(H_c),
        },
        "system_order": system_order(H_c),
    }

    img_hint = _extract_image_hint(netlist_path)
    if img_hint is not None:
        result["circuit_image_data_uri"] = _file_to_data_uri(img_hint)

    if analyses.get("symbolic_freq"):
        s_sym = sp.Symbol("s")
        repl = {}
        for sym in H_c.free_symbols:
            if sym == s_sym:
                continue
            repl[sym] = sp.Symbol(str(sym), real=True, positive=True)
        H_freq = sp.simplify(H_c.subs(repl)) if repl else H_c

        try:
            zeros, poles = zeros_poles_symbolic(H_freq)
        except Exception:
            zeros, poles = [], []
        omega, Hjw, A_w, _ = amplitude_response_symbolic(H_freq)

        freq = {
            "zeros": [str(z) for z in zeros],
            "poles": [str(p) for p in poles],
            "latex": {
                "h_jw": _latex_pretty(Hjw),
                "a_w": _latex_pretty(A_w),
                "zeros": _latex_roots(zeros),
                "poles": _latex_roots(poles),
            },
        }

        if analyses.get("symbolic_3db"):
            band = symbolic_3db_band(H_freq, omega_symbol=omega)
            freq["band_3db"] = {
                "omega_peak": str(band.get("omega_peak")),
                "omega_3db_low": str(band.get("omega_3db_low")),
                "omega_3db_high": str(band.get("omega_3db_high")),
                "bandwidth_3db": str(band.get("bandwidth_3db")),
                "latex": {
                    "omega_peak": _latex_pretty(band.get("omega_peak")) if band.get("omega_peak") is not None else None,
                    "omega_3db_low": _latex_pretty(band.get("omega_3db_low")) if band.get("omega_3db_low") is not None else None,
                    "omega_3db_high": _latex_pretty(band.get("omega_3db_high")) if band.get("omega_3db_high") is not None else None,
                    "bandwidth_3db": _latex_pretty(band.get("bandwidth_3db")) if band.get("bandwidth_3db") is not None else None,
                },
            }

        result["symbolic_freq"] = freq

    if analyses.get("amp_plot"):
        H_amp = H_c
        if H_amp.free_symbols - {sp.Symbol("s")}:
            subs = _default_numeric_substitutions(H_amp)
            H_amp = sp.simplify(H_amp.subs(subs))
            result["amp_plot_defaults"] = {str(k): v for k, v in subs.items()}

        w_max = float(payload.get("amp_w_max", 7.0))
        n_pts = int(payload.get("amp_points", 1200))
        omega_arr, A_arr = amplitude_curve_from_transfer(H_amp, w_max=w_max, n=n_pts)
        markers = amplitude_markers_numeric(omega_arr, A_arr)

        result["amp_plot"] = {
            "markers": markers,
            "image_base64": _amplitude_plot_base64(omega_arr, A_arr, markers),
        }

    if measures:
        s = sp.Symbol("s")
        t = sp.Symbol("t")
        out_items = []
        for meas in measures:
            ms = evaluate_measure(ckt, meas)
            msc = _compact_expr(ms)
            item = {
                "name": meas,
                "s_expr": str(msc),
                "s_latex": _latex_pretty(msc),
            }

            mt = None
            if measure_time or measure_plot:
                mt = sp.simplify(inverse_laplace_with_delay(msc, s=s, t=t))
                item["t_expr"] = str(mt)
                item["t_latex"] = _latex_pretty(mt)

            if measure_plot and mt is not None:
                mplot = mt
                subs = _default_numeric_substitutions(mplot)
                subs = {k: v for k, v in subs.items() if str(k) != "t"}
                if subs:
                    item["plot_defaults"] = {str(k): v for k, v in subs.items()}
                    mplot = sp.simplify(mplot.subs(subs))

                item["plot_base64"] = _expr_time_plot_base64(
                    mplot,
                    t_max=float(payload.get("t_max", 6.0)),
                    t_pre=float(payload.get("t_pre", 1.0)),
                    n=int(payload.get("n", 1000)),
                    title=f"{meas}(t)",
                )

            out_items.append(item)

        result["measures"] = out_items

    return result


def list_netlists() -> list[dict]:
    items = []
    for p in sorted(EXAMPLES_DIR.glob("*.cir")):
        hint = _extract_image_hint(p)
        out_node_hint = _extract_node_hint(p, "OUT_NODE")
        in_node_hint = _extract_node_hint(p, "IN_NODE")
        items.append(
            {
                "path": str(p.relative_to(PROJECT_ROOT)),
                "name": p.name,
                "has_image": hint is not None,
                "out_node_hint": out_node_hint,
                "in_node_hint": in_node_hint,
            }
        )
    return items


def save_netlist(payload: dict) -> dict:
    name = str(payload.get("name", "")).strip()
    text = str(payload.get("netlist_text", ""))
    image_data = payload.get("image_data")
    if not name:
        raise ValueError("Netlist name is required.")
    if not text.strip():
        raise ValueError("Netlist text is empty.")

    base = name[:-4] if name.lower().endswith(".cir") else name
    if not re.fullmatch(r"[A-Za-z0-9._-]+", base):
        raise ValueError("Netlist name may contain only letters, numbers, dot, dash and underscore.")

    path = (EXAMPLES_DIR / f"{base}.cir").resolve()
    if path.parent != EXAMPLES_DIR.resolve():
        raise ValueError("Invalid netlist path.")

    final_text = text.rstrip()
    image_rel_path = None

    if image_data:
        m = re.fullmatch(r"data:(image/[A-Za-z0-9.+-]+);base64,(.+)", str(image_data), re.DOTALL)
        if not m:
            raise ValueError("Invalid image payload.")
        raw_bytes = base64.b64decode(m.group(2), validate=True)
        image_kind = imghdr.what(None, raw_bytes)
        ext = {
            "png": ".png",
            "jpeg": ".jpg",
            "gif": ".gif",
            "bmp": ".bmp",
            "webp": ".webp",
        }.get(image_kind)
        if ext is None:
            raise ValueError("Unsupported image format. Use PNG/JPG/GIF/BMP/WEBP.")

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        image_path = (IMAGES_DIR / f"{base}{ext}").resolve()
        if image_path.parent != IMAGES_DIR.resolve():
            raise ValueError("Invalid image path.")
        image_path.write_bytes(raw_bytes)
        image_rel_path = str(image_path.relative_to(PROJECT_ROOT))

        img_line = f"* IMG: {image_rel_path}"
        lines = final_text.splitlines()
        has_img = any(line.strip().lower().startswith("* img:") for line in lines)
        if has_img:
            final_text = "\n".join(
                img_line if line.strip().lower().startswith("* img:") else line
                for line in lines
            )
        else:
            final_text = f"{img_line}\n{final_text}"

    path.write_text(final_text + "\n", encoding="utf-8")
    return {
        "ok": True,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "name": path.name,
        "image_path": image_rel_path,
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str = "text/html; charset=utf-8"):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            return self._send_file(WEB_ROOT / "index.html")
        if parsed.path == "/create-netlist":
            return self._send_file(WEB_ROOT / "create_netlist.html")
        if parsed.path == "/api/netlists":
            return self._send_json({"ok": True, "items": list_netlists()})

        return self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n)
            payload = json.loads(body.decode("utf-8"))
            if parsed.path == "/api/analyze":
                res = analyze_payload(payload)
                return self._send_json(res)
            if parsed.path == "/api/convert-falstad":
                falstad_text = str(payload.get("falstad_text", "")).strip()
                translated = convert_falstad_to_spice(falstad_text)
                return self._send_json({"ok": True, "translated_netlist": translated})
            if parsed.path == "/api/to-symbolic":
                netlist_text = str(payload.get("netlist_text", ""))
                symbolic = netlist_to_symbolic(netlist_text)
                return self._send_json({"ok": True, "netlist_text": symbolic})
            if parsed.path == "/api/save-netlist":
                return self._send_json(save_netlist(payload))
            return self._send_json({"ok": False, "error": "Not found"}, status=404)
        except Exception as exc:
            return self._send_json({"ok": False, "error": str(exc)}, status=400)


def main():
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
