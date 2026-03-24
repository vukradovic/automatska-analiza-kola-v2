import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re

from .mna import solve_mna_s_domain


try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - graceful runtime fallback
    signal = None
    SCIPY_AVAILABLE = False


def transfer_function(circuit, out_node: str, in_node: str):
    Vnodes, x, xsyms = solve_mna_s_domain(circuit)

    if out_node not in Vnodes:
        raise ValueError(f"Unknown out_node='{out_node}'. Available: {list(Vnodes.keys())}")
    if in_node not in Vnodes:
        raise ValueError(f"Unknown in_node='{in_node}'. Available: {list(Vnodes.keys())}")

    # Keep initial simplification light; aggressive simplify can explode.
    Vout = sp.together(Vnodes[out_node])
    Vin = sp.together(Vnodes[in_node])

    if sp.simplify(Vin) == 0:
        raise ZeroDivisionError(f"Vin for node '{in_node}' is 0, cannot form transfer function.")

    # Critical: cancel common polynomial factors, including algebraic
    # extensions (e.g. sqrt(2)) common in textbook filter parametrizations.
    H = sp.cancel(sp.together(Vout / Vin), extension=True)
    H = sp.simplify(H)
    return H, Vout, Vin


def solve_circuit_expressions(circuit):
    """
    Returns:
      node_voltages: dict[node_name] -> V(node,s)
      branch_currents: dict[element_name] -> I(element,s), orientation n1 -> n2
    """
    s = sp.Symbol("s")
    Vnodes, x, xsyms = solve_mna_s_domain(circuit)

    # Unknown currents already solved by MNA (V/E/O/T elements).
    branch_currents = {}
    for i, sym in enumerate(xsyms):
        name = str(sym)
        if name.startswith("I_"):
            branch_currents[name[2:]] = sp.simplify(x[i, 0])

    def v_of(node_name):
        if node_name == "0":
            return sp.Integer(0)
        return Vnodes[node_name]

    # Passive currents from constitutive relations.
    for e in circuit.elements:
        if e.name in branch_currents:
            continue
        if e.kind not in {"R", "C", "L"}:
            continue

        vdiff = sp.simplify(v_of(e.n1) - v_of(e.n2))
        val = sp.sympify(e.value)
        if e.kind == "R":
            i_expr = vdiff / val
        elif e.kind == "C":
            i_expr = s * val * vdiff
        else:  # L
            i_expr = vdiff / (s * val)
        branch_currents[e.name] = sp.simplify(i_expr)

    return Vnodes, branch_currents


_MEAS_V1_RE = re.compile(r"^\s*V\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*$", re.IGNORECASE)
_MEAS_V2_RE = re.compile(
    r"^\s*V\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*$", re.IGNORECASE
)
_MEAS_I_RE = re.compile(r"^\s*I\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*$", re.IGNORECASE)


def evaluate_measure(circuit, measure: str):
    """
    Supported:
      V(node)
      V(node_plus,node_minus)
      I(element_name)
    """
    Vnodes, Ielems = solve_circuit_expressions(circuit)

    m = _MEAS_V2_RE.match(measure)
    if m:
        a, b = m.group(1), m.group(2)
        va = sp.Integer(0) if a == "0" else Vnodes[a]
        vb = sp.Integer(0) if b == "0" else Vnodes[b]
        return sp.simplify(va - vb)

    m = _MEAS_V1_RE.match(measure)
    if m:
        a = m.group(1)
        va = sp.Integer(0) if a == "0" else Vnodes[a]
        return sp.simplify(va)

    m = _MEAS_I_RE.match(measure)
    if m:
        ename = m.group(1)
        if ename not in Ielems:
            raise ValueError(
                f"Unknown element '{ename}' for current measure. Available: {sorted(Ielems.keys())}"
            )
        return sp.simplify(Ielems[ename])

    raise ValueError(
        f"Unsupported measure syntax '{measure}'. Use V(node), V(a,b), or I(element)."
    )


def time_response_from_transfer(H, Vin_s):
    s = sp.Symbol("s")
    t = sp.Symbol("t")  # keep t without positive=True, so Heaviside is explicit

    Vout_s = sp.simplify(H * Vin_s)
    vout_t = inverse_laplace_with_delay(Vout_s, s=s, t=t)

    return sp.simplify(vout_t)


def inverse_laplace_with_delay(expr_s, s=None, t=None):
    """
    Inverse Laplace with explicit handling of delay factors exp(-s*tau):
      exp(-s*tau) * F(s) -> f(t-tau) * Heaviside(t-tau)
    """
    s = s or sp.Symbol("s")
    t = t or sp.Symbol("t")

    def _extract_delay(term):
        delay = sp.Integer(0)
        rest = sp.powsimp(sp.simplify(term), force=True)

        for fac in list(sp.Mul.make_args(rest)):
            if fac.func != sp.exp:
                continue
            arg = sp.simplify(fac.args[0]).expand()

            # Require pure linear-in-s exponent: arg = coeff*s
            coeff = sp.simplify(sp.diff(arg, s))
            residual = sp.simplify(arg - coeff * s)
            if residual != 0 or coeff.has(s):
                continue

            # Keep only physical delays exp(-s*tau), ignore advances.
            if coeff.is_negative is False:
                continue

            d = sp.simplify(-coeff)
            delay += d
            rest = sp.simplify(rest / fac)
        return sp.simplify(delay), sp.simplify(rest)

    expr_s = sp.cancel(sp.together(sp.simplify(expr_s)), extension=True)

    total = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(expr_s)):
        delay, base = _extract_delay(term)
        base_t = sp.inverse_laplace_transform(base, s, t, noconds=True)
        if delay != 0:
            total += sp.simplify(base_t.subs(t, t - delay) * sp.Heaviside(t - delay))
        else:
            total += sp.simplify(base_t)

    total = total.replace(
        lambda e: isinstance(e, sp.Pow) and e.base.func == sp.Heaviside and e.exp.is_Integer and e.exp >= 1,
        lambda e: e.base,
    )
    return sp.simplify(total)


def _to_numeric_float(expr):
    val = complex(sp.N(expr))
    if abs(val.imag) > 1e-10:
        raise ValueError(f"Expression is not real numeric: {expr}")
    return float(val.real)


def _step_amplitude_from_vin(Vin_s):
    s = sp.Symbol("s")
    K = sp.simplify(Vin_s * s)
    if s in K.free_symbols:
        raise ValueError(
            "Numeric step mode currently supports Vin(s)=K/s (step) with constant K."
        )
    if K.free_symbols:
        raise ValueError(f"Step amplitude is symbolic ({K}); provide numeric values.")
    return _to_numeric_float(K)


def _rational_coeffs(H):
    s = sp.Symbol("s")
    H_r = sp.together(sp.simplify(H))
    num, den = sp.fraction(H_r)

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    if num_poly is None or den_poly is None:
        raise ValueError("H(s) is not polynomial ratio in s.")

    coeffs = num_poly.all_coeffs() + den_poly.all_coeffs()
    if any(c.free_symbols for c in coeffs):
        raise ValueError("H(s) contains unresolved symbolic parameters.")

    num_c = [float(complex(sp.N(c)).real) for c in num_poly.all_coeffs()]
    den_c = [float(complex(sp.N(c)).real) for c in den_poly.all_coeffs()]
    return num_c, den_c


def system_order(H):
    s = sp.Symbol("s")
    try:
        _, den = sp.fraction(sp.together(sp.simplify(H)))
        den_poly = sp.Poly(den, s)
        if den_poly is None:
            return None
        return int(den_poly.degree())
    except Exception:
        # Delay systems with exp(-s*tau) are not polynomial in s.
        return None


def estimate_tau_from_transfer(H):
    """
    If H has a single real pole p<0, returns tau=-1/p.
    Otherwise returns None.
    """
    s = sp.Symbol("s")
    try:
        _, den = sp.fraction(sp.together(sp.simplify(H)))
        den_poly = sp.Poly(den, s)
        if den_poly is None or den_poly.degree() != 1:
            return None

        roots = [complex(r.evalf()) for r in sp.nroots(den_poly)]
        if len(roots) != 1:
            return None

        p = roots[0]
        if abs(p.imag) > 1e-10 or p.real >= 0:
            return None

        return -1.0 / p.real
    except Exception:
        # Non-rational / delay systems are not compatible with this estimator.
        return None


def numeric_step_response_from_transfer(H, Vin_s, t_max=1.0, t_pre=0.2, n=1200):
    """
    Numeric step response using scipy.signal for rational H(s) and Vin(s)=K/s.
    Returns (ts, ys) over [-t_pre, t_max] with explicit pre-step zero segment.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not available; cannot run numeric step mode.")

    K = _step_amplitude_from_vin(Vin_s)
    num, den = _rational_coeffs(H)

    lti = signal.TransferFunction(num, den)

    t_max = float(t_max)
    t_pre = float(t_pre)
    n = int(n)

    if t_max <= 0:
        raise ValueError("t_max must be > 0")
    if t_pre < 0:
        raise ValueError("t_pre must be >= 0")
    if n < 10:
        raise ValueError("n should be >= 10")

    frac_pre = t_pre / (t_pre + t_max) if (t_pre + t_max) > 0 else 0.0
    n_pre = int(max(2, round(n * frac_pre))) if t_pre > 0 else 0
    n_pos = max(2, n - n_pre)

    t_pos = np.linspace(0.0, t_max, n_pos)
    t_out, y_out = signal.step(lti, T=t_pos)
    y_out = K * np.asarray(y_out, dtype=float)

    if n_pre > 0:
        t_neg = np.linspace(-t_pre, 0.0, n_pre, endpoint=False)
        y_neg = np.zeros_like(t_neg)
        ts = np.concatenate([t_neg, t_out])
        ys = np.concatenate([y_neg, y_out])
    else:
        ts = t_out
        ys = y_out

    return ts, ys


def bode_from_transfer(H, w_min=1e-1, w_max=1e5, n=500):
    """
    Numeric Bode evaluation of H(jw), independent from SciPy.
    Returns (w, mag_db, phase_deg).
    """
    s = sp.Symbol("s")
    Hs = sp.simplify(H)

    if any(sym != s for sym in Hs.free_symbols):
        raise ValueError("H(s) has unresolved symbols; numeric Bode requires numeric parameters.")

    w = np.logspace(np.log10(float(w_min)), np.log10(float(w_max)), int(n))
    Hjw = sp.lambdify(s, Hs, modules="numpy")(1j * w)
    Hjw = np.asarray(Hjw, dtype=np.complex128)
    if Hjw.ndim == 0:
        Hjw = np.full_like(w, Hjw, dtype=np.complex128)

    mag_db = 20.0 * np.log10(np.maximum(np.abs(Hjw), 1e-300))
    phase_deg = np.unwrap(np.angle(Hjw)) * 180.0 / np.pi

    return w, mag_db, phase_deg


def zeros_poles_symbolic(H):
    """
    Symbolic zeros/poles from rational H(s).
    Returns (zeros, poles) as lists of roots (with multiplicity).
    """
    s = sp.Symbol("s")
    num, den = sp.fraction(sp.together(sp.simplify(H)))
    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    def _roots(poly_expr, poly_obj):
        if poly_obj.degree() <= 0:
            return []
        # Prefer exact polynomial roots first; for algebraic extensions
        # fallback to equation solving.
        try:
            return poly_obj.all_roots()
        except Exception:
            return sp.solve(sp.Eq(poly_expr, 0), s)

    zeros = _roots(num, num_poly)
    poles = _roots(den, den_poly)
    return zeros, poles


def amplitude_response_symbolic(H, omega_symbol=None):
    """
    Symbolic frequency response:
      Hjw = H(j*omega),
      A2  = |H(j*omega)|^2,
      A   = sqrt(A2).
    """
    s = sp.Symbol("s")
    omega = omega_symbol if omega_symbol is not None else sp.Symbol("omega", real=True)

    Hjw = sp.simplify(H.subs(s, sp.I * omega))
    A2 = sp.simplify(sp.together(sp.expand_complex(Hjw * sp.conjugate(Hjw))))
    A = sp.simplify(sp.sqrt(A2))
    return omega, Hjw, A, A2


def _positive_real_solutions(expr, var):
    """
    Solve expr(var)=0 and keep solutions that are real and positive
    (or not-provably-nonpositive).
    """
    target = sp.numer(sp.together(sp.simplify(expr)))
    raw = sp.solve(sp.Eq(target, 0), var)

    out = []
    for sol in raw:
        ssol = sp.simplify(sol)
        if ssol.has(sp.I):
            continue
        if ssol.is_real is False:
            continue
        if ssol.is_positive is False:
            continue
        out.append(ssol)
    return out


def symbolic_3db_band(H, omega_symbol=None):
    """
    Symbolic 3dB analysis using A(omega)^2.
    Returns dict with:
      omega, A2, omega_peak, A2_peak, omega_3db (list), omega_3db_low,
      omega_3db_high, bandwidth_3db.
    Notes:
      - Works best for filters with a unique positive-frequency peak.
      - If peak cannot be identified symbolically, 3dB roots may be empty.
    """
    omega, _, _, A2 = amplitude_response_symbolic(H, omega_symbol=omega_symbol)

    dA2 = sp.diff(A2, omega)
    critical = _positive_real_solutions(dA2, omega)

    omega_peak = critical[0] if len(critical) == 1 else None
    A2_peak = sp.simplify(A2.subs(omega, omega_peak)) if omega_peak is not None else None

    # Boundary candidates, useful for low-pass/high-pass symbolic cutoffs.
    A2_at_0 = sp.simplify(sp.limit(A2, omega, 0, dir="+"))
    A2_at_inf = sp.simplify(sp.limit(A2, omega, sp.oo))

    if A2_peak is None:
        if A2_at_0.is_finite and A2_at_inf == 0 and A2_at_0 != 0:
            omega_peak = sp.Integer(0)
            A2_peak = A2_at_0
        elif A2_at_inf.is_finite and A2_at_0 == 0 and A2_at_inf != 0:
            omega_peak = sp.oo
            A2_peak = A2_at_inf

    omega_3db = []
    if A2_peak is not None:
        eq_3db = sp.simplify(A2 - A2_peak / 2)
        omega_3db = _positive_real_solutions(eq_3db, omega)
        omega_3db = sorted(omega_3db, key=sp.default_sort_key)

    omega_3db_low = omega_3db[0] if len(omega_3db) >= 1 else None
    omega_3db_high = omega_3db[-1] if len(omega_3db) >= 2 else None
    bandwidth_3db = (
        sp.simplify(omega_3db_high - omega_3db_low)
        if omega_3db_low is not None and omega_3db_high is not None
        else None
    )

    return {
        "omega": omega,
        "A2": A2,
        "omega_peak": omega_peak,
        "A2_peak": A2_peak,
        "omega_3db": omega_3db,
        "omega_3db_low": omega_3db_low,
        "omega_3db_high": omega_3db_high,
        "bandwidth_3db": bandwidth_3db,
    }


def plot_time_response(v_t, t_max=1.0, t_pre=0.2, n=1200, title="Time response", tau=None):
    """
    Plots symbolic v(t) on [-t_pre, t_max], preserving Heaviside(t).
    """
    t = sp.Symbol("t")

    heaviside = lambda x, y=0.0: np.heaviside(x, y)
    f = sp.lambdify(t, v_t, modules=[{"Heaviside": heaviside}, "numpy"])

    ts = np.linspace(-float(t_pre), float(t_max), int(n))
    ys = f(ts)

    if np.isscalar(ys):
        ys = np.full_like(ts, float(ys), dtype=float)
    else:
        ys = np.asarray(ys, dtype=float).reshape(-1)
        if ys.size == 1:
            ys = np.full_like(ts, float(ys[0]), dtype=float)

    plt.figure()
    plt.plot(ts, ys)
    plt.axvline(0.0, linestyle="--")

    if tau is not None:
        tau = float(tau)
        y_tau = f(np.array([tau]))[0]
        plt.axvline(tau, linestyle="--")
        plt.plot([tau], [y_tau], marker="o")

    plt.xlabel("t [s]")
    plt.ylabel("v(t)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_numeric_time_response(ts, ys, title="Time response", tau=None):
    plt.figure()
    plt.plot(ts, ys)
    plt.axvline(0.0, linestyle="--")

    if tau is not None:
        tau = float(tau)
        y_tau = np.interp(tau, ts, ys)
        plt.axvline(tau, linestyle="--")
        plt.plot([tau], [y_tau], marker="o")

    plt.xlabel("t [s]")
    plt.ylabel("v(t)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_bode(w, mag_db, phase_deg, title="Bode plot"):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.semilogx(w, mag_db)
    ax1.set_ylabel("Magnitude [dB]")
    ax1.grid(True, which="both")

    ax2.semilogx(w, phase_deg)
    ax2.set_xlabel("omega [rad/s]")
    ax2.set_ylabel("Phase [deg]")
    ax2.grid(True, which="both")

    fig.suptitle(title)
    plt.show()


def amplitude_curve_from_transfer(H, w_max=10.0, n=800):
    """
    Numeric amplitude characteristic A(omega)=|H(j*omega)| on [0, w_max].
    H must be numeric with respect to all parameters except s.
    """
    s = sp.Symbol("s")
    Hs = sp.simplify(H)
    if any(sym != s for sym in Hs.free_symbols):
        raise ValueError("H(s) has unresolved symbols; provide numeric parameter values.")

    w_max = float(w_max)
    n = int(n)
    if w_max <= 0:
        raise ValueError("w_max must be > 0")
    if n < 10:
        raise ValueError("n should be >= 10")

    omega = np.linspace(0.0, w_max, n)
    Hjw = sp.lambdify(s, Hs, modules="numpy")(1j * omega)
    Hjw = np.asarray(Hjw, dtype=np.complex128)
    if Hjw.ndim == 0:
        Hjw = np.full_like(omega, Hjw, dtype=np.complex128)
    A = np.abs(Hjw)
    return omega, A


def amplitude_markers_numeric(omega, A):
    """
    Estimate peak and 3dB points from sampled amplitude curve.
    Returns dict with keys:
      omega_peak, A_peak, omega_3db_low, omega_3db_high, A_3db.
    """
    omega = np.asarray(omega, dtype=float)
    A = np.asarray(A, dtype=float)
    if omega.size != A.size or omega.size < 3:
        raise ValueError("omega and A must have same length >= 3")

    idx_peak = int(np.argmax(A))
    omega_peak = float(omega[idx_peak])
    A_peak = float(A[idx_peak])
    if A_peak <= 0:
        return {
            "omega_peak": omega_peak,
            "A_peak": A_peak,
            "omega_3db_low": None,
            "omega_3db_high": None,
            "A_3db": None,
        }

    A_3db = A_peak / np.sqrt(2.0)
    diff = A - A_3db

    def _interp_zero(x1, y1, x2, y2):
        if abs(y2 - y1) < 1e-15:
            return float(x1)
        return float(x1 - y1 * (x2 - x1) / (y2 - y1))

    omega_3db_low = None
    for i in range(idx_peak - 1, -1, -1):
        if diff[i] == 0:
            omega_3db_low = float(omega[i])
            break
        if diff[i] * diff[i + 1] < 0:
            omega_3db_low = _interp_zero(omega[i], diff[i], omega[i + 1], diff[i + 1])
            break

    omega_3db_high = None
    for i in range(idx_peak, len(diff) - 1):
        if diff[i] == 0:
            omega_3db_high = float(omega[i])
            break
        if diff[i] * diff[i + 1] < 0:
            omega_3db_high = _interp_zero(omega[i], diff[i], omega[i + 1], diff[i + 1])
            break

    return {
        "omega_peak": omega_peak,
        "A_peak": A_peak,
        "omega_3db_low": omega_3db_low,
        "omega_3db_high": omega_3db_high,
        "A_3db": float(A_3db),
    }


def plot_amplitude_characteristic(
    omega,
    A,
    title="Amplitude characteristic A(omega)",
    markers=None,
):
    plt.figure()
    plt.plot(omega, A, label="A(omega)")

    if markers is not None:
        wp = markers.get("omega_peak")
        Ap = markers.get("A_peak")
        w1 = markers.get("omega_3db_low")
        w2 = markers.get("omega_3db_high")
        A3 = markers.get("A_3db")

        if wp is not None and Ap is not None:
            plt.plot([wp], [Ap], marker="o")
            plt.axvline(wp, linestyle="--", alpha=0.6)

        if A3 is not None:
            plt.axhline(A3, linestyle="--", alpha=0.6)

        if w1 is not None:
            plt.axvline(w1, linestyle="--", alpha=0.6)
        if w2 is not None:
            plt.axvline(w2, linestyle="--", alpha=0.6)
        if w1 is not None and w2 is not None and A3 is not None:
            plt.fill_between(
                omega,
                0.0,
                A,
                where=(omega >= w1) & (omega <= w2),
                alpha=0.12,
            )

    plt.xlabel("omega [rad/s]")
    plt.ylabel("A(omega)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def step_metrics(v_t):
    """
    Returns (v0_minus, v0_plus, v_inf) from symbolic v(t).
    """
    t = sp.Symbol("t")

    v0_minus = sp.limit(v_t, t, 0, dir="-")
    v0_plus = sp.limit(v_t, t, 0, dir="+")
    v_inf = sp.limit(v_t, t, sp.oo)

    return sp.simplify(v0_minus), sp.simplify(v0_plus), sp.simplify(v_inf)
