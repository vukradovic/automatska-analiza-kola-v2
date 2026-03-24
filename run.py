import argparse
from pathlib import Path
import sympy as sp

from engine.spice_parser import parse_spice
from engine.analysis import (
    transfer_function,
    time_response_from_transfer,
    numeric_step_response_from_transfer,
    plot_time_response,
    plot_numeric_time_response,
    bode_from_transfer,
    plot_bode,
    step_metrics,
    system_order,
    estimate_tau_from_transfer,
    zeros_poles_symbolic,
    amplitude_response_symbolic,
    symbolic_3db_band,
    amplitude_curve_from_transfer,
    amplitude_markers_numeric,
    plot_amplitude_characteristic,
    evaluate_measure,
    inverse_laplace_with_delay,
)


def parse_args():
    p = argparse.ArgumentParser(description="SPICE-like symbolic/numeric circuit analysis")
    p.add_argument("--netlist", default="examples/opamp.cir", help="Path to .cir netlist")
    p.add_argument("--out-node", default="out", help="Output node name")
    p.add_argument("--in-node", default="in", help="Input node name")
    p.add_argument(
        "--mode",
        choices=["symbolic", "numeric", "auto"],
        default="auto",
        help="Time-domain mode",
    )
    p.add_argument("--t-max", type=float, default=None, help="Max time for plot [s]")
    p.add_argument("--t-pre", type=float, default=0.2, help="Pre-step window [s]")
    p.add_argument("--n", type=int, default=1200, help="Number of samples")
    p.add_argument("--bode", action="store_true", help="Plot Bode magnitude/phase")
    p.add_argument("--w-min", type=float, default=1e-1, help="Bode min omega [rad/s]")
    p.add_argument("--w-max", type=float, default=1e5, help="Bode max omega [rad/s]")
    p.add_argument("--w-points", type=int, default=500, help="Bode points")
    p.add_argument("--amp-plot", action="store_true", help="Plot amplitude characteristic A(omega).")
    p.add_argument("--amp-w-max", type=float, default=10.0, help="Amplitude plot max omega [rad/s]")
    p.add_argument("--amp-points", type=int, default=800, help="Amplitude plot points")
    p.add_argument(
        "--tf-only",
        action="store_true",
        help="Compute and print transfer function data only (skip time response/plots).",
    )
    p.add_argument(
        "--symbolic-freq",
        action="store_true",
        help="Compute symbolic H(jomega), A(omega), zeros and poles.",
    )
    p.add_argument(
        "--symbolic-3db",
        action="store_true",
        help="Also solve symbolic 3dB points (can be very slow for complex expressions).",
    )
    p.add_argument(
        "--no-auto-plot-params",
        action="store_true",
        help="Disable automatic numeric defaults for symbolic parameters in numeric plot modes.",
    )
    p.add_argument(
        "--measure",
        action="append",
        default=[],
        help="Additional symbolic measure. Supported: V(node), V(a,b), I(element). Can be repeated.",
    )
    p.add_argument(
        "--measure-time",
        action="store_true",
        help="For each --measure, also compute inverse Laplace to time domain (can be slow).",
    )
    p.add_argument(
        "--measure-plot",
        action="append",
        default=[],
        help="Plot measured quantity in time domain. Use same syntax as --measure and repeat as needed.",
    )
    return p.parse_args()


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


def _compact_transfer_for_display(H):
    """
    Try to produce a shorter equivalent symbolic form of H(s) for printing.
    Heuristic: cancel/together and, if Om exists, normalize with x=s/Om.
    """
    s = sp.Symbol("s")
    H0 = sp.simplify(sp.cancel(sp.together(H)))
    best = H0
    best_len = len(str(best))

    Om = None
    for sym in H0.free_symbols:
        if str(sym) == "Om":
            Om = sym
            break

    if Om is not None:
        x = sp.Symbol("x")
        try:
            Hx = sp.simplify(sp.cancel(sp.together(H0.subs(s, Om * x))))
            if Om not in Hx.free_symbols:
                H_back = sp.simplify(sp.cancel(sp.together(Hx.subs(x, s / Om))))
                cand_len = len(str(H_back))
                if cand_len < best_len:
                    best = H_back
                    best_len = cand_len
        except Exception:
            pass

    return best


def _compact_expr_for_display(expr):
    """Compact rational expression print helper."""
    try:
        return sp.simplify(sp.cancel(sp.together(expr), extension=True))
    except Exception:
        return sp.simplify(expr)


def _normalized_transfer_display(H):
    """
    Return normalized transfer with x=s/Om when possible:
      H_norm(x) = H(s=Om*x)
    Useful for reading high-order symbolic expressions.
    """
    s = sp.Symbol("s")
    Om = None
    for sym in H.free_symbols:
        if str(sym) == "Om":
            Om = sym
            break
    if Om is None:
        return None, None

    x = sp.Symbol("x", real=True)
    try:
        Hx = sp.simplify(sp.cancel(sp.together(H.subs(s, Om * x))))
        return x, Hx
    except Exception:
        return None, None


def main():
    args = parse_args()

    print("Parsing netlist...", flush=True)
    netlist_path = Path(args.netlist)
    if not netlist_path.exists():
        fallback = Path("examples") / args.netlist
        if fallback.exists():
            netlist_path = fallback
        else:
            raise FileNotFoundError(
                f"Netlist not found: '{args.netlist}'. Tried '{netlist_path}' and '{fallback}'."
            )

    text = netlist_path.read_text(encoding="utf-8")
    ckt = parse_spice(text)

    print("Building and solving symbolic MNA (this can take time)...", flush=True)
    H, Vout, Vin = transfer_function(ckt, out_node=args.out_node, in_node=args.in_node)
    print("MNA solved.", flush=True)

    print("Vin(s)  =", Vin)
    print("Vout(s) =", Vout)
    Vin_compact = _compact_expr_for_display(Vin)
    Vout_compact = _compact_expr_for_display(Vout)
    if str(Vin_compact) != str(Vin):
        print("Vin(s) compact =", Vin_compact)
    if str(Vout_compact) != str(Vout):
        print("Vout(s) compact =", Vout_compact)
    print("H(s)    =", H)
    H_compact = _compact_transfer_for_display(H)
    if str(H_compact) != str(H):
        print("H(s) compact =", H_compact)
    x, Hx = _normalized_transfer_display(H)
    if x is not None and Hx is not None:
        print(f"H_norm({x}) [s=Om*{x}] =", Hx)

    order = system_order(H)
    print("system order =", order)

    if args.measure:
        s = sp.Symbol("s")
        t = sp.Symbol("t")
        mt_map = {}
        for meas in args.measure:
            try:
                ms = evaluate_measure(ckt, meas)
                msc = _compact_expr_for_display(ms)
                print(f"{meas}(s) =", ms)
                if str(msc) != str(ms):
                    print(f"{meas}(s) compact =", msc)
                if args.measure_time or args.measure_plot:
                    mt = sp.simplify(inverse_laplace_with_delay(msc, s=s, t=t))
                    mt_alt = sp.simplify(sp.expand(mt))
                    mt_best = mt if len(str(mt)) <= len(str(mt_alt)) else mt_alt
                    mt_map[meas] = mt_best
                    if args.measure_time:
                        print(f"{meas}(t) =", mt_best)
            except Exception as exc:
                print(f"measure '{meas}' skipped: {exc}")

        if args.measure_plot:
            for meas in args.measure_plot:
                if meas not in mt_map:
                    try:
                        ms = evaluate_measure(ckt, meas)
                        msc = _compact_expr_for_display(ms)
                        mt_map[meas] = sp.simplify(inverse_laplace_with_delay(msc, s=s, t=t))
                    except Exception as exc:
                        print(f"measure plot '{meas}' skipped: {exc}")
                        continue

                mplot = mt_map[meas]
                if not args.no_auto_plot_params:
                    subs = _default_numeric_substitutions(mplot)
                    # Keep time variable symbolic
                    subs = {k: v for k, v in subs.items() if str(k) != "t"}
                    if subs:
                        print(f"measure plot defaults applied for {meas}: {subs}")
                        mplot = sp.simplify(mplot.subs(subs))

                t_max_plot = args.t_max if args.t_max is not None else 6.0
                plot_time_response(
                    mplot,
                    t_max=t_max_plot,
                    t_pre=args.t_pre,
                    n=args.n,
                    title=f"{meas}(t)",
                    tau=None,
                )

    if args.symbolic_freq:
        # Assume passive circuit parameters are real/positive for cleaner symbolic
        # frequency expressions (avoids re()/im() explosion in solutions).
        s_sym = sp.Symbol("s")
        H_freq = H
        repl = {}
        for sym in H.free_symbols:
            if sym == s_sym:
                continue
            repl[sym] = sp.Symbol(str(sym), real=True, positive=True)
        if repl:
            H_freq = sp.simplify(H.subs(repl))

        try:
            zeros, poles = zeros_poles_symbolic(H_freq)
            print("zeros =", zeros)
            print("poles =", poles)
        except Exception as exc:
            print(f"symbolic zeros/poles skipped: {exc}")

        try:
            omega, Hjw, A_w, A2_w = amplitude_response_symbolic(H_freq)
            print(f"H(j{omega}) =", Hjw)
            print(f"A({omega}) =", A_w)

            if args.symbolic_3db:
                print("Solving symbolic 3dB points (may be slow)...", flush=True)
                band = symbolic_3db_band(H_freq, omega_symbol=omega)
                if band["omega_peak"] is not None:
                    print(f"omega_peak = {band['omega_peak']}")
                    print(f"A_peak^2 = {band['A2_peak']}")
                    print(f"omega_3dB = {band['omega_3db']}")
                    if band["omega_3db_low"] is not None:
                        print(f"omega_3dB_low = {band['omega_3db_low']}")
                    if band["omega_3db_high"] is not None:
                        print(f"omega_3dB_high = {band['omega_3db_high']}")
                    if band["bandwidth_3db"] is not None:
                        print(f"B_3dB = {band['bandwidth_3db']}")
                else:
                    print("symbolic 3dB: peak not uniquely identified.")
            else:
                print("symbolic 3dB skipped (use --symbolic-3db).")
        except Exception as exc:
            print(f"symbolic frequency analysis skipped: {exc}")

    tau = estimate_tau_from_transfer(H)
    if tau is not None:
        print("estimated tau =", tau)

    if args.tf_only:
        print("tf-only mode: skipping time response.")
    else:
        t_max = args.t_max
        if t_max is None:
            t_max = 10.0 * tau if tau is not None else 1.0

        mode = args.mode
        if mode == "auto":
            mode = "numeric"

        H_num = H
        Vin_num = Vin

        if mode == "numeric":
            try:
                ts, ys = numeric_step_response_from_transfer(H_num, Vin_num, t_max=t_max, t_pre=args.t_pre, n=args.n)
                print("time mode = numeric")
                plot_numeric_time_response(ts, ys, title="Step response: vout(t)", tau=tau)
            except Exception as exc:
                auto_subs_used = False
                if not args.no_auto_plot_params:
                    subs = _default_numeric_substitutions(H, Vin)
                    if subs:
                        H_num = sp.simplify(H.subs(subs))
                        Vin_num = sp.simplify(Vin.subs(subs))
                        print(f"numeric plot defaults applied: {subs}")
                        try:
                            ts, ys = numeric_step_response_from_transfer(
                                H_num, Vin_num, t_max=t_max, t_pre=args.t_pre, n=args.n
                            )
                            auto_subs_used = True
                            print("time mode = numeric (with default substitutions)")
                            plot_numeric_time_response(ts, ys, title="Step response: vout(t)", tau=tau)
                        except Exception as exc2:
                            print(f"numeric mode still unavailable ({exc2}); falling back to symbolic")

                if not auto_subs_used:
                    print(f"numeric mode unavailable ({exc}); falling back to symbolic")
                    vout_t = time_response_from_transfer(H, Vin)
                    print("vout(t) =", vout_t)
                    v0m, v0p, vinf = step_metrics(vout_t)
                    print("v(0-) =", v0m)
                    print("v(0+) =", v0p)
                    print("v(inf) =", vinf)
                    plot_time_response(
                        vout_t,
                        t_max=t_max,
                        t_pre=args.t_pre,
                        n=args.n,
                        title="Step response: vout(t)",
                        tau=tau,
                    )
        else:
            vout_t = time_response_from_transfer(H, Vin)
            print("time mode = symbolic")
            print("vout(t) =", vout_t)

            v0m, v0p, vinf = step_metrics(vout_t)
            print("v(0-) =", v0m)
            print("v(0+) =", v0p)
            print("v(inf) =", vinf)

            plot_time_response(vout_t, t_max=t_max, t_pre=args.t_pre, n=args.n, title="Step response: vout(t)", tau=tau)

    if args.bode:
        try:
            H_bode = H
            if not args.no_auto_plot_params:
                subs = _default_numeric_substitutions(H)
                if subs:
                    H_bode = sp.simplify(H.subs(subs))
            w, mag_db, phase_deg = bode_from_transfer(H_bode, w_min=args.w_min, w_max=args.w_max, n=args.w_points)
            plot_bode(w, mag_db, phase_deg, title="Bode from H(s)")
        except Exception as exc:
            print(f"Bode skipped: {exc}")

    if args.amp_plot:
        try:
            H_amp = H
            if not args.no_auto_plot_params:
                subs = _default_numeric_substitutions(H)
                if subs:
                    H_amp = sp.simplify(H.subs(subs))
                    print(f"amplitude plot defaults applied: {subs}")

            omega, A = amplitude_curve_from_transfer(H_amp, w_max=args.amp_w_max, n=args.amp_points)
            markers = amplitude_markers_numeric(omega, A)

            print("A_peak =", markers["A_peak"])
            print("omega_peak_num =", markers["omega_peak"])
            if markers["omega_3db_low"] is not None:
                print("omega_3dB_low_num =", markers["omega_3db_low"])
            if markers["omega_3db_high"] is not None:
                print("omega_3dB_high_num =", markers["omega_3db_high"])
            if markers["omega_3db_low"] is not None and markers["omega_3db_high"] is not None:
                print(
                    "B_3dB_num =",
                    markers["omega_3db_high"] - markers["omega_3db_low"],
                )

            plot_amplitude_characteristic(omega, A, markers=markers)
        except Exception as exc:
            print(f"Amplitude plot skipped: {exc}")


if __name__ == "__main__":
    main()
