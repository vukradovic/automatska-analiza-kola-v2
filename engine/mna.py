import sympy as sp
from .circuit import Circuit


def _node_index_map(circuit: Circuit):
    # MNA: ne uvodimo jednačinu za ground '0'
    nodes = sorted(n for n in circuit.nodes if n != "0")
    return {n: i for i, n in enumerate(nodes)}, nodes


def _to_exact(expr):
    """
    Pretvori Float u tačnu vrednost (Integer/Rational) kad god može.
    Ovo dramatično stabilizuje rešavanje kod velikih pojačanja (npr. 1e9).
    """
    e = sp.sympify(expr)

    # Ako je Float (npr 1000000000.0), pokušaj da ga pretvoriš u Integer ako je celobrojan
    if e.is_Float:
        # ako je skoro ceo broj
        nearest = int(e)
        if abs(float(e) - nearest) < 1e-12:
            return sp.Integer(nearest)
        # inače pretvori u racionalno iz stringa (tačno koliko SymPy može)
        return sp.Rational(str(e))

    return e


def _stamp_admittance(G, a, b, y):
    # a,b su indeksi u nodalnom vektoru (bez ground), mogu biti None za ground
    if a is not None:
        G[a, a] += y
    if b is not None:
        G[b, b] += y
    if a is not None and b is not None:
        G[a, b] -= y
        G[b, a] -= y


def build_mna_s_domain(circuit: Circuit):
    """
    Vraća (A, z, x_symbols, node_list)
    gde je A*x = z u s-domenu, x = [v_nodes..., i_Vsrc..., i_Esrc..., i_Osrc..., i_T1..., i_T2...]
    """
    s = sp.Symbol("s")

    node_map, node_list = _node_index_map(circuit)
    n = len(node_list)

    vsrcs = [e for e in circuit.elements if e.kind == "V"]
    evcvs = [e for e in circuit.elements if e.kind == "E"]
    iopamps = [e for e in circuit.elements if e.kind == "O"]
    tlines = [e for e in circuit.elements if e.kind == "T"]

    m = len(vsrcs) + len(evcvs) + len(iopamps) + 2 * len(tlines)

    A = sp.zeros(n + m, n + m)
    z = sp.zeros(n + m, 1)

    # --- Pasivni elementi: R, C, L (u s-domenu kao admitanse)
    for e in circuit.elements:
        if e.kind not in {"R", "C", "L"}:
            continue

        n1 = None if e.n1 == "0" else node_map[e.n1]
        n2 = None if e.n2 == "0" else node_map[e.n2]

        val = _to_exact(e.value)

        if e.kind == "R":
            y = 1 / val
        elif e.kind == "C":
            y = s * val
        else:  # "L"
            y = 1 / (s * val)

        _stamp_admittance(A, n1, n2, y)

    # --- Nezavisni naponski izvori (V): dodaj dodatne nepoznate struje
    # MNA stamping: poveži izvor između n+ i n- sa current unknown i jednačinom napona
    def _stamp_vsource(rowcol_index, nplus, nminus, Vval):
        # KCL deo: +I u nplus, -I u nminus
        if nplus is not None:
            A[nplus, rowcol_index] += 1
        if nminus is not None:
            A[nminus, rowcol_index] -= 1

        # Naponska jednačina: V(n+) - V(n-) = Vval
        if nplus is not None:
            A[rowcol_index, nplus] += 1
        if nminus is not None:
            A[rowcol_index, nminus] -= 1

        z[rowcol_index, 0] += Vval

    # indeksi dodatnih promenljivih počinju od n
    k = 0
    for vs in vsrcs:
        idx = n + k
        nplus = None if vs.n1 == "0" else node_map[vs.n1]
        nminus = None if vs.n2 == "0" else node_map[vs.n2]

        # DC step u t=0: V(s) = Vdc/s
        Vdc = _to_exact(vs.value)
        Vval = Vdc / s
        _stamp_vsource(idx, nplus, nminus, Vval)
        k += 1

    # --- VCVS (E): tretiramo kao "naponski izvor" čiji napon zavisi od (Vcp-Vcn)
    # Ename nout nref ncp ncn gain
    # Jednačina: V(nout)-V(nref) - gain*(V(ncp)-V(ncn)) = 0
    for ev in evcvs:
        idx = n + k

        nout = None if ev.n1 == "0" else node_map[ev.n1]
        nref = None if ev.n2 == "0" else node_map[ev.n2]

        # kontrolni čvorovi su u extra: "ncp ncn"
        ncp_name, ncn_name = ev.extra.split()[:2]
        ncp = None if ncp_name == "0" else node_map[ncp_name]
        ncn = None if ncn_name == "0" else node_map[ncn_name]

        gain = _to_exact(ev.value)

        # KCL deo (kao kod naponskog izvora): struja kroz E ulazi u čvorove nout/nref
        if nout is not None:
            A[nout, idx] += 1
            A[idx, nout] += 1  # doprinos u naponskoj jednačini
        if nref is not None:
            A[nref, idx] -= 1
            A[idx, nref] -= 1  # doprinos u naponskoj jednačini

        # kontrolni deo u naponskoj jednačini:
        # Vout - Vref - gain*(Vcp - Vcn) = 0
        if ncp is not None:
            A[idx, ncp] += -gain
        if ncn is not None:
            A[idx, ncn] += +gain

        k += 1

    # --- Ideal op-amp (O): enforce V+ - V- = 0 directly (no finite gain parameter)
    # Oname nout nref nplus nminus
    for op in iopamps:
        idx = n + k

        nout = None if op.n1 == "0" else node_map[op.n1]
        nref = None if op.n2 == "0" else node_map[op.n2]

        nplus_name, nminus_name = op.extra.split()[:2]
        nplus = None if nplus_name == "0" else node_map[nplus_name]
        nminus = None if nminus_name == "0" else node_map[nminus_name]

        # Output branch current unknown contributes to KCL at output nodes.
        if nout is not None:
            A[nout, idx] += 1
        if nref is not None:
            A[nref, idx] -= 1

        # Ideal op-amp constraint: V+ - V- = 0
        if nplus is not None:
            A[idx, nplus] += 1
        if nminus is not None:
            A[idx, nminus] -= 1

        k += 1

    # --- Ideal transmission line (T): two-port delay equations in s-domain
    # Tname n1 n2 n3 n4 Zc tau
    # Equations:
    #   (V1-V2) - Zc*I1 - e^(-s*tau)*((V3-V4) + Zc*I2) = 0
    #   (V3-V4) - Zc*I2 - e^(-s*tau)*((V1-V2) + Zc*I1) = 0
    for tl in tlines:
        idx1 = n + k
        idx2 = n + k + 1

        p1p = None if tl.n1 == "0" else node_map[tl.n1]
        p1n = None if tl.n2 == "0" else node_map[tl.n2]

        n3_name, n4_name, tau_raw = tl.extra.split()[:3]
        p2p = None if n3_name == "0" else node_map[n3_name]
        p2n = None if n4_name == "0" else node_map[n4_name]

        zc = _to_exact(tl.value)
        tau = _to_exact(tau_raw)
        d = sp.exp(-s * tau)

        # KCL contributions from port currents
        if p1p is not None:
            A[p1p, idx1] += 1
        if p1n is not None:
            A[p1n, idx1] -= 1
        if p2p is not None:
            A[p2p, idx2] += 1
        if p2n is not None:
            A[p2n, idx2] -= 1

        # Row idx1: (V1-V2) - Zc*I1 - d*((V3-V4) + Zc*I2) = 0
        if p1p is not None:
            A[idx1, p1p] += 1
        if p1n is not None:
            A[idx1, p1n] -= 1
        if p2p is not None:
            A[idx1, p2p] += -d
        if p2n is not None:
            A[idx1, p2n] += +d
        A[idx1, idx1] += -zc
        A[idx1, idx2] += -d * zc

        # Row idx2: (V3-V4) - Zc*I2 - d*((V1-V2) + Zc*I1) = 0
        if p2p is not None:
            A[idx2, p2p] += 1
        if p2n is not None:
            A[idx2, p2n] -= 1
        if p1p is not None:
            A[idx2, p1p] += -d
        if p1n is not None:
            A[idx2, p1n] += +d
        A[idx2, idx2] += -zc
        A[idx2, idx1] += -d * zc

        k += 2

    # simboli nepoznatih (opciono)
    v_syms = [sp.Symbol(f"V_{nname}") for nname in node_list]
    i_syms = [sp.Symbol(f"I_{e.name}") for e in (vsrcs + evcvs + iopamps)]
    for tl in tlines:
        i_syms.append(sp.Symbol(f"I_{tl.name}_1"))
        i_syms.append(sp.Symbol(f"I_{tl.name}_2"))
    x_syms = v_syms + i_syms

    return A, z, x_syms, node_list


def solve_mna_s_domain(circuit: Circuit):
    A_mat, z, x_syms, node_list = build_mna_s_domain(circuit)

    x = A_mat.gauss_jordan_solve(z)[0]

    # Ako postoji simbol A (ili bilo koji simbol koji počinje sa 'A'),
    # tretiraj ga kao idealni gain i uzmi limit -> oo
    gain_syms = [sym for sym in x.free_symbols if str(sym).startswith("A")]
    for g in gain_syms:
        x = x.applyfunc(lambda expr: sp.limit(expr, g, sp.oo))

    V = {}
    for i, nname in enumerate(node_list):
        V[nname] = sp.simplify(x[i, 0])

    return V, x, x_syms
