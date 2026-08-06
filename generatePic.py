import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import subprocess
import json
import sys
import hashlib
import pickle
from pathlib import Path
from matplotlib import patheffects


def kagome_kondo_ising():
    """Draw kagome lattice (Co sublattice) with Tb localized spins and c-f hybridization."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Kagome lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])

    # Kagome basis: 3 sites per unit cell at midpoints of the triangular lattice bonds
    # Site A: (a1)/2, Site B: (a2)/2, Site C: (a1+a2)/2
    basis = [a1 / 2, a2 / 2, (a1 + a2) / 2]

    # Generate kagome sites for a 4x4 patch
    co_sites = []
    n_cells = 4
    for i in range(n_cells):
        for j in range(n_cells):
            origin = i * a1 + j * a2
            for b in basis:
                co_sites.append(origin + b)
    co_sites = np.array(co_sites)

    # Triangular lattice sites (Tb positions) — at the hexagonal centers
    # These are at the original triangular lattice points
    tb_sites = []
    for i in range(n_cells + 1):
        for j in range(n_cells + 1):
            tb_sites.append(i * a1 + j * a2)
    # Also add sites shifted by (2a1+a2)/3 and (a1+2a2)/3 for the hexagonal centers
    hex_centers = []
    for i in range(n_cells):
        for j in range(n_cells):
            # Up triangles center
            c1 = (i * a1 + j * a2 + (i + 1) * a1 + j * a2 + i * a1 + (j + 1) * a2) / 3
            c2 = ((i + 1) * a1 + j * a2 + (i + 1) * a1 + (j + 1) * a2 + i * a1 + (j + 1) * a2) / 3
            hex_centers.extend([c1, c2])
    hex_centers = np.array(hex_centers)

    # Plot Co kagome bonds (nearest-neighbor connections on kagome)
    bond_len = 0.5 + 0.01  # slightly generous threshold
    for i, si in enumerate(co_sites):
        for j, sj in enumerate(co_sites):
            if j > i:
                d = np.linalg.norm(si - sj)
                if d < bond_len:
                    ax.plot([si[0], sj[0]], [si[1], sj[1]], 'b-', lw=1.2, alpha=0.5)

    # Plot Co sites (kagome lattice)
    ax.scatter(co_sites[:, 0], co_sites[:, 1], s=80, c='steelblue', zorder=5,
               edgecolors='navy', linewidths=0.8, label=r'Co (kagome)')

    # Plot Tb localized spin sites at hexagonal centers
    ax.scatter(hex_centers[:, 0], hex_centers[:, 1], s=180, c='orangered', zorder=6,
               edgecolors='darkred', linewidths=0.8, label=r'Tb (localized)', marker='o')

    # Draw Ising spin arrows on Tb sites
    arrow_scale = 0.15
    for site in hex_centers:
        # Alternate up/down to show ferrimagnetic/antiferromagnetic ordering
        direction = 1 if (hash(tuple(site.round(3))) % 2 == 0) else -1
        ax.annotate('', xy=(site[0], site[1] + direction * arrow_scale),
                     xytext=(site[0], site[1] - direction * arrow_scale),
                     arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

    # Draw c-f hybridization (dashed lines between nearest Co-Tb pairs)
    for tb in hex_centers:
        for co in co_sites:
            d = np.linalg.norm(tb - co)
            if d < 0.35:
                ax.plot([tb[0], co[0]], [tb[1], co[1]], '--', color='gray',
                        lw=0.7, alpha=0.4, zorder=3)

    ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(-0.3, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    co_patch = mpatches.Patch(color='steelblue', label='Co (kagome sublattice)')
    tb_patch = mpatches.Patch(color='orangered', label='Tb (localized spins)')
    hybrid_line = plt.Line2D([0], [0], linestyle='--', color='gray', label='c-f hybridization')
    ax.legend(handles=[co_patch, tb_patch, hybrid_line], loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='gray')

    # Title
    ax.set_title('Kondo-Ising Model on Kagome Lattice\n'
                 r'$H = -t\sum_{\langle ij\rangle\sigma} c_{i\sigma}^\dagger c_{j\sigma}'
                 r' + J_K\sum_i S_i^z s_i^z - J\sum_{\langle ij\rangle} S_i^z S_j^z$',
                 fontsize=13, pad=15)

    plt.tight_layout()
    outpath = Path('pics/kagome_kondo_ising.png')
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def kagome_kondo_ising_v2():
    """Isometric bilayer Kondo-Ising schematic on kagome lattice, side-view perspective.

    Like FIG.1 of the AM paper: isometric 3D view with Co kagome layer on top
    and Tb localized spin layer below, connected by vertical Kondo coupling J_K.
    The kagome lattice is drawn in pseudo-3D by applying an oblique projection.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Oblique projection: map (x, y, z) -> (x + 0.35*z, y + 0.35*z)
    # Layer separation in the z-direction
    layer_sep = 2.8  # larger gap: Kondo-Ising model, not real crystal

    def proj(x, y, z):
        return x + 0.3 * z, y + 0.35 * z

    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])
    basis = [a1 / 2, a2 / 2, (a1 + a2) / 2]
    n_cells = 3

    # Generate Co kagome sites (top layer, z=1)
    co_sites_2d = []
    for i in range(n_cells):
        for j in range(n_cells):
            origin = i * a1 + j * a2
            for b in basis:
                co_sites_2d.append(origin + b)
    co_sites_2d = np.array(co_sites_2d)
    co_sites = np.array([proj(s[0], s[1], layer_sep) for s in co_sites_2d])

    # Generate Tb triangular lattice sites (bottom layer, z=0)
    tb_sites_2d = []
    for i in range(n_cells + 1):
        for j in range(n_cells + 1):
            tb_sites_2d.append(i * a1 + j * a2)
    tb_sites_2d = np.array(tb_sites_2d)
    tb_sites = np.array([proj(s[0], s[1], 0) for s in tb_sites_2d])

    # --- Draw Tb layer first (bottom, behind) ---
    # Ising bonds between nearest Tb sites
    tb_bond_len = 1.0 + 0.05
    for i_s, si in enumerate(tb_sites_2d):
        for j_s, sj in enumerate(tb_sites_2d):
            if j_s > i_s:
                d = np.linalg.norm(si - sj)
                if d < tb_bond_len:
                    p1 = proj(si[0], si[1], 0)
                    p2 = proj(sj[0], sj[1], 0)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='#b2182b',
                            lw=1.0, alpha=0.3, zorder=1)

    # Tb sites with AFM arrows (blue up, gold down)
    arrow_len = 0.22
    for idx, site in enumerate(tb_sites_2d):
        px, py = proj(site[0], site[1], 0)
        i_idx = int(round(site[0]))
        j_idx = int(round((2 * site[1] - site[0]) / np.sqrt(3)))
        direction = 1 if (i_idx + j_idx) % 2 == 0 else -1
        color = '#2166ac' if direction == 1 else '#d4a017'
        ax.annotate('', xy=(px, py + direction * arrow_len),
                     xytext=(px, py - direction * arrow_len * 0.3),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
                     zorder=6)

    ax.scatter(tb_sites[:, 0], tb_sites[:, 1], s=140, c='#fddbc7', zorder=5,
               edgecolors='#b2182b', linewidths=1.0, marker='o')

    # --- Kondo coupling: vertical dashed lines from Tb to nearest Co neighbors ---
    # Like v1: connect each Tb to all Co sites within a short cutoff (surrounding kagome sites)
    jk_cutoff = 0.6
    for idx, ts in enumerate(tb_sites_2d):
        tp = proj(ts[0], ts[1], 0)
        for cs in co_sites_2d:
            d = np.linalg.norm(cs - ts)
            if d < jk_cutoff:
                cp = proj(cs[0], cs[1], layer_sep)
                ax.plot([tp[0], cp[0]], [tp[1], cp[1]], '--', color='#666666',
                        lw=0.8, alpha=0.5, zorder=3)

    # Label one Kondo coupling line
    label_idx = len(tb_sites_2d) // 2 + n_cells // 2 + 1
    if label_idx < len(tb_sites_2d):
        ts = tb_sites_2d[label_idx]
        tp = proj(ts[0], ts[1], 0)
        # Find one nearest Co for label placement
        min_d = float('inf')
        closest_co = None
        for cs in co_sites_2d:
            d = np.linalg.norm(cs - ts)
            if d < min_d:
                min_d = d
                closest_co = cs
        if min_d < jk_cutoff:
            cp = proj(closest_co[0], closest_co[1], layer_sep)
            mid = ((tp[0] + cp[0]) / 2 + 0.15, (tp[1] + cp[1]) / 2)
            ax.annotate(r'$J_K$', xy=mid, fontsize=12, color='#444444', fontweight='bold')

    # --- Draw Co layer (top, in front) ---
    bond_len = 0.5 + 0.02
    for i_s, si in enumerate(co_sites_2d):
        for j_s, sj in enumerate(co_sites_2d):
            if j_s > i_s:
                d = np.linalg.norm(si - sj)
                if d < bond_len:
                    p1 = proj(si[0], si[1], layer_sep)
                    p2 = proj(sj[0], sj[1], layer_sep)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='#2166ac',
                            lw=1.8, alpha=0.7, zorder=8)

    # Co sublattice coloring: 3 colors for 3 kagome basis sites
    basis_colors = ['#2166ac', '#4393c3', '#92c5de']  # dark, medium, light blue
    colors_co = [basis_colors[b] for i in range(n_cells) for j in range(n_cells) for b in range(3)]
    ax.scatter(co_sites[:, 0], co_sites[:, 1], s=100, c=colors_co, zorder=10,
               edgecolors='#053061', linewidths=0.8)

    # Hopping label on one bond
    ax.annotate(r'$t$', xy=(proj(0.25, 0.0, layer_sep)[0],
                             proj(0.25, 0.0, layer_sep)[1] + 0.12),
                fontsize=13, color='#2166ac', fontweight='bold', zorder=11)

    # Ising label
    ax.annotate(r'$J$', xy=(proj(0.5, 0.0, 0)[0],
                             proj(0.5, 0.0, 0)[1] - 0.25),
                fontsize=13, color='#b2182b', fontweight='bold', zorder=11)

    # --- Layer labels ---
    co_center = np.mean(co_sites, axis=0)
    tb_center = np.mean(tb_sites, axis=0)
    ax.annotate('Co kagome (itinerant)', xy=(co_center[0] + 1.5, co_center[1] + 0.3),
                fontsize=11, color='#2166ac', fontweight='bold',
                ha='center', zorder=12)
    ax.annotate('Tb (localized spins)', xy=(tb_center[0] + 1.5, tb_center[1] - 0.3),
                fontsize=11, color='#b2182b', fontweight='bold',
                ha='center', zorder=12)

    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-0.8, 4.3)  # reduced from 5.0 to trim top whitespace
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Hamiltonian at bottom ---
    ax.text(0.5, -0.06,
            r'$H = -t\!\sum_{\langle ij\rangle\sigma} c_{i\sigma}^\dagger c_{j\sigma}'
            r'\;+\; J_K\!\sum_i S_i^z s_i^z'
            r'\;-\; J\!\sum_{\langle ij\rangle} S_i^z S_j^z$',
            transform=ax.transAxes, ha='center', fontsize=14,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7', edgecolor='#999999'))

    # --- Legend ---
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166ac',
                    markeredgecolor='#053061', markersize=9, label='Co (sublattice A)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4393c3',
                    markeredgecolor='#053061', markersize=9, label='Co (sublattice B)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#92c5de',
                    markeredgecolor='#053061', markersize=9, label='Co (sublattice C)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#fddbc7',
                    markeredgecolor='#b2182b', markersize=11, label='Tb (localized)'),
        plt.Line2D([0], [0], linestyle='-', color='#2166ac', lw=2, label=r'Hopping $t$'),
        plt.Line2D([0], [0], linestyle='-', color='#b2182b', lw=1.5, alpha=0.5,
                    label=r'Ising $J$'),
        plt.Line2D([0], [0], linestyle='--', color='#666666', lw=1.2,
                    label=r'Kondo $J_K$'),
        plt.Line2D([0], [0], marker='$↑$', color='#2166ac', markersize=12,
                    linestyle='none', label='Spin up'),
        plt.Line2D([0], [0], marker='$↓$', color='#d4a017', markersize=12,
                    linestyle='none', label='Spin down'),
    ]
    ax.legend(handles=legend_elements, loc='center left', fontsize=9,
              framealpha=0.95, edgecolor='#999999', bbox_to_anchor=(-0.05, 0.5))
    # bbox_to_anchor=(-0.05, 0.5)  # for fine-tuning legend position

    plt.tight_layout()
    outpath = Path('pics/kagome_kondo_ising_v2.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


# ---------------------------------------------------------------------------
# MC data plotting infrastructure
# ---------------------------------------------------------------------------

BINARY_DIR = Path(os.environ.get('MC_BINARY_DIR', Path(__file__).parent / 'examples' / 'build'))


def parse_mc_output(stdout: str) -> dict:
    """Parse C++ MC output: JSON header + tab-separated data + optional time series."""
    result = {'params': {}, 'observables': {}, 'time_series': {}}
    lines = stdout.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('# {'):
            result['params'] = json.loads(line[2:])
        elif line.startswith('# algorithm:'):
            result['algorithm'] = line.split(':', 1)[1].strip()
        elif line.startswith('# seed:'):
            result['seed'] = int(line.split(':', 1)[1].strip())
        elif line.startswith('# time_series_begin:'):
            ts_name = line.split(':', 1)[1].strip()
            ts_values = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('# time_series_end'):
                try:
                    ts_values.append(float(lines[i].strip()))
                except ValueError:
                    pass
                i += 1
            result['time_series'][ts_name] = np.array(ts_values)
        elif line.startswith('#') or not line or line.startswith('observable'):
            pass
        else:
            parts = line.split('\t')
            if len(parts) >= 2:
                result['observables'][parts[0]] = {
                    'mean': float(parts[1]),
                    'variance': float(parts[2]) if len(parts) > 2 else 0.0,
                    'mean2': float(parts[3]) if len(parts) > 3 else 0.0,
                }
        i += 1
    return result


def run_mc_binary(algorithm: str, L: int, J: float, T: float,
                  sweeps: int = 5000, therm: int = 2000,
                  Jp: float = 0.0, seed: int = 0,
                  all_up: bool = True, ts: bool = False,
                  auto_therm: bool = False, dim: int = 0,
                  overlap_step: int = 0, q: int = 2,
                  overlap_steps=None, cluster_only: bool = False) -> dict:
    """Run a C++ MC binary and return parsed results.

    q=2 (default) uses the Ising binaries (metropolis/swendsen_wang/wolff);
    q!=2 uses the q-state Potts binaries (potts_metropolis/...).  Only the
    Ising-family algorithms support q; Heisenberg binaries ignore it.

    overlap_steps: if given (a list of N_t), switch to MULTI-Nt mode — always
    use the Potts binary (potts_{algorithm}; the Ising binaries are frozen and
    lack --overlap-steps), pass --q and --overlap-steps n1,n2,…, and enable
    --ts so the per-Nt time series are captured.  Caller MUST pass T = potts_tc(q)
    (Potts convention), not overlap_tc(2).
    """
    multi_nt = overlap_steps is not None
    if multi_nt:
        # Always the Potts binary (Ising binaries are frozen, no --overlap-steps).
        binary = BINARY_DIR / f'potts_{algorithm}'
        if not binary.exists():
            raise FileNotFoundError(f'Binary not found: {binary}')
        cmd = [str(binary), '--L', str(L), '--J', str(J), '--T', str(T),
               '--sweeps', str(sweeps), '--therm', str(therm),
               '--q', str(q),
               '--overlap-steps', ','.join(str(int(n)) for n in overlap_steps),
               '--ts']
        if seed != 0:
            cmd += ['--seed', str(seed)]
        if all_up:
            cmd += ['--all-up']
        if auto_therm:
            cmd += ['--auto-therm']
        if cluster_only and algorithm == 'wolff':
            cmd += ['--cluster-only']   # skip config overlap (large-L memory)
    else:
        binary_name = algorithm if q == 2 else f'potts_{algorithm}'
        binary = BINARY_DIR / binary_name
        if not binary.exists():
            raise FileNotFoundError(f'Binary not found: {binary}')
        cmd = [str(binary), '--L', str(L), '--J', str(J), '--T', str(T),
               '--sweeps', str(sweeps), '--therm', str(therm)]
        if Jp != 0.0:
            cmd += ['--Jp', str(Jp)]
        if dim > 0:
            cmd += ['--dim', str(dim)]
        if seed != 0:
            cmd += ['--seed', str(seed)]
        if all_up:
            cmd += ['--all-up']
        if ts:
            cmd += ['--ts']
        if auto_therm:
            cmd += ['--auto-therm']
        if overlap_step > 0:
            cmd += ['--overlap-step', str(overlap_step)]
        if q != 2:
            cmd += ['--q', str(q)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=21600)
    if proc.returncode != 0:
        raise RuntimeError(f'{binary_name} failed: {proc.stderr}')
    result = parse_mc_output(proc.stdout)
    if proc.stderr:
        result['stderr'] = proc.stderr
    return result


def temperature_sweep(algorithm: str, T_list, L: int, J: float = 1.0,
                      sweeps: int = 5000, therm: int = 2000, **kwargs) -> list:
    """Run algorithm at each temperature, return list of result dicts."""
    results = []
    for T in T_list:
        r = run_mc_binary(algorithm, L, J, T, sweeps, therm, **kwargs)
        r['T'] = T
        results.append(r)
        m_val = (r["observables"].get("abs_magnetization", {})
                 or r["observables"].get("magnetization", {})).get("mean", "N/A")
        print(f'  {algorithm} T={T:.2f} |m|={m_val}')
    return results


def onsager_exact_M(T_arr, J=1.0):
    """Onsager exact spontaneous magnetization for 2D Ising (NN only)."""
    T_c = 2.0 / np.log(1.0 + np.sqrt(2.0))
    M = np.zeros_like(T_arr)
    mask = T_arr < T_c
    with np.errstate(divide='ignore', invalid='ignore'):
        sinh_val = np.sinh(2.0 * J / T_arr[mask])
        ratio = 1.0 / sinh_val**4
    valid = ratio < 1.0
    M[mask] = np.where(valid, (1.0 - ratio)**(1.0/8.0), 0.0)
    return M


def find_tc_numerical(T, M):
    """Estimate Tc from MC data by finding T where -dM/dT is maximal."""
    idx = np.argsort(T)
    T_sorted, M_sorted = T[idx], M[idx]
    T_fine = np.linspace(T_sorted[0], T_sorted[-1], 1000)
    M_fine = np.interp(T_fine, T_sorted, M_sorted)
    dM_dT = np.gradient(M_fine, T_fine)
    tc_idx = np.argmin(dM_dT)
    return T_fine[tc_idx]


T_SWEEP_LIST = np.array([0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8,
                         2.0, 2.1, 2.2, 2.25, 2.27, 2.29, 2.3, 2.35,
                         2.4, 2.5, 2.7, 3.0, 3.5, 4.0])

# Dense temperature list for overlap verification (V6), ~28 points with ~9 near Tc
OVERLAP_T_LIST = np.array([
    0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.7, 1.8, 1.9, 2.0, 2.05, 2.1, 2.15, 2.2,
    2.22, 2.24, 2.26, 2.27, 2.28, 2.29, 2.30, 2.32, 2.34,
    2.4, 2.5, 2.6, 2.7, 3.0, 3.5
])

# --- q-state Potts support (Lecture 2 generalization; q ∈ {2,3,4}) ---
ISING_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))           # 2D Ising Tc ≈ 2.269
POTTS3_TC = 1.0 / np.log(1.0 + np.sqrt(3.0))           # 2D 3-state Potts Tc ≈ 0.9950
POTTS4_TC = 1.0 / np.log(1.0 + np.sqrt(4.0))           # 2D 4-state Potts Tc ≈ 0.9102 (marginal)

# 2D q-state Potts critical exponents (q=2≡Ising; q=4 is marginal, same η as Ising
# but with logarithmic corrections). γ/ν, η = anomalous dimension.
_Q_GAMMA_NU = {2: 7.0 / 4.0, 3: 26.0 / 15.0, 4: 7.0 / 4.0}
_Q_ETA = {2: 1.0 / 4.0, 3: 4.0 / 15.0, 4: 1.0 / 4.0}


def overlap_tc(q: int = 2) -> float:
    """Exact critical temperature Tc = 1/ln(1+√q) for the 2D q-state Potts model."""
    if q == 2:
        return ISING_TC
    return 1.0 / np.log(1.0 + np.sqrt(float(q)))


def potts_tc(q: int = 2) -> float:
    """Exact Tc in the Potts *binary* convention (H = -J·δ_ij): Tc = 1/ln(1+√q).

    For q=2 this is 1.1346 — NOT the Ising value 2.269. The q=2 Potts binary
    (potts_wolff, FK bond prob 1-e^{-βJ}) sits at a different T than the Ising
    binary (wolff, 1-e^{-2βJ}) for the SAME physical transition (the q=2 Potts
    H = -J·δ maps to Ising with coupling J/2). Use potts_tc — not overlap_tc —
    whenever a run targets the Potts binary (all multi-Nt runs do).
    """
    return 1.0 / np.log(1.0 + np.sqrt(float(q)))


def overlap_t_list(q: int = 2) -> np.ndarray:
    """Temperature grid dense near Tc for overlap sweeps (q-scaled)."""
    if q == 2:
        return OVERLAP_T_LIST
    # scale the Ising grid so its structure (dense near Tc) maps to this q's Tc
    return OVERLAP_T_LIST * (overlap_tc(q) / ISING_TC)


def overlap_suffix(q: int = 2) -> str:
    """Filename suffix for q-state overlap figures ('' for Ising q=2)."""
    return '' if q == 2 else f'_q{q}'


def overlap_gamma_nu(q: int = 2) -> float:
    """Critical ratio γ/ν for 2D q-state Potts: 7/4 (Ising, q=4), 26/15 (q=3)."""
    return _Q_GAMMA_NU[q]


def overlap_indep_exp(q: int = 2) -> float:
    """Independent-cluster FSS exponent L^{-2η}: 0.5 (Ising, q=4), 8/15 (q=3).
    Mean ⟨R⟩ and variance share this exponent in the independent limit."""
    return 2.0 * _Q_ETA[q]


def overlap_naive_a_exp(q: int = 2) -> float:
    """Naive single-cluster inclusion a=χ/N ~ L^{γ/ν-2}: 1/4 (Ising, q=4), 4/15 (q=3).
    This is what the mean overlap ⟨R(2)⟩ is NOT (dynamical scaling)."""
    return 2.0 - overlap_gamma_nu(q)


def sw_var_nt1_exp(q: int = 2) -> float:
    """SW Var(U_1) analytic FSS exponent = γ/ν - 2 (the Nt=1 limit of the two-time result).

    Rigorous two-time FK (= Pilé et al., arXiv:2604.10254v2, Eq. A4):
        Var(U_n) = (q-1)/q^2 · ⟨|C_t ∩ C_{t+n}|⟩ / N.
    At Nt=1 consecutive-step clusters overlap, ⟨|C_t ∩ C_{t+1}|⟩ ≈ χ, so
        Var(U_1) ≈ (q-1)/q^2 · χ/N ~ L^{γ/ν-2}:  Ising/q=4 L^{-1/4}, q=3 L^{-4/15}.
    (χ/N is the Nt=1 approximation, not a one-time identity; numerics confirm L^{-1/4}.)"""
    return overlap_gamma_nu(q) - 2.0


def sw_lit_exp(q: int = 2) -> float:
    """SW Var(U_2) FSS literature exponent ψ^SW (Pilé et al., arXiv:2604.10254v2,
    Table I — FSS of Var(U_2) at Tc vs 1/L). q-aware: Ising 0.3458(9), q=3 0.318(1), q=4 0.288(4)."""
    return {2: 0.346, 3: 0.318, 4: 0.288}[q]


def plot_metropolis_mt():
    """Metropolis |m| vs T with Onsager exact overlay — verification plot."""
    print('Running Metropolis temperature sweep...')
    results = temperature_sweep('metropolis', T_SWEEP_LIST, L=16, J=1.0,
                                sweeps=10000, therm=5000)

    mc_T = np.array([r['T'] for r in results])
    mc_m = np.array([r['observables']['abs_magnetization']['mean'] for r in results])

    T_fine = np.linspace(0.0, 4.5, 300)
    M_exact = onsager_exact_M(T_fine, J=1.0)
    T_c_onsager = 2.0 / np.log(1.0 + np.sqrt(2.0))
    T_c_num = find_tc_numerical(mc_T, mc_m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_fine, M_exact, 'k-', lw=2, label='Onsager exact')
    ax.plot(mc_T, mc_m, 'ro', ms=6, label='Metropolis (L=16)', zorder=5)
    ax.axvline(T_c_onsager, color='gray', ls='--', lw=1, alpha=0.5)
    ax.axvline(T_c_num, color='red', ls=':', lw=1.5, alpha=0.8,
               label=rf'$T_c^{{\mathrm{{num}}}} = {T_c_num:.3f}$')
    ax.set_xlabel(r'$T$', fontsize=14)
    ax.set_ylabel(r'$\langle |m| \rangle$', fontsize=14)
    ax.set_title('Metropolis MC vs Onsager Exact Solution', fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xlim(0.0, 4.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Path('pics/metropolis_mt.png')
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_thermalization(T=2.5, sweeps=100000, therm=0):
    """Dual-start convergence: plot ⟨m⟩ from all-up and random initial conditions."""
    print('Running thermalization convergence check...')
    #T = 2.5
    #sweeps = 10000
    #therm = 0  # no fixed thermalization — watch the full trajectory

    r_up = run_mc_binary('metropolis', L=16, J=1.0, T=T, sweeps=sweeps, therm=therm,
                         all_up=True, ts=True, seed=42)
    r_rand = run_mc_binary('metropolis', L=16, J=1.0, T=T, sweeps=sweeps, therm=therm,
                           all_up=False, ts=True, seed=123)

    ts_up = r_up['time_series'].get('magnetization', np.array([]))
    ts_rand = r_rand['time_series'].get('magnetization', np.array([]))

    if len(ts_up) == 0 or len(ts_rand) == 0:
        print('Warning: no time series data. Check --ts flag.')
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts_up, 'b-', lw=0.8, alpha=0.7, label='All-up init')
    ax.plot(ts_rand, 'r-', lw=0.8, alpha=0.7, label='Random init')
    ax.axhline(y=0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Sweep', fontsize=14)
    ax.set_ylabel(r'$\langle m \rangle$', fontsize=14)
    ax.set_title(f'Thermalization Convergence (T={T}, L=16)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Path('pics/thermalization_convergence_2.5K.png')
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_sw_mt():
    """Swendsen-Wang |m| vs T with Onsager exact overlay — verification plot."""
    print('Running Swendsen-Wang temperature sweep...')
    results = temperature_sweep('swendsen_wang', T_SWEEP_LIST, L=16, J=1.0,
                                sweeps=5000, therm=2000)

    mc_T = np.array([r['T'] for r in results])
    mc_m = np.array([r['observables']['abs_magnetization']['mean'] for r in results])

    T_fine = np.linspace(0.0, 4.5, 300)
    M_exact = onsager_exact_M(T_fine, J=1.0)
    T_c_onsager = 2.0 / np.log(1.0 + np.sqrt(2.0))
    T_c_num = find_tc_numerical(mc_T, mc_m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_fine, M_exact, 'k-', lw=2, label='Onsager exact')
    ax.plot(mc_T, mc_m, 's', color='#2196F3', ms=6, label='Swendsen-Wang (L=16)', zorder=5)
    ax.axvline(T_c_onsager, color='gray', ls='--', lw=1, alpha=0.5)
    ax.axvline(T_c_num, color='#2196F3', ls=':', lw=1.5, alpha=0.8,
               label=rf'$T_c^{{\mathrm{{num}}}} = {T_c_num:.3f}$')
    ax.set_xlabel(r'$T$', fontsize=14)
    ax.set_ylabel(r'$\langle |m| \rangle$', fontsize=14)
    ax.set_title('Swendsen-Wang MC vs Onsager Exact Solution', fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xlim(0.0, 4.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Path('pics/sw_mt.png')
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_wolff_mt():
    """Wolff |m| vs T with Onsager exact overlay — verification plot."""
    print('Running Wolff temperature sweep...')
    results = temperature_sweep('wolff', T_SWEEP_LIST, L=16, J=1.0,
                                sweeps=5000, therm=2000)

    mc_T = np.array([r['T'] for r in results])
    mc_m = np.array([r['observables']['abs_magnetization']['mean'] for r in results])

    T_fine = np.linspace(0.0, 4.5, 300)
    M_exact = onsager_exact_M(T_fine, J=1.0)
    T_c_onsager = 2.0 / np.log(1.0 + np.sqrt(2.0))
    T_c_num = find_tc_numerical(mc_T, mc_m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_fine, M_exact, 'k-', lw=2, label='Onsager exact')
    ax.plot(mc_T, mc_m, '^', color='#4CAF50', ms=6, label='Wolff (L=16)', zorder=5)
    ax.axvline(T_c_onsager, color='gray', ls='--', lw=1, alpha=0.5)
    ax.axvline(T_c_num, color='#4CAF50', ls=':', lw=1.5, alpha=0.8,
               label=rf'$T_c^{{\mathrm{{num}}}} = {T_c_num:.3f}$')
    ax.set_xlabel(r'$T$', fontsize=14)
    ax.set_ylabel(r'$\langle |m| \rangle$', fontsize=14)
    ax.set_title('Wolff MC vs Onsager Exact Solution', fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xlim(0.0, 4.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Path('pics/wolff_mt.png')
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def autocorrelation_time(ts):
    """Integrated autocorrelation time from magnetization time series.

    τ_int = 1/2 + Σ_{k=1}^M ρ(k), truncated at first negative ρ(k).
    """
    n = len(ts)
    mean = np.mean(ts)
    var = np.var(ts)
    if var == 0:
        return 1.0
    ts_ctr = ts - mean
    max_lag = min(n // 2, 500)
    tau = 0.5
    for lag in range(1, max_lag):
        rho = np.mean(ts_ctr[:n - lag] * ts_ctr[lag:]) / var
        if rho < 0:
            break
        tau += rho
    return tau


def compute_tau_int(algorithm, L, T, sweeps, therm, seed=42):
    """Run binary with --ts and return integrated autocorrelation time of |m|."""
    r = run_mc_binary(algorithm, L=L, J=1.0, T=T, sweeps=sweeps, therm=therm,
                      all_up=False, ts=True, seed=seed)
    ts_m = r['time_series'].get('magnetization', np.array([]))
    ts_abs_m = np.abs(ts_m)
    if len(ts_abs_m) == 0:
        print(f'  Warning: no time series for {algorithm} L={L}')
        return float('nan')
    tau = autocorrelation_time(ts_abs_m)
    return tau


def plot_dynamic_exponent():
    """Compute dynamic critical exponent z for all three algorithms.

    Runs at Tc for multiple L, computes τ_int(L), fits τ ∝ L^z on log-log scale.
    """
    T_c = 2.0 / np.log(1.0 + np.sqrt(2.0))
    print(f'\n=== Dynamic Critical Exponent z (T_c = {T_c:.4f}) ===')

    # L values and sweep counts per algorithm
    L_metro = np.array([8, 16, 32])
    L_cluster = np.array([8, 16, 32, 64])
    sweeps_metro, therm_metro = 50000, 10000
    sweeps_cluster, therm_cluster = 20000, 5000

    algorithms = {
        'metropolis':  ('Metropolis', L_metro, sweeps_metro, therm_metro, 'o', 'red'),
        'swendsen_wang': ('Swendsen-Wang', L_cluster, sweeps_cluster, therm_cluster, 's', '#2196F3'),
        'wolff':         ('Wolff', L_cluster, sweeps_cluster, therm_cluster, '^', '#4CAF50'),
    }

    results = {}
    for alg, (label, L_vals, sweeps, therm, marker, color) in algorithms.items():
        print(f'\n{alg}:')
        tau_vals = []
        for L in L_vals:
            tau = compute_tau_int(alg, L, T_c, sweeps, therm)
            tau_vals.append(tau)
            print(f'  L={L:3d}  τ_int={tau:.2f}')
        tau_vals = np.array(tau_vals)
        log_L = np.log(L_vals)
        log_tau = np.log(tau_vals)
        # Linear fit: log(τ) = z · log(L) + c
        z, c = np.polyfit(log_L, log_tau, 1)
        print(f'  → z = {z:.3f}')
        results[alg] = (label, L_vals, tau_vals, z, c, marker, color)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for alg, (label, L_vals, tau_vals, z, c, marker, color) in results.items():
        L_fine = np.logspace(np.log10(L_vals[0] * 0.8), np.log10(L_vals[-1] * 1.2), 50)
        tau_fit = np.exp(c) * L_fine ** z
        ax.loglog(L_vals, tau_vals, marker, color=color, ms=8, zorder=5,
                  label=f'{label} ($z = {z:.2f}$)')
        ax.loglog(L_fine, tau_fit, '--', color=color, lw=1.2, alpha=0.6)

    ax.set_xlabel(r'$L$', fontsize=14)
    ax.set_ylabel(r'$\tau_{\mathrm{int}}$', fontsize=14)
    ax.set_title(r'Dynamic Critical Exponent: $\tau_{\mathrm{int}} \propto L^z$', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Reference lines for literature z values
    ax.text(0.05, 0.95, r'Literature: $z_{\mathrm{Metro}}\approx 2.17,\; z_{\mathrm{SW}}\approx 0.35,\; z_{\mathrm{Wolff}}\approx 0.25$',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    outpath = Path('pics/dynamic_exponent_z.png')
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved: {outpath}')


HEISENBERG_T_SWEEP = np.array([0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                               1.35, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5,
                               1.6, 1.7, 1.8, 2.0, 2.5, 3.0])


def plot_heisenberg_metropolis_mt():
    """Heisenberg Metropolis |m| vs T on 3D cubic lattice (Tc ≈ 1.443J)."""
    print('Running Heisenberg Metropolis temperature sweep (3D cubic)...')
    results = temperature_sweep('heisenberg_metropolis', HEISENBERG_T_SWEEP,
                                L=16, J=1.0, sweeps=10000, therm=5000, dim=3)

    mc_T = np.array([r['T'] for r in results])
    mc_m = np.array([r['observables']['magnetization']['mean'] for r in results])

    T_c_lit = 1.443  # 3D Heisenberg Tc from high-precision MC
    T_c_num = find_tc_numerical(mc_T, mc_m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mc_T, mc_m, 'o', color='#E91E63', ms=6,
            label='Heisenberg Metropolis (L=16)', zorder=5)
    ax.axvline(T_c_lit, color='gray', ls='--', lw=1, alpha=0.5,
               label=rf'$T_c^{{\mathrm{{lit}}}} = {T_c_lit:.3f}$')
    ax.axvline(T_c_num, color='#E91E63', ls=':', lw=1.5, alpha=0.8,
               label=rf'$T_c^{{\mathrm{{num}}}} = {T_c_num:.3f}$')
    ax.set_xlabel(r'$T$', fontsize=14)
    ax.set_ylabel(r'$\langle |\mathbf{m}| \rangle$', fontsize=14)
    ax.set_title(r'Heisenberg (O(3)) Metropolis MC — 3D Cubic Lattice', fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xlim(0.3, 3.2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Path('pics/heisenberg_metropolis_mt.png')
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_heisenberg_wolff_mt():
    """Heisenberg Wolff |m| vs T on 3D cubic lattice."""
    print('Running Heisenberg Wolff temperature sweep (3D cubic)...')
    results = temperature_sweep('heisenberg_wolff', HEISENBERG_T_SWEEP,
                                L=16, J=1.0, sweeps=5000, therm=2000, dim=3)

    mc_T = np.array([r['T'] for r in results])
    mc_m = np.array([r['observables']['magnetization']['mean'] for r in results])

    T_c_lit = 1.443
    T_c_num = find_tc_numerical(mc_T, mc_m)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mc_T, mc_m, '^', color='#9C27B0', ms=6,
            label='Heisenberg Wolff (L=16)', zorder=5)
    ax.axvline(T_c_lit, color='gray', ls='--', lw=1, alpha=0.5,
               label=rf'$T_c^{{\mathrm{{lit}}}} = {T_c_lit:.3f}$')
    ax.axvline(T_c_num, color='#9C27B0', ls=':', lw=1.5, alpha=0.8,
               label=rf'$T_c^{{\mathrm{{num}}}} = {T_c_num:.3f}$')
    ax.set_xlabel(r'$T$', fontsize=14)
    ax.set_ylabel(r'$\langle |\mathbf{m}| \rangle$', fontsize=14)
    ax.set_title(r'Heisenberg (O(3)) Wolff MC — 3D Cubic Lattice', fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xlim(0.3, 3.2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = Path('pics/heisenberg_wolff_mt.png')
    outpath.parent.mkdir(exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def plot_overlap_timeseries(sweeps: int = 5000, therm: int = 0, q: int = 2):
    """Plot ⟨|m|⟩(t) vs U_2(t) time series for all three algorithms.

    Runs each algorithm at T≈Tc with overlap enabled and plots the
    magnetization and overlap time series side by side.

    Note: Metropolis/SW use configuration overlap U_n = (1/N)Σ δ(s_i(t), s_i(t+n)),
    while Wolff uses cluster geometric overlap U_n^(W) = (1/N)|C_t ∩ C_{t+n}|.
    These are fundamentally different quantities.

    @param sweeps  Number of measurement sweeps.
    @param therm   Thermalization sweeps (0 = include thermalization in plot).
    @param q       Potts states (2 = Ising).
    """
    Tc = overlap_tc(q)
    temperatures = [0.66 * Tc, Tc, 1.33 * Tc]
    temp_labels = [r'$T < T_c$', r'$T \approx T_c$', r'$T > T_c$']
    algorithms = [
        ('metropolis', 'Metropolis', 'Config overlap'),
        ('swendsen_wang', 'Swendsen-Wang', 'Config overlap'),
        ('wolff', 'Wolff', 'Cluster overlap'),
    ]

    L = 64

    fig, axes = plt.subplots(len(algorithms), len(temperatures),
                             figsize=(15, 10), sharex=True)

    for row, (algo, algo_name, overlap_type) in enumerate(algorithms):
        for col, (T, tlabel) in enumerate(zip(temperatures, temp_labels)):
            ax = axes[row, col]
            print(f'  {algo_name} T={T:.3f} sweeps={sweeps} therm={therm}...')
            r = run_mc_binary(algo, L, 1.0, T, sweeps=sweeps, therm=therm,
                              ts=True, overlap_step=2, seed=42, q=q)

            ts_mag = r['time_series'].get('abs_magnetization', np.array([]))
            ts_overlap = r['time_series'].get('overlap', np.array([]))

            # Plot magnetization on left y-axis
            color_mag = _OKABE['blue']
            ax.plot(ts_mag, lw=0.3, alpha=0.6, color=color_mag)
            ax.set_ylabel(r'$|m|$', color=color_mag, fontsize=10)
            ax.tick_params(axis='y', labelcolor=color_mag)
            ax.set_ylim(-0.05, 1.05)

            # Plot overlap on right y-axis
            ax2 = ax.twinx()
            color_ov = _OKABE['vermillion']
            if len(ts_overlap) > 0:
                ax2.plot(ts_overlap, lw=0.3, alpha=0.6, color=color_ov)
            ax2.set_ylabel(rf'$U_2$', color=color_ov, fontsize=10)
            ax2.tick_params(axis='y', labelcolor=color_ov)
            ax2.set_ylim(-0.05, 1.05)

            # Labels
            if row == 0:
                ax.set_title(tlabel, fontsize=12)
            if row == len(algorithms) - 1:
                ax.set_xlabel('Sweep')

    row_labels = [f'{name}\n({otype})' for _, name, otype in algorithms]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label + '\n' + r'$|m|$', fontsize=9, color=color_mag)

    tag = f'{sweeps//1000}k'
    qtag = 'Ising' if q == 2 else f'{q}-state Potts'
    fig.suptitle(rf'Overlap $U_2$ vs Magnetization $|m|$ ({qtag}) — $L={L}$, '
                 rf'sweeps={tag}, therm={therm}',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = Path(f'pics/overlap_timeseries_{tag}{overlap_suffix(q)}.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


def _cached_sweep(algorithm, T_list, L, sweeps, therm, overlap_step, seed, q):
    """temperature_sweep result, disk-cached by the full parameter tuple."""
    Ttup = tuple(float(t) for t in T_list)
    key = ('sweep', algorithm, q, int(L), Ttup, int(sweeps), int(therm),
           int(overlap_step), int(seed))

    def compute():
        return temperature_sweep(algorithm, np.array(Ttup, dtype=float), L, 1.0,
                                 sweeps=sweeps, therm=therm,
                                 overlap_step=overlap_step, seed=seed, q=q)
    return _load_or_compute(key, compute, label=f'{algorithm}-sweep')


def _cached_run(algorithm, L, T, sweeps, therm, overlap_step, seed, q):
    """Single run_mc_binary result, disk-cached by the full parameter tuple."""
    key = ('run', algorithm, q, int(L), float(T), int(sweeps), int(therm),
           int(overlap_step), int(seed))

    def compute():
        return run_mc_binary(algorithm, L, 1.0, T, sweeps=sweeps, therm=therm,
                             overlap_step=overlap_step, seed=seed, q=q)
    return _load_or_compute(key, compute, label=f'{algorithm}-run')


def _panel_letter(ax, letter):
    """Bold (a)/(b)/... panel label in the top-left of an Axes."""
    ax.text(0.03, 0.95, f'({letter})', transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='top')


def plot_overlap_vs_temperature(q: int = 2):
    """Plot ⟨U_2⟩ and Var(U_2) vs T for all three algorithms.

    Temperature sweep showing how overlap mean and variance behave across
    the phase transition. Uses overlap_t_list(q) (dense near Tc).

    Note: Metropolis/SW use config overlap, Wolff uses cluster overlap.
    """
    Tc = overlap_tc(q)
    T_list = overlap_t_list(q)
    L = 64
    algorithms = [('metropolis', 'Metropolis', _OKABE['blue']),
                  ('swendsen_wang', 'SW', _OKABE['bluish_green']),
                  ('wolff', 'Wolff', _OKABE['vermillion'])]
    with plt.rc_context(_PAPER_RC):
        fig, (ax_mean, ax_var) = plt.subplots(1, 2, figsize=(7.0, 3.0))
        for algo, name, color in algorithms:
            results = _cached_sweep(algo, T_list, L, sweeps=20000, therm=5000,
                                    overlap_step=2, seed=42, q=q)
            T_arr = np.array([r['T'] for r in results])
            mean_ov = np.array([r['observables'].get('overlap', {}).get('mean', 0.0)
                                for r in results])
            var_ov = np.array([r['observables'].get('overlap', {}).get('variance', 0.0)
                               for r in results])
            ax_mean.plot(T_arr, mean_ov, 'o-', color=color, ms=3, lw=1.0,
                         label=name, alpha=0.85)
            ax_var.plot(T_arr, var_ov, 'o-', color=color, ms=3, lw=1.0,
                        label=name, alpha=0.85)
        for ax in (ax_mean, ax_var):
            ax.axvline(Tc, ls='--', color='gray', lw=0.8, alpha=0.7)
            ax.set_xlim(float(T_list.min()) * 0.85, float(T_list.max()) * 1.05)
            ax.tick_params(direction='in', top=True, right=True)
        ax_mean.set_xlabel(r'$T$')
        ax_mean.set_ylabel(r'overlap mean ($N_t{=}2$)')
        ax_var.set_xlabel(r'$T$')
        ax_var.set_ylabel(r'overlap variance')
        ax_var.set_yscale('log')
        ax_mean.legend(fontsize=7, loc='best')
        _panel_letter(ax_mean, 'a')
        _panel_letter(ax_var, 'b')
        fig.subplots_adjust(wspace=0.34)
        outpath = Path(f'pics/paper/overlap_vs_temperature{overlap_suffix(q)}.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


# ---------------------------------------------------------------------------
# V6: Theoretical verification plots
# ---------------------------------------------------------------------------

def plot_overlap_quick_checks(q: int = 2):
    """V6.1 + V6.2: Quick sanity checks for high-T Metropolis and SW U = 1/q.

    V6.1: Metropolis high-T limit. At T >> Tc, U_{N_t} approaches the
      thermodynamic high-T limit: Ising (1 + exp(-2*N_t))/2; q-state Potts 1/q.
      Test at T=1000 with N_t=1 and N_t=2.

    V6.2: Swendsen-Wang config overlap is exactly 1/q for all N_t >= 1
      and all temperatures (Eq. 2.1).
    """
    T_high = 1000.0
    Tc = overlap_tc(q)
    # Metropolis high-T limit: P_coin(Nt) = 1/q + (q-1)/q·exp(-Nt·q/(q-1))
    # (Ising: (1+e^{-2Nt})/2).
    U1_exact = 1.0 / q + (q - 1.0) / q * np.exp(-1.0 * q / (q - 1.0))
    U2_exact = 1.0 / q + (q - 1.0) / q * np.exp(-2.0 * q / (q - 1.0))
    sizes = [10, 20, 40]
    with plt.rc_context(_PAPER_RC):
        fig, (ax_m, ax_sw) = plt.subplots(1, 2, figsize=(7.0, 3.0))
        # (a) Metropolis high-T
        for L in sizes:
            u1 = [_cached_run('metropolis', L, T_high, sweeps=100000, therm=10000,
                              overlap_step=1, seed=s, q=q)['observables']
                  .get('overlap', {}).get('mean', 0.0) for s in (0, 1, 2)]
            u2 = [_cached_run('metropolis', L, T_high, sweeps=100000, therm=10000,
                              overlap_step=2, seed=s, q=q)['observables']
                  .get('overlap', {}).get('mean', 0.0) for s in (0, 1, 2)]
            ax_m.errorbar(1.0 / L, np.mean(u1), yerr=np.std(u1), fmt='o',
                          color=_OKABE['blue'], ms=4, capsize=2,
                          label=r'$N_t{=}1$' if L == sizes[0] else '')
            ax_m.errorbar(1.0 / L, np.mean(u2), yerr=np.std(u2), fmt='s',
                          color=_OKABE['vermillion'], ms=4, capsize=2,
                          label=r'$N_t{=}2$' if L == sizes[0] else '')
        ax_m.axhline(U1_exact, ls='--', color=_OKABE['blue'], lw=1.0, alpha=0.7)
        ax_m.axhline(U2_exact, ls='--', color=_OKABE['vermillion'], lw=1.0, alpha=0.7)
        ax_m.set_xlabel(r'$1/L$')
        ax_m.set_ylabel(r'$P_{\mathrm{coin}}$ ($T{=}1000$)')
        ax_m.set_xlim(0, 0.12)
        ax_m.legend(fontsize=7, loc='best')
        ax_m.tick_params(direction='in', top=True, right=True)
        _panel_letter(ax_m, 'a')
        # (b) SW config overlap = 1/q exactly
        _tsw = [t * (Tc / ISING_TC) for t in (1.0, 1.5, 2.0, ISING_TC, 2.5, 3.0, 4.0)]
        L_sw = 20
        sw_means, sw_errs = [], []
        for T in _tsw:
            vals = [_cached_run('swendsen_wang', L_sw, float(T), sweeps=50000, therm=5000,
                                overlap_step=1, seed=s, q=q)['observables']
                    .get('overlap', {}).get('mean', 1.0 / q) for s in (42, 123, 456)]
            sw_means.append(np.mean(vals))
            sw_errs.append(np.std(vals))
        ax_sw.errorbar(_tsw, sw_means, yerr=sw_errs, fmt='o', color=_OKABE['bluish_green'],
                       ms=4, capsize=2, label='SW measured')
        ax_sw.axhline(1.0 / q, ls='--', color='gray', lw=1.0, alpha=0.7,
                      label=r'$1/q$ exact')
        ax_sw.axvline(Tc, ls=':', color='gray', lw=0.8, alpha=0.5)
        ax_sw.set_xlabel(r'$T$')
        ax_sw.set_ylabel(r'$P_{\mathrm{coin}}$')
        ax_sw.set_ylim(1.0 / q - 0.03, 1.0 / q + 0.03)
        ax_sw.legend(fontsize=7, loc='best')
        ax_sw.tick_params(direction='in', top=True, right=True)
        _panel_letter(ax_sw, 'b')
        fig.subplots_adjust(wspace=0.34)
        outpath = Path(f'pics/paper/overlap_quick_checks{overlap_suffix(q)}.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


def plot_overlap_cluster_mean(q: int = 2):
    """V6.4: Wolff cluster overlap ⟨U_2^{(W)}⟩ temperature sweep, multiple L.

    Verifies that ⟨U_2^{(W)}⟩ drops from ~1 (low T) to ~0 (high T),
    acting as an order parameter. Uses dense T points near Tc.
    """
    Tc = overlap_tc(q)
    T_list = overlap_t_list(q)
    L_list = [64, 128]
    colors = [_OKABE['vermillion'], _OKABE['orange']]
    markers = ['o', 's']
    with plt.rc_context(_PAPER_RC):
        fig, (ax_mean, ax_var) = plt.subplots(1, 2, figsize=(7.0, 3.0))
        for i, L in enumerate(L_list):
            results = _cached_sweep('wolff', T_list, L, sweeps=20000, therm=5000,
                                    overlap_step=2, seed=42, q=q)
            T_arr = np.array([r['T'] for r in results])
            mean_ov = np.array([r['observables'].get('overlap', {}).get('mean', 0.0)
                                for r in results])
            var_ov = np.array([r['observables'].get('overlap', {}).get('variance', 0.0)
                               for r in results])
            ax_mean.plot(T_arr, mean_ov, f'{markers[i]}-', color=colors[i],
                         ms=3, lw=1.0, label=f'$L={L}$', alpha=0.85)
            ax_var.plot(T_arr, var_ov, f'{markers[i]}-', color=colors[i],
                        ms=3, lw=1.0, label=f'$L={L}$', alpha=0.85)
        for ax in (ax_mean, ax_var):
            ax.axvline(Tc, ls='--', color='gray', lw=0.8, alpha=0.7)
            ax.set_xlim(float(T_list.min()) * 0.85, float(T_list.max()) * 1.05)
            ax.tick_params(direction='in', top=True, right=True)
        ax_mean.set_xlabel(r'$T$')
        ax_mean.set_ylabel(r'$\langle\mathcal{R}^W(2)\rangle$')
        ax_mean.set_ylim(-0.02, 1.0)
        ax_mean.legend(fontsize=7)
        ax_var.set_xlabel(r'$T$')
        ax_var.set_ylabel(r'$\mathrm{Var}(\mathcal{R}^W)$')
        ax_var.set_yscale('log')
        _panel_letter(ax_mean, 'a')
        _panel_letter(ax_var, 'b')
        fig.subplots_adjust(wspace=0.34)
        outpath = Path(f'pics/paper/overlap_cluster_mean{overlap_suffix(q)}.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


# --- Wolff cluster-overlap FSS data, shared by the Var and mean figures ---
# --- FSS data collection: memory + disk cache (FSS sweeps are expensive) ---
_FSS_CACHE_DIR = Path(__file__).parent / '.fss_cache'
_FSS_MEM_CACHE = {}


def _load_or_compute(key, compute_fn, label: str = ''):
    """Memory + disk cache for expensive FSS data collection.

    key: hashable tuple (fully identifies the sim: algorithm, q, L_list, Nt_list,
    sweeps, therm, seed). Cached to .fss_cache/<hash>.pkl so plot tweaks never
    re-run the MC sims.  Bumping sweeps/therm/L_list changes the key → recompute.
    """
    if key in _FSS_MEM_CACHE:
        return _FSS_MEM_CACHE[key]
    _FSS_CACHE_DIR.mkdir(exist_ok=True)
    h = hashlib.md5(repr(key).encode()).hexdigest()[:16]
    path = _FSS_CACHE_DIR / f'{h}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f'  [disk-cache hit] {label}  {key}')
        _FSS_MEM_CACHE[key] = data
        return data
    data = compute_fn()
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as exc:  # cache is best-effort
        print(f'  [warn] cache write failed: {exc}')
    _FSS_MEM_CACHE[key] = data
    return data


# --- Multi-Nt overlap runs: per-(algo,q,L,seed) .npz series cache ------------
# 7 seeds for SEM error bars; dense Nt for the Nt-axis plots; NT_FSS = the
# headline-FSS column subset. Tc uses the Potts binary convention (potts_tc).
SEEDS = (42, 7, 123, 256, 2024, 114514, 2077)
NT_DENSE = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)  # ~log-spaced
NT_FSS = (1, 2, 100, 400)                                                 # headline FSS columns
NT_FULLNT = (1, 2, 10, 50, 100, 200, 400, 1000)                           # full-Nt supplementary grid rows
NT_LARGE = tuple(sorted(set(NT_FSS) | set(NT_FULLNT)))                    # large-L (≥1024) canonical set — headline+fullnt only
NT_FINE = tuple(sorted({int(round(x)) for x in np.logspace(0, 3, 70)}))   # ~60 pts 1..1000, dense ⟨R^W⟩-vs-Nt curve (small L only)
# One canonical multi-Nt run per (L,seed); plots slice subsets. Large L uses the
# reduced NT_LARGE set (overlap measurement is O(|Nt|·N) per sweep and would
# dominate at big L); small L uses the full NT_CANONICAL.
NT_CANONICAL = tuple(sorted(set(NT_DENSE) | set(NT_FSS) | set(NT_FULLNT)))


def _canonical_nt(L_list):
    """Canonical Nt set for a run: NT_LARGE if any L≥1024 (overlap-compute is
    O(|Nt|·N) per sweep, costly at big L), else NT_CANONICAL."""
    return NT_LARGE if max(int(L) for L in L_list) >= 1024 else NT_CANONICAL

_OVERLAP_SERIES_DIR = Path(__file__).parent / 'overlap_series'
_OVERLAP_SERIES_MEM = {}


def _therm_for_L(L: int, base: int = 10000) -> int:
    """Thermalization ramp: base for L≤256, +1000 per doubling past L=256.

    L=256→10000, 512→11000, 1024→12000, 2048→13000, 4096→14000.
    """
    L = int(L)
    if L <= 256:
        return base
    doublings = int(round(np.log2(L / 256.0)))
    return base + 1000 * doublings


def _collect_overlap_series(algo: str, q: int, L: int, seed: int,
                            Nt_list, sweeps: int, therm: int,
                            all_up: bool = True, cluster_only: bool = False) -> dict:
    """Run the Potts binary ONCE for (algo, q, L, seed), measuring overlap at every
    N_t in Nt_list from a single trajectory; cache mean/var + raw time series as
    .npz.  Returns {'nt','mean':{Nt:float},'var':{Nt:float},'series':{Nt:array}}.
    Uses T = potts_tc(q). Series length = measurement sweeps (history is primed
    during thermalization, so every measurement overlap is valid). cluster_only
    (Wolff) skips the config-overlap buffer to save memory at large L."""
    nt_tuple = tuple(int(n) for n in Nt_list)
    key = (algo, q, int(L), int(seed), nt_tuple, int(sweeps), int(therm),
           bool(all_up), bool(cluster_only))
    if key in _OVERLAP_SERIES_MEM:
        return _OVERLAP_SERIES_MEM[key]
    _OVERLAP_SERIES_DIR.mkdir(exist_ok=True)
    h = hashlib.md5(repr(key).encode()).hexdigest()[:16]
    path = _OVERLAP_SERIES_DIR / f'{h}.npz'
    if path.exists():
        z = np.load(path, allow_pickle=True)
        nt = z['nt']
        csm = float(z['cluster_size_mean']) if 'cluster_size_mean' in z.files else 0.0
        out = {'nt': nt,
               'mean': {int(n): float(m) for n, m in zip(nt, z['mean'])},
               'var':  {int(n): float(v) for n, v in zip(nt, z['var'])},
               'series': {int(n): z[f'series_Nt{int(n)}'] for n in nt},
               'cluster_size_mean': csm}
        _OVERLAP_SERIES_MEM[key] = out
        print(f'  [series-cache hit] {algo} q={q} L={L} seed={seed}')
        return out

    Tc = potts_tc(q)
    print(f'  {algo} series  q={q}  L={L}  seed={seed}  all_up={all_up}  Nt={nt_tuple}')
    r = run_mc_binary(algo, L, 1.0, Tc, sweeps=sweeps, therm=therm,
                      seed=seed, q=q, all_up=all_up, overlap_steps=nt_tuple,
                      cluster_only=cluster_only)
    obs = r['observables']
    ts = r['time_series']
    mean, var, series = {}, {}, {}
    for n in nt_tuple:
        name = f'overlap_Nt{n}'
        ov = obs.get(name, {})
        mean[n] = float(ov.get('mean', 0.0))
        var[n] = float(ov.get('variance', 0.0))
        series[n] = np.asarray(ts.get(name, []), dtype=float)
    cluster_size_mean = float(obs.get('cluster_size', {}).get('mean', 0.0))   # Wolff χ = β⟨|C|⟩
    save = {'nt': np.array(nt_tuple, dtype=int),
            'mean': np.array([mean[n] for n in nt_tuple]),
            'var':  np.array([var[n] for n in nt_tuple]),
            'cluster_size_mean': np.array(cluster_size_mean)}
    for n in nt_tuple:
        save[f'series_Nt{n}'] = series[n]
    try:
        np.savez(path, **save)
    except Exception as exc:  # cache is best-effort
        print(f'  [warn] series cache write failed: {exc}')
    out = {'nt': save['nt'], 'mean': mean, 'var': var, 'series': series,
           'cluster_size_mean': cluster_size_mean}
    _OVERLAP_SERIES_MEM[key] = out
    return out


def _collect_wolff_overlap_fss(q: int = 2,
                               L_list=(16, 32, 64, 128, 256),
                               Nt_list=(1, 2, 100),
                               sweeps: int = 20000, therm=None,
                               seeds=SEEDS, all_up: bool = True,
                               nt_run=None) -> dict:
    """Wolff cluster overlap 𝓡^W at Tc, multi-seed.

    For each (L, seed) runs ONE canonical multi-Nt trajectory (cached as .npz in
    overlap_series/), measuring every N_t in NT_CANONICAL; the requested Nt_list
    is then sliced out (free — extra Nt costs no MCMC). Aggregates over `seeds`
    and returns
        {Nt: {'L','mean','mean_sem','var','var_sem','mean_ps','var_ps'}}
    mean/var are seed-averaged; *_sem = std/√n_seeds; *_ps are (n_seeds, n_L)
    per-seed arrays for exponent-error fits. T = potts_tc(q); thermalization uses
    the _therm_for_L ramp when therm is None. all_up flags ordered init (large L).
    """
    requested = sorted({int(n) for n in Nt_list})
    canonical = _canonical_nt(L_list) if nt_run is None else tuple(int(n) for n in nt_run)
    missing = [n for n in requested if n not in canonical]
    if missing:
        raise ValueError(f'Nt {missing} not available for L_list={list(L_list)} '
                         f'(canonical set has {len(canonical)} Nt)')
    per_L = {}
    for L in L_list:
        th = _therm_for_L(L) if therm is None else int(therm)
        per_L[L] = [_collect_overlap_series('wolff', q, int(L), int(s), canonical,
                                             int(sweeps), th, all_up, cluster_only=True)
                    for s in seeds]
    datasets = {}
    L_arr = np.array([int(L) for L in L_list], dtype=float)
    ns = len(seeds)
    sem_scale = np.sqrt(ns) if ns > 1 else 1.0
    for n in requested:
        mean_ps = np.array([[d['mean'][n] for d in per_L[L]] for L in L_list]).T  # (n_seeds, n_L)
        var_ps = np.array([[d['var'][n] for d in per_L[L]] for L in L_list]).T
        datasets[n] = {
            'L': L_arr,
            'mean': mean_ps.mean(axis=0),
            'var': var_ps.mean(axis=0),
            'mean_sem': (mean_ps.std(axis=0, ddof=1) / sem_scale) if ns > 1 else np.zeros(len(L_arr)),
            'var_sem': (var_ps.std(axis=0, ddof=1) / sem_scale) if ns > 1 else np.zeros(len(L_arr)),
            'mean_ps': mean_ps,
            'var_ps': var_ps,
        }
    return datasets


def plot_overlap_variance_fss(q: int = 2):
    """Wolff cluster-overlap Var(𝓡^W) FSS collapse-test at Tc (paper format).

    Rows Nt∈{1,2,∞=100}; cols = {L^{-(2-γ/ν)} [Nt=1 χ/N regime], L^{-0.42}
    [Nt=2 lit], L^{-2η} [Nt=∞ (χ/N)²]}. The diagonal collapses (high R²).
    Also emits a +L^{-1} comparison variant."""
    # L∈{16..256} random (cached) + Nt=400. L=512: random OK for Ising (q=2), but
    # q=3,4 Potts need all-up (random under-thermalizes at L=512). L=1024 only q=2
    # (potts_wolff fails even with all-up at 10^6 for q=3,4). → q=2: 7 pts, q=3,4: 6.
    _wolff_collects = [
        _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(1, 2, 100)),
        _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(400,)),
        _collect_wolff_overlap_fss(q=q, L_list=(512,), Nt_list=(1, 2, 100, 400), all_up=True),
    ]
    if q == 2:
        _wolff_collects.append(
            _collect_wolff_overlap_fss(q=q, L_list=(1024,), Nt_list=(1, 2, 100, 400), all_up=True))
        _wolff_collects.append(
            _collect_wolff_overlap_fss(q=q, L_list=(2048, 4096), Nt_list=(1, 2, 100, 400), sweeps=10000, all_up=True))
    datasets = _merge_L(*_wolff_collects)
    naive = overlap_naive_a_exp(q)
    indep = overlap_indep_exp(q)
    exp_cols = [(naive, r'$L^{-%.3f}$ ($\chi/N$ naive)' % naive),
                (0.42, r'$L^{-0.42}$ (lit., $N_t=2$)'),
                (indep, r'$L^{-%.3f}$ (indep. $(\chi/N)^2$)' % indep)]
    ylab = lambda Nt: r'$\mathrm{Var}\,\mathcal{R}^W(N_t=%s)$' % _nt_tex(Nt)
    nts = [1, 2, 100, 400]
    _collapse_grid(datasets, nts, exp_cols, q, 'var', ylab,
                   f'pics/paper/overlap_variance_fss{overlap_suffix(q)}.png',
                   suptitle=r'FSS of $\mathrm{Var}(\mathcal{R}^W)$ at $T_c$')
    _collapse_grid(datasets, nts, exp_cols, q, 'var', ylab,
                   f'pics/paper/overlap_variance_fss_L1{overlap_suffix(q)}.png',
                   suptitle=r'FSS of $\mathrm{Var}(\mathcal{R}^W)$ at $T_c$ (+$L^{-1}$)',
                   include_L1=True)


def _plot_fss_collapse(datasets, panels, q, which, ylabel_fn, title, outpath,
                       ncols=3):
    """[LEGACY slide-format collapse grid] Kept for reference; paper figures use
    _collapse_grid instead. Original flat-list linear L^{-e} grid with R²."""
    palette = [_OKABE['vermillion'], _OKABE['blue'], _OKABE['bluish_green'], _OKABE['purple'], _OKABE['orange']]
    nts = sorted({p[0] for p in panels})
    nt_color = {nt: palette[i % len(palette)] for i, nt in enumerate(nts)}
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows),
                             squeeze=False)
    for k, (Nt, exponent, xlabel) in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        d = datasets[Nt]
        L_arr = d['L']
        y = d['var'] if which == 'var' else d['mean']
        yerr = d['var_sem'] if which == 'var' else d['mean_sem']
        x = L_arr ** (-exponent)
        color = nt_color[Nt]
        ax.errorbar(x, y, yerr=yerr, fmt='o', color=color, ms=8, elinewidth=0.8,
                    capsize=3, zorder=3, label='MC data')
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        x_fit = np.array([0.0, x.max() * 1.05])
        ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=color, alpha=0.6,
                lw=1.5, label=rf'$R^2 = {r2:.6f}$')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel_fn(Nt), fontsize=10)
        ax.set_title(rf'$N_t={Nt}$,  {xlabel}', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
    for k in range(len(panels), nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    qtag = 'Ising' if q == 2 else f'{q}-state Potts'
    fig.suptitle(rf'{title} ({qtag})', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


# --- paper-figure style + collapse-grid helpers (journal format) ---
_PAPER_RC = {
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9.0,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    # Vector output: embed text as TrueType (not Type 3) so PDFs stay fully
    # editable in Illustrator / Inkscape (journal-standard vector figures).
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

# Okabe–Ito colorblind-safe palette (all paper figures). Replaces the legacy
# hardcoded hexes so overlapping curves/points remain distinguishable to
# color-deficient readers (Elsevier accessibility practice).
_OKABE = {
    'blue':         '#0072B2',   # was #004983
    'vermillion':   '#D55E00',   # was #C0392B
    'bluish_green': '#009E73',   # was #27AE60
    'purple':       '#CC79A7',   # was #6A3D9A / #8E44AD
    'orange':       '#E69F00',   # was #D35400 / #E67E22 / #a6761d
    'yellow':       '#F0E442',   # was #e6ab02
    'sky_blue':     '#56B4E9',
    'black':        '#000000',
}


def _paper_style():
    """Apply journal rcParams globally (call before generating paper figures)."""
    plt.rcParams.update(_PAPER_RC)


def _paper_savefig(fig, outpath):
    """Save a paper figure as BOTH a 300-dpi PNG (preview) and a vector PDF
    (journal-standard; Illustrator/Inkscape-friendly via pdf.fonttype=42).

    outpath is the existing .png path; the .pdf twin is written alongside.
    pdflatex's default extension order (pdf > png) then picks the vector file
    automatically — the \\includegraphics{basename} in paper.tex is unchanged.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    for p in (outpath, outpath.with_suffix('.pdf')):
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {p}')
    plt.close(fig)


def _nt_tex(Nt):
    """Render an Nt value as its literal integer (figures show the REAL measurement
    Nt — e.g. 100, 400 — never a misleading ∞; the Nt→∞ independent-limit
    interpretation is discussed in the caption/text, not baked into the axis)."""
    return str(Nt)


def _merge_L(*datasets_list):
    """Merge several {Nt: {...}} FSS dicts by concatenating per-Nt arrays along L
    (sorted by L). Stitch separately-cached L-ranges (e.g. L≤512 random-init +
    L=1024 all-up-init) into one FSS curve. All dicts share the same seed
    ordering (SEEDS), so per-L SEMs concatenate correctly; per-seed arrays
    (n_seeds, n_L) concatenate along axis=1."""
    out = {}
    for ds in datasets_list:
        for Nt, d in ds.items():
            if Nt in out:
                o = out[Nt]
                idx = np.argsort(np.concatenate([o['L'], d['L']]))

                def cat1(a, b):
                    return np.concatenate([a, b])[idx]
                out[Nt] = {
                    'L': cat1(o['L'], d['L']),
                    'mean': cat1(o['mean'], d['mean']),
                    'var': cat1(o['var'], d['var']),
                    'mean_sem': cat1(o['mean_sem'], d['mean_sem']),
                    'var_sem': cat1(o['var_sem'], d['var_sem']),
                    'mean_ps': np.concatenate([o['mean_ps'], d['mean_ps']], axis=1)[:, idx],
                    'var_ps': np.concatenate([o['var_ps'], d['var_ps']], axis=1)[:, idx],
                }
            else:
                out[Nt] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    return out


def _collapse_grid(datasets, nts, exp_cols, q, which, row_ylabel_fn, outpath,
                   suptitle='', include_L1=False):
    """Paper-format FSS collapse-test grid (LINEAR axes): rows = Nt, cols = exponent.

    Each panel plots the observable vs L^{-e} with a linear fit + R^2. Shared row
    y-labels (left column), shared col x-labels (bottom row), panel letters
    (a)/(b)/..., compact journal styling. Optionally appends an L^{-1} comparison
    column (the generic 1/L axis, no physical meaning).

    datasets      : {Nt: {'L','mean','mean_sem','var','var_sem','mean_ps','var_ps'}}.
    nts           : ordered Nt values for the rows.
    exp_cols      : list of (exponent_e, xlabel) for the columns (theory exponents).
    which         : 'var' or 'mean' (which tuple entry to plot).
    row_ylabel_fn : callable Nt -> str, the left-column y-label for that row.
    """
    cols = list(exp_cols) + ([(1.0, r'$L^{-1}$')] if include_L1 else [])
    nrows, ncols = len(nts), len(cols)
    # panel labels (a, b, …, z, then aa, ab, …) — robust for large N_t grids (8×3=24)
    def _lbl(idx):
        out = ''
        idx += 1
        while idx > 0:
            idx, rem = divmod(idx - 1, 26)
            out = chr(ord('a') + rem) + out
        return out
    letters = [_lbl(i) for i in range(nrows * ncols)]
    with plt.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.3 * ncols + 0.4, 1.55 * nrows + 0.4),
                                 squeeze=False)
        for ri, Nt in enumerate(nts):
            d = datasets[Nt]
            L_arr = d['L']
            y = d['var'] if which == 'var' else d['mean']
            yerr = d['var_sem'] if which == 'var' else d['mean_sem']
            for ci, (exponent, xlabel) in enumerate(cols):
                ax = axes[ri][ci]
                x = L_arr ** (-exponent)
                ax.plot(x, y, 'o', mfc='white', mec=_OKABE['blue'], ms=4, zorder=3)
                ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor=_OKABE['blue'],
                            elinewidth=1.2, capsize=3, zorder=4)
                coeffs = np.polyfit(x, y, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                x_fit = np.array([0.0, x.max() * 1.05])
                ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=_OKABE['vermillion'],
                        alpha=0.7, lw=1.0)
                ax.text(0.97, 0.03, rf'$R^2={r2:.6f}$', transform=ax.transAxes,
                        fontsize=7, ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.18', fc='white',
                                  ec='none', alpha=0.85))
                ax.text(0.03, 0.95, f'({letters[ri * ncols + ci]})',
                        transform=ax.transAxes, fontsize=8.5, fontweight='bold',
                        ha='left', va='top')
                if ri == nrows - 1:
                    ax.set_xlabel(xlabel)
                if ci == 0:
                    ax.set_ylabel(row_ylabel_fn(Nt))
                ax.set_xlim(left=0)
                ax.tick_params(direction='in', top=True, right=True)
        fig.subplots_adjust(hspace=0.22, wspace=0.26)
        if suptitle:
            qtag = 'Ising' if q == 2 else f'{q}-state Potts'
            fig.suptitle(f'{suptitle} ({qtag})', fontsize=9.5, y=0.995)
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


def plot_overlap_mean_fss(q: int = 2):
    """Wolff mean cluster-overlap ⟨𝓡^W(Nt)⟩ FSS collapse-test at Tc (paper format).

    Rows Nt∈{1,2,∞=100}; cols = {L^{-(2-γ/ν)} [Nt=1 analytic χ/N], L^{-0.42}
    [Nt=2 lit], L^{-2η} [Nt=∞ analytic (χ/N)²]}. The diagonal collapses (high R²).
    Also emits a +L^{-1} comparison variant."""
    # L∈{16..256} random (cached) + Nt=400. L=512: random OK for Ising (q=2), but
    # q=3,4 Potts need all-up (random under-thermalizes at L=512). L=1024 only q=2
    # (potts_wolff fails even with all-up at 10^6 for q=3,4). → q=2: 7 pts, q=3,4: 6.
    _wolff_collects = [
        _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(1, 2, 100)),
        _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(400,)),
        _collect_wolff_overlap_fss(q=q, L_list=(512,), Nt_list=(1, 2, 100, 400), all_up=True),
    ]
    if q == 2:
        _wolff_collects.append(
            _collect_wolff_overlap_fss(q=q, L_list=(1024,), Nt_list=(1, 2, 100, 400), all_up=True))
        _wolff_collects.append(
            _collect_wolff_overlap_fss(q=q, L_list=(2048, 4096), Nt_list=(1, 2, 100, 400), sweeps=10000, all_up=True))
    datasets = _merge_L(*_wolff_collects)
    naive = overlap_naive_a_exp(q)
    indep = overlap_indep_exp(q)
    exp_cols = [(naive, r'$L^{-%.3f}$ ($\chi/N$ naive)' % naive),
                (0.42, r'$L^{-0.42}$ (lit., $N_t=2$)'),
                (indep, r'$L^{-%.3f}$ (indep. $(\chi/N)^2$)' % indep)]
    ylab = lambda Nt: r'$\langle\mathcal{R}^W(N_t=%s)\rangle$' % _nt_tex(Nt)
    nts = [1, 2, 100, 400]
    _collapse_grid(datasets, nts, exp_cols, q, 'mean', ylab,
                   f'pics/paper/overlap_mean_fss{overlap_suffix(q)}.png',
                   suptitle=r'FSS of $\langle\mathcal{R}^W\rangle$ at $T_c$')
    _collapse_grid(datasets, nts, exp_cols, q, 'mean', ylab,
                   f'pics/paper/overlap_mean_fss_L1{overlap_suffix(q)}.png',
                   suptitle=r'FSS of $\langle\mathcal{R}^W\rangle$ at $T_c$ (+$L^{-1}$)',
                   include_L1=True)


# --- Config-overlap (spin overlap U) FSS for Metropolis / SW ---
def _collect_config_overlap_fss(algorithm: str, q: int = 2,
                                L_list=(16, 32, 64, 128),
                                Nt_list=(1, 2, 100),
                                sweeps: int = 20000, therm=None,
                                seeds=SEEDS, all_up: bool = True,
                                nt_run=None) -> dict:
    """Metropolis / Swendsen-Wang config overlap P_coin at Tc, multi-seed.

    For each (L, seed) runs ONE canonical multi-Nt trajectory (cached as .npz in
    overlap_series/), measuring every N_t in NT_CANONICAL; the requested Nt_list
    is sliced out (free). For Met/SW the primary observable 'overlap_Nt{n}' IS
    the config overlap P_coin. Returns the same dict shape as
    _collect_wolff_overlap_fss. T = potts_tc(q); therm ramp when therm is None.
    """
    requested = sorted({int(n) for n in Nt_list})
    canonical = _canonical_nt(L_list) if nt_run is None else tuple(int(n) for n in nt_run)
    missing = [n for n in requested if n not in canonical]
    if missing:
        raise ValueError(f'Nt {missing} not available for L_list={list(L_list)} '
                         f'(canonical set has {len(canonical)} Nt)')
    per_L = {}
    for L in L_list:
        th = _therm_for_L(L) if therm is None else int(therm)
        per_L[L] = [_collect_overlap_series(algorithm, q, int(L), int(s), canonical,
                                             int(sweeps), th, all_up)
                    for s in seeds]
    datasets = {}
    L_arr = np.array([int(L) for L in L_list], dtype=float)
    ns = len(seeds)
    sem_scale = np.sqrt(ns) if ns > 1 else 1.0
    for n in requested:
        mean_ps = np.array([[d['mean'][n] for d in per_L[L]] for L in L_list]).T
        var_ps = np.array([[d['var'][n] for d in per_L[L]] for L in L_list]).T
        datasets[n] = {
            'L': L_arr,
            'mean': mean_ps.mean(axis=0),
            'var': var_ps.mean(axis=0),
            'mean_sem': (mean_ps.std(axis=0, ddof=1) / sem_scale) if ns > 1 else np.zeros(len(L_arr)),
            'var_sem': (var_ps.std(axis=0, ddof=1) / sem_scale) if ns > 1 else np.zeros(len(L_arr)),
            'mean_ps': mean_ps,
            'var_ps': var_ps,
        }
    return datasets


def plot_overlap_sw_variance_fss(q: int = 2):
    """SW config-overlap Var(P_coin) FSS collapse-test at Tc (paper format).

    Rows Nt∈{1,2,∞=100}; cols = {L^{-(2-γ/ν)} [Nt=1 exact (q−1)/q²·χ/N],
    L^{-ψ^SW} [Nt=2 lit, q-aware 0.346/0.318/0.288], L^{-2η} [Nt=∞]}. The diagonal
    collapses. Also emits a +L^{-1} comparison variant."""
    # SW to large L for Ising (q=2), consistent with Wolff: bit-packed spin_history_
    # makes config-overlap feasible at L=4096; all_up=True (default). q=3,4 stay ≤512.
    sw_collects = [
        _collect_config_overlap_fss('swendsen_wang', q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(1, 2, 100, 400)),
        _collect_config_overlap_fss('swendsen_wang', q=q, L_list=(512,), Nt_list=(1, 2, 100, 400)),
    ]
    if q == 2:
        sw_collects.append(_collect_config_overlap_fss('swendsen_wang', q=q, L_list=(1024,), Nt_list=(1, 2, 100, 400)))
        sw_collects.append(_collect_config_overlap_fss('swendsen_wang', q=q, L_list=(2048,), Nt_list=(1, 2, 100, 400), sweeps=10000))
    datasets = _merge_L(*sw_collects)
    nt1 = overlap_naive_a_exp(q)      # 2-γ/ν, positive e for L^{-e}
    sw_lit = sw_lit_exp(q)            # ψ^SW, q-aware (Pilé Table I)
    indep = overlap_indep_exp(q)
    exp_cols = [(nt1, r'$L^{-%.3f}$ (exact, $N_t=1$)' % nt1),
                (sw_lit, r'$L^{-%.3f}$ (lit., $N_t=2$)' % sw_lit),
                (indep, r'$L^{-%.3f}$ (indep.)' % indep)]
    ylab = lambda Nt: r'$\mathrm{Var}(P_{\mathrm{coin}})$, $N_t=%s$' % _nt_tex(Nt)
    nts = [1, 2, 100, 400]
    _collapse_grid(datasets, nts, exp_cols, q, 'var', ylab,
                   f'pics/paper/overlap_sw_variance_fss{overlap_suffix(q)}.png',
                   suptitle=r'FSS of SW $\mathrm{Var}(P_{\mathrm{coin}})$ at $T_c$')
    _collapse_grid(datasets, nts, exp_cols, q, 'var', ylab,
                   f'pics/paper/overlap_sw_variance_fss_L1{overlap_suffix(q)}.png',
                   suptitle=r'FSS of SW $\mathrm{Var}(P_{\mathrm{coin}})$ at $T_c$ (+$L^{-1}$)',
                   include_L1=True)


def plot_overlap_metropolis_variance_fss(q: int = 2):
    """Metropolis config-overlap Var(P_coin) FSS at Tc — REFERENCE only (paper format).

    No analytic scaling at Tc (critical slowing down). 3×2 grid tests L^{-2η}
    (independent) and L^{-1}; uniformly poor R² demonstrates the absence of a
    universal exponent. (L^{-1} already a column, so no separate comparison variant.)"""
    datasets = _collect_config_overlap_fss('metropolis', q=q,
                                           L_list=[16, 32, 64, 128],
                                           Nt_list=(1, 2, 100))
    indep = overlap_indep_exp(q)
    exp_cols = [(indep, r'$L^{-%.3f}$ (indep.)' % indep),
                (1.0, r'$L^{-1}$')]
    ylab = lambda Nt: r'$\mathrm{Var}(P_{\mathrm{coin}})$, $N_t=%s$' % _nt_tex(Nt)
    _collapse_grid(datasets, [1, 2, 100], exp_cols, q, 'var', ylab,
                   f'pics/paper/overlap_metropolis_variance_fss{overlap_suffix(q)}.png',
                   suptitle=r'Metropolis $\mathrm{Var}(P_{\mathrm{coin}})$ at $T_c$ (reference)')


def plot_overlap_wolff_chi_relation(q: int = 2):
    """V6.3: Verify Wolff U_1 = 1 - χ/N_L (formula 3.5).

    At each temperature, independently measure:
      - U_1 (config overlap with N_t=1)
      - χ from cluster_size observable (<|C|> = χ, FK identity Eq. 3.2)
    Plot measured U_1 vs (1 - χ/N_L). Points should lie on diagonal.
    (Identity is q-independent.)
    """
    Tc = overlap_tc(q)
    T_list = np.array([0.5, 1.0, 1.5, 2.0, 2.1, 2.2, 2.25, 2.27,
                       2.3, 2.4, 2.5, 3.0]) * (Tc / ISING_TC)
    L = 40
    N_L = L * L
    u1_measured, prediction = [], []
    for T in T_list:
        r = _cached_run('wolff', L, float(T), sweeps=50000, therm=10000,
                        overlap_step=1, seed=42, q=q)
        u1 = r['observables'].get('config_overlap', {}).get('mean', 0.0)
        chi = r['observables'].get('cluster_size', {}).get('mean', 0.0)  # FK: χ=⟨|C|⟩
        u1_measured.append(u1)
        prediction.append(1.0 - chi / N_L)
    u1_measured = np.array(u1_measured)
    prediction = np.array(prediction)
    with plt.rc_context(_PAPER_RC):
        fig, (ax_scatter, ax_ts) = plt.subplots(1, 2, figsize=(7.0, 3.0))
        ax_scatter.plot(prediction, u1_measured, 'o', color=_OKABE['vermillion'], ms=4,
                        label=r'measured $P_{\mathrm{coin}}$')
        lims = [min(prediction.min(), u1_measured.min()) - 0.02,
                max(prediction.max(), u1_measured.max()) + 0.02]
        ax_scatter.plot(lims, lims, '--', color='gray', lw=1.0, alpha=0.7,
                        label='diagonal (exact)')
        ax_scatter.set_xlabel(r'$1-\chi/N$ (from $\langle|C|\rangle$)')
        ax_scatter.set_ylabel(r'$P_{\mathrm{coin}}(N_t{=}1)$, measured')
        ax_scatter.legend(fontsize=7, loc='best')
        ax_scatter.set_aspect('equal', adjustable='datalim')
        ax_scatter.tick_params(direction='in', top=True, right=True)
        _panel_letter(ax_scatter, 'a')
        ax_ts.plot(T_list, u1_measured, 'o-', color=_OKABE['vermillion'], ms=3, lw=1.0,
                   label=r'$P_{\mathrm{coin}}$ (measured)')
        ax_ts.plot(T_list, prediction, 's--', color=_OKABE['blue'], ms=3, lw=1.0,
                   label=r'$1-\chi/N$')
        ax_ts.axvline(Tc, ls=':', color='gray', lw=0.8, alpha=0.5)
        ax_ts.set_xlabel(r'$T$')
        ax_ts.set_ylabel(r'$P_{\mathrm{coin}}(N_t{=}1)$')
        ax_ts.legend(fontsize=7, loc='best')
        ax_ts.tick_params(direction='in', top=True, right=True)
        _panel_letter(ax_ts, 'b')
        fig.subplots_adjust(wspace=0.34)
        outpath = Path(f'pics/paper/overlap_wolff_chi{overlap_suffix(q)}.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


def plot_overlap_self_consistency(q: int = 2):
    """V6.6: Self-consistency check for formula (3.7).

    Verify: U_2(config) = 1 - 2χ/N_L + (q/(q-1))⟨U_2^{(W)}(cluster)⟩
    (coefficient 2 for Ising q=2). All three quantities measured independently
    from a single Wolff run.
    """
    Tc = overlap_tc(q)
    T_list = np.array([0.5, 1.0, 1.5, 2.0, 2.1, 2.2, 2.25, 2.27,
                       2.3, 2.4, 2.5, 3.0]) * (Tc / ISING_TC)
    L = 40
    N_L = L * L
    coef = q / (q - 1.0)   # q/(q-1): 2 (Ising), 3/2 (q=3), 4/3 (q=4)
    lhs, rhs, cluster_mean, config_mean = [], [], [], []
    for T in T_list:
        r = _cached_run('wolff', L, float(T), sweeps=50000, therm=10000,
                        overlap_step=2, seed=42, q=q)
        u2_config = r['observables'].get('config_overlap', {}).get('mean', 0.0)
        u2_cluster = r['observables'].get('overlap', {}).get('mean', 0.0)
        chi_over_N = r['observables'].get('cluster_size', {}).get('mean', 0.0) / N_L
        lhs.append(u2_config)
        rhs.append(1.0 - 2.0 * chi_over_N + coef * u2_cluster)
        cluster_mean.append(u2_cluster)
        config_mean.append(u2_config)
    lhs = np.array(lhs)
    rhs = np.array(rhs)
    residual = lhs - rhs
    with plt.rc_context(_PAPER_RC):
        fig, (ax_res, ax_val) = plt.subplots(1, 2, figsize=(7.0, 3.0))
        ax_res.plot(T_list, residual, 'o-', color=_OKABE['vermillion'], ms=3, lw=1.0)
        ax_res.axhline(0, ls='--', color='gray', lw=0.8, alpha=0.7)
        ax_res.axvline(Tc, ls=':', color='gray', lw=0.8, alpha=0.5)
        ax_res.set_xlabel(r'$T$')
        ax_res.set_ylabel(r'residual')
        ax_res.tick_params(direction='in', top=True, right=True)
        _panel_letter(ax_res, 'a')
        ax_val.plot(T_list, config_mean, 'o-', color=_OKABE['blue'], ms=3, lw=1.0,
                    label=r'$P_{\mathrm{coin}}$ (measured)')
        ax_val.plot(T_list, rhs, 's--', color=_OKABE['vermillion'], ms=3, lw=1.0,
                    label=r'$1-2\chi/N + %.3f\,\langle\mathcal{R}^W\rangle$' % coef)
        ax_val.plot(T_list, cluster_mean, '^:', color=_OKABE['bluish_green'], ms=3, lw=1.0,
                    label=r'$\langle\mathcal{R}^W\rangle$')
        ax_val.axvline(Tc, ls=':', color='gray', lw=0.8, alpha=0.5)
        ax_val.set_xlabel(r'$T$')
        ax_val.set_ylabel(r'overlap')
        ax_val.legend(fontsize=6.5, loc='best')
        ax_val.tick_params(direction='in', top=True, right=True)
        _panel_letter(ax_val, 'b')
        fig.subplots_adjust(wspace=0.34)
        outpath = Path(f'pics/paper/overlap_self_consistency{overlap_suffix(q)}.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


def plot_overlap_rw1_temperature(q: int = 2):
    """Wolff Nt=1 cluster overlap vs temperature (paper format).

    ⟨𝓡^W(1)⟩ has no settled FSS exponent (the mean-overlap theory leaves Nt=1
    open). Instead of forcing an exponent, probe Nt=1's critical behavior directly:
    ⟨𝓡^W(1)⟩ drops steeply across Tc (order-parameter-like) and Var(𝓡^W(1)) peaks
    at Tc and sharpens with L (susceptibility-like) — the phase-transition signature."""
    Tc = overlap_tc(q)
    T_list = overlap_t_list(q)
    L_list = [64, 128]
    colors = [_OKABE['blue'], _OKABE['vermillion']]
    markers = ['o', 's']
    with plt.rc_context(_PAPER_RC):
        fig, (ax_mean, ax_var) = plt.subplots(1, 2, figsize=(7.0, 3.0))
        for i, L in enumerate(L_list):
            results = _cached_sweep('wolff', T_list, L, sweeps=20000, therm=5000,
                                    overlap_step=1, seed=42, q=q)
            T_arr = np.array([r['T'] for r in results])
            mean_ov = np.array([r['observables'].get('overlap', {}).get('mean', 0.0)
                                for r in results])
            var_ov = np.array([r['observables'].get('overlap', {}).get('variance', 0.0)
                               for r in results])
            ax_mean.plot(T_arr, mean_ov, f'{markers[i]}-', color=colors[i],
                         ms=3, lw=1.0, label=f'$L={L}$', alpha=0.85)
            ax_var.plot(T_arr, var_ov, f'{markers[i]}-', color=colors[i],
                        ms=3, lw=1.0, label=f'$L={L}$', alpha=0.85)
        for ax in (ax_mean, ax_var):
            ax.axvline(Tc, ls='--', color='gray', lw=0.8, alpha=0.7)
            ax.set_xlim(float(T_list.min()) * 0.85, float(T_list.max()) * 1.05)
            ax.tick_params(direction='in', top=True, right=True)
        ax_mean.set_xlabel(r'$T$')
        ax_mean.set_ylabel(r'$\langle\mathcal{R}^W(1)\rangle$')
        ax_mean.legend(fontsize=7)
        ax_var.set_xlabel(r'$T$')
        ax_var.set_ylabel(r'$\mathrm{Var}(\mathcal{R}^W(1))$')
        ax_var.set_yscale('log')
        _panel_letter(ax_mean, 'a')
        _panel_letter(ax_var, 'b')
        fig.subplots_adjust(wspace=0.34)
        outpath = Path(f'pics/paper/overlap_rw1_temperature{overlap_suffix(q)}.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


def plot_overlap_schematic(q: int = 2):
    """§4 schematic (paper format): definitions of P_coin and ⟨𝓡^W⟩ (Wolff, Ising).

    (a) Configuration overlap P_coin: two spin configs at t and t+Nt; unchanged
        spins ringed in green; P_coin = (#unchanged)/N.
    (b) Wolff cluster overlap ⟨𝓡^W⟩: seed site (★), its FK cluster at t (blue) and
        at t+Nt (red), intersection (purple); ⟨𝓡^W⟩ = |C_t ∩ C_{t+Nt}|/N.
    Synthetic deterministic cartoon (no MC)."""
    rng = np.random.default_rng(7)
    n = 6
    cfg_t = rng.choice([-1, 1], size=(n, n))
    flip = rng.random((n, n)) < 0.30
    cfg_tn = cfg_t.copy()
    cfg_tn[flip] *= -1
    same = (cfg_t == cfg_tn)
    pcoin = same.sum() / (n * n)

    def draw_spins(ax, cfg, x0, mask):
        for i in range(n):
            for j in range(n):
                c = _OKABE['blue'] if cfg[i, j] > 0 else _OKABE['vermillion']
                ax.add_patch(mpatches.Rectangle((x0 + j, n - 1 - i), 1, 1,
                                                facecolor=c, edgecolor='white', lw=0.8))
                if mask[i, j]:
                    ax.add_patch(mpatches.Circle((x0 + j + 0.5, n - 1 - i + 0.5), 0.16,
                                                 facecolor='none', edgecolor=_OKABE['bluish_green'], lw=1.8))

    seed = (n // 2, n // 2)
    ii, jj = np.indices((n, n))
    Ct = ((ii - seed[0]) ** 2 + (jj - seed[1]) ** 2) <= 2.2 ** 2
    Ctn = ((ii - (seed[0] + 1)) ** 2 + (jj - (seed[1] + 1)) ** 2) <= 2.2 ** 2

    with plt.rc_context(_PAPER_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
        # (a) P_coin
        draw_spins(ax_a, cfg_t, 0, same)
        draw_spins(ax_a, cfg_tn, n + 1.5, same)
        ax_a.annotate('', xy=(n + 1.3, n / 2), xytext=(n + 0.1, n / 2),
                      arrowprops=dict(arrowstyle='->', lw=1.2))
        ax_a.text(n + 0.7, n / 2 + 0.7, r'$N_t$', ha='center', fontsize=9)
        ax_a.text(-0.2, -1.3,
                  r'(a)  $P_{\mathrm{coin}}=\dfrac{\#\,\mathrm{unchanged}}{N}='
                  + f'{pcoin:.2f}$', fontsize=9.5, va='top')
        ax_a.set_xlim(-0.5, 2 * n + 2)
        ax_a.set_ylim(-2.2, n + 1)
        ax_a.set_aspect('equal')
        ax_a.axis('off')
        # (b) ⟨𝓡^W⟩
        for i in range(n):
            for j in range(n):
                if Ct[i, j] and Ctn[i, j]:
                    c = _OKABE['purple']      # intersection
                elif Ct[i, j]:
                    c = _OKABE['blue']      # C_t only
                elif Ctn[i, j]:
                    c = _OKABE['vermillion']      # C_{t+Nt} only
                else:
                    c = '#EEEEEE'      # background
                ax_b.add_patch(mpatches.Rectangle((j, n - 1 - i), 1, 1,
                                                  facecolor=c, edgecolor='white',
                                                  lw=0.8, alpha=0.9))
        ax_b.plot(seed[1] + 0.5, n - 1 - seed[0] + 0.5, marker='*', color='gold',
                  markersize=13, markeredgecolor='black', zorder=5)
        ax_b.text(-0.2, -1.3,
                  r'(b)  $\langle\mathcal{R}^W\rangle=\dfrac{|C_t\cap C_{t+N_t}|}{N}$',
                  fontsize=9.5, va='top')
        leg = [mpatches.Patch(color=_OKABE['blue'], label=r'$C_t\setminus C_{t+N_t}$'),
               mpatches.Patch(color=_OKABE['vermillion'], label=r'$C_{t+N_t}\setminus C_t$'),
               mpatches.Patch(color=_OKABE['purple'], label=r'$C_t\cap C_{t+N_t}$')]
        ax_b.legend(handles=leg, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
        ax_b.set_xlim(-0.5, n + 0.5)
        ax_b.set_ylim(-2.2, n + 1)
        ax_b.set_aspect('equal')
        ax_b.axis('off')
        fig.subplots_adjust(wspace=0.15)
        outpath = Path('pics/paper/overlap_schematic.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


def plot_overlap_schematic_cb(q: int = 2):
    """Colorblind-friendly variant of plot_overlap_schematic: adds ↑/↓ spin
    markers in panel (a) and t / t+N_t time labels in panel (b).  Same synthetic
    cartoon, output _cb.png (alongside the original _schematic.png)."""
    rng = np.random.default_rng(7)
    n = 6
    cfg_t = rng.choice([-1, 1], size=(n, n))
    flip = rng.random((n, n)) < 0.30
    cfg_tn = cfg_t.copy()
    cfg_tn[flip] *= -1
    same = (cfg_t == cfg_tn)
    pcoin = same.sum() / (n * n)

    def draw_spins(ax, cfg, x0, mask):
        for i in range(n):
            for j in range(n):
                c = _OKABE['blue'] if cfg[i, j] > 0 else _OKABE['vermillion']
                ax.add_patch(mpatches.Rectangle((x0 + j, n - 1 - i), 1, 1,
                                                facecolor=c, edgecolor='white', lw=0.8))
                # colorblind: ↑/↓ arrow overlays colour differentiation
                # (zorder=2: above patch, below green circle at zorder=5)
                arrow = r'$\uparrow$' if cfg[i, j] > 0 else r'$\downarrow$'
                ax.text(x0 + j + 0.5, n - 1 - i + 0.5, arrow,
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white', zorder=2,
                        path_effects=[patheffects.withStroke(linewidth=1.0,
                                                             foreground='black')])
                if mask[i, j]:
                    ax.add_patch(mpatches.Circle((x0 + j + 0.5, n - 1 - i + 0.5), 0.16,
                                                 facecolor='none', edgecolor=_OKABE['bluish_green'],
                                                 lw=1.8, zorder=5))

    seed = (n // 2, n // 2)
    ii, jj = np.indices((n, n))
    Ct = ((ii - seed[0]) ** 2 + (jj - seed[1]) ** 2) <= 2.2 ** 2
    Ctn = ((ii - (seed[0] + 1)) ** 2 + (jj - (seed[1] + 1)) ** 2) <= 2.2 ** 2

    with plt.rc_context(_PAPER_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
        # (a) P_coin
        draw_spins(ax_a, cfg_t, 0, same)
        draw_spins(ax_a, cfg_tn, n + 1.5, same)
        ax_a.annotate('', xy=(n + 1.3, n / 2), xytext=(n + 0.1, n / 2),
                      arrowprops=dict(arrowstyle='->', lw=1.2))
        ax_a.text(n + 0.7, n / 2 + 0.7, r'$N_t$', ha='center', fontsize=9)
        ax_a.text(-0.2, -1.3,
                  r'(a)  $P_{\mathrm{coin}}=\dfrac{\#\,\mathrm{unchanged}}{N}='
                  + f'{pcoin:.2f}$', fontsize=9.5, va='top')
        ax_a.set_xlim(-0.5, 2 * n + 2)
        ax_a.set_ylim(-2.2, n + 1)
        ax_a.set_aspect('equal')
        ax_a.axis('off')
        # (b) ⟨𝓡^W⟩ — colorblind: "t" / "t+N_t" labels supplement legend
        for i in range(n):
            for j in range(n):
                if Ct[i, j] and Ctn[i, j]:
                    c, h = _OKABE['purple'], '+++'      # intersection: grid hatch
                elif Ct[i, j]:
                    c, h = _OKABE['blue'], '|||'      # C_t only: vertical hatch
                elif Ctn[i, j]:
                    c, h = _OKABE['vermillion'], '---'      # C_{t+N_t} only: horizontal hatch
                else:
                    c, h = '#EEEEEE', None
                ax_b.add_patch(mpatches.Rectangle((j, n - 1 - i), 1, 1,
                                                  facecolor=c, edgecolor='white',
                                                  lw=0.8, alpha=0.9))
                if h:   # colorblind hatch overlay (dark lines, keeps white cell border)
                    ax_b.add_patch(mpatches.Rectangle((j, n - 1 - i), 1, 1,
                                                      facecolor='none', edgecolor='#222222',
                                                      hatch=h, lw=0, alpha=1.0, zorder=2))
        # Two independent Wolff seed sites (★): C_t grows from (3,3), C_{t+N_t}
        # from (4,4) — 𝓡^W does NOT require a shared seed (each sweep re-picks a
        # fresh random seed; the theory's shared i_0 is only a derivation device).
        ax_b.plot([seed[1] + 0.5, seed[1] + 1.5],
                  [n - 1 - seed[0] + 0.5, n - 1 - (seed[0] + 1) + 0.5],
                  marker='*', color='gold', linestyle='',
                  markersize=13, markeredgecolor='black', zorder=6)
        # --- cell-category labels (colourblind: text disambiguates colour) ---
        blue_only = np.argwhere(Ct & ~Ctn)
        red_only  = np.argwhere(Ctn & ~Ct)
        purple    = np.argwhere(Ct & Ctn)

        def _cell_xy(idx):
            i, j = idx
            return (j + 0.5, n - 1 - i + 0.5)

        t_xy   = _cell_xy(blue_only[0])                 # blue  → "t"
        tn_xy  = _cell_xy(red_only[0])                  # red   → "t+N_t"
        bth_xy = _cell_xy(purple[len(purple) // 2])     # purple → "both"

        ax_b.annotate(r'$t$', xy=t_xy,
                      fontsize=9, fontweight='bold', color=_OKABE['blue'], ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=_OKABE['blue'], alpha=0.85))
        ax_b.annotate(r'$t{+}N_t$', xy=tn_xy,
                      fontsize=9, fontweight='bold', color=_OKABE['vermillion'], ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=_OKABE['vermillion'], alpha=0.85))
        ax_b.annotate(r'$\mathrm{both}$', xy=bth_xy,
                      fontsize=8, fontweight='bold', color=_OKABE['purple'], ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=_OKABE['purple'], alpha=0.85))
        ax_b.text(-0.2, -1.3,
                  r'(b)  $\langle\mathcal{R}^W\rangle=\dfrac{|C_t\cap C_{t+N_t}|}{N}$',
                  fontsize=9.5, va='top')
        leg = [mpatches.Patch(color=_OKABE['blue'], label=r'$C_t\;(t)$'),
               mpatches.Patch(color=_OKABE['vermillion'], label=r'$C_{t+N_t}\;(t{+}N_t)$'),
               mpatches.Patch(color=_OKABE['purple'], label=r'$C_t\cap C_{t+N_t}$')]
        ax_b.legend(handles=leg, fontsize=6.5, loc='upper center',
                    bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
        ax_b.set_xlim(-0.5, n + 0.5)
        ax_b.set_ylim(-2.2, n + 1)
        ax_b.set_aspect('equal')
        ax_b.axis('off')
        fig.subplots_adjust(wspace=0.15)
        outpath = Path('pics/paper/overlap_schematic_cb.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


def _r2_on_axis(L_arr, y_arr, exponent):
    """Coefficient of determination R^2 for a linear fit of y vs L^{-exponent}."""
    x = L_arr ** (-exponent)
    coeffs = np.polyfit(x, y_arr, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _r2_mean_std(d, which, exponent):
    """R^2 (mean ± std across seeds) for observable 'mean' or 'var' of a dataset
    dict vs L^{-exponent}. Returns (mean, std)."""
    L_arr = d['L']
    ps = d['mean_ps'] if which == 'mean' else d['var_ps']
    vals = [_r2_on_axis(L_arr, row, exponent) for row in ps]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def _slope_mean_std(d, which):
    """Effective exponent (mean ± std across seeds) from a per-seed log-log fit
    of observable 'mean' or 'var' vs L: e_eff = -<slope>, slope = polyfit(log L,
    log row, 1)[0] per seed. Mirrors _r2_mean_std but returns the fitted exponent
    (with uncertainty), not R^2 — the field-standard point estimate."""
    L_arr = d['L']
    ps = d['mean_ps'] if which == 'mean' else d['var_ps']
    slopes = []
    for row in ps:
        s, _ = np.polyfit(np.log(L_arr), np.log(row), 1)
        slopes.append(-float(s))
    m = float(np.mean(slopes))
    s = float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0
    return m, s


def plot_overlap_nt_convergence(q: int = 2):
    r"""N_t-convergence corroboration (Wolff ⟨𝓡^W⟩ at Tc, Ising only): as the
    separation N_t grows, the measured scaling of ⟨𝓡^W⟩ approaches the derived
    independent limit (χ/N)^2 ~ L^{-2η}. Two panels:
      (a) effective exponent e_eff(N_t) from a log-log ⟨𝓡^W⟩ vs L fit, rising from
          the short-time ~0.42 toward the analytic 2η (=0.5 for Ising);
      (b) R^2 of the analytic L^{-2η} collapse axis, rising toward 1.
    N_t = {1,2,10,50,100,200,400,1000}. The monotone approach to the derived
    values corroborates the (χ/N)^2 derivation; it is NOT a claim that a fixed
    N_t 'reaches' the limit. (Wolff, q=2 Ising only, per scope.)"""
    assert q == 2, 'N_t convergence figure is Ising-only (q=2) by scope'
    Nt_list = (1, 2, 8, 21, 55, 100, 233, 400, 987)   # canonical ~log-spaced set
    datasets = _merge_L(
        _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=Nt_list),
        _collect_wolff_overlap_fss(q=q, L_list=(512,), Nt_list=Nt_list),
    )
    indep = overlap_indep_exp(q)   # 2η = 0.5 (Ising)
    nts, e_eff, e_eff_err, r2s = [], [], [], []
    for Nt in Nt_list:
        d = datasets[Nt]
        L_arr = d['L']
        mean_arr = d['mean']
        # Per-seed effective exponents → mean ± std (slope spread across seeds).
        slopes = []
        for row in d['mean_ps']:                       # one seed across L
            s, _ = np.polyfit(np.log(L_arr), np.log(row), 1)
            slopes.append(-float(s))
        nts.append(Nt)
        e_eff.append(float(np.mean(slopes)))
        e_eff_err.append(float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0)
        r2s.append(_r2_on_axis(L_arr, mean_arr, indep))   # R^2 on seed-averaged curve
    nts = np.array(nts, dtype=float)
    e_eff = np.array(e_eff)
    e_eff_err = np.array(e_eff_err)
    with plt.rc_context(_PAPER_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
        ax1.errorbar(nts, e_eff, yerr=e_eff_err, fmt='o-', color=_OKABE['blue'],
                     ms=5, lw=1.2, elinewidth=0.7, capsize=2, zorder=3)
        ax1.axhline(indep, color=_OKABE['vermillion'], ls='--', lw=1.0,
                    label=rf'derived $(\chi/N)^2 \sim L^{{-{indep:g}}}$')
        ax1.axhline(0.42, color=_OKABE['bluish_green'], ls=':', lw=1.0,
                    label=r'lit.\ short-time $0.42$')
        ax1.set_xscale('log')
        ax1.set_xlabel(r'separation $N_t$')
        ax1.set_ylabel(r'effective exponent $e_{\mathrm{eff}}$')
        ax1.set_title(r'(a) $\langle\mathcal{R}^W\rangle \sim L^{-e_{\mathrm{eff}}}$', fontsize=9)
        ax1.legend(fontsize=7, loc='center right')
        ax1.grid(True, alpha=0.3)
        ax2.plot(nts, r2s, 's-', color=_OKABE['blue'], ms=5, lw=1.2, zorder=3)
        ax2.axhline(1.0, color=_OKABE['vermillion'], ls='--', lw=1.0, label=r'perfect collapse')
        ax2.set_xscale('log')
        ax2.set_xlabel(r'separation $N_t$')
        ax2.set_ylabel(r'$R^2$ on $L^{-2\eta}$ axis')
        ax2.set_title(r'(b) agreement with $(\chi/N)^2$ scaling', fontsize=9)
        ax2.legend(fontsize=7, loc='lower right')
        ax2.grid(True, alpha=0.3)
        fig.suptitle(r'Wolff $\langle\mathcal{R}^W\rangle$ at $T_c$ (Ising): approach to the '
                     r'derived $(\chi/N)^2$ limit as $N_t$ grows', fontsize=9.5, y=1.0)
        fig.tight_layout()
        outpath = Path('pics/paper/overlap_nt_convergence.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)
        print('  Nt   e_eff ± std   R^2(L^-2eta)')
        for n, e, de, r in zip(nts, e_eff, e_eff_err, r2s):
            print(f'  {int(n):>4}  {e:.4f}±{de:.4f}  {r:.6f}')


def plot_overlap_rw_vs_nt(q: int = 2):
    r"""⟨𝓡^W⟩ vs separation N_t at Tc (Ising): dense curve N_t=1…1000 with SEM
    error bars across SEEDS, L∈{64,128,256} overlaid.

    Each curve decays from the short-time regime (boosted above the independent
    limit by residual two-step bond correlation) toward the analytic asymptote
    (χ/N)^2 ~ L^{-2η} (FK: χ = β⟨|C|⟩), drawn per-L as a color-matched dashed
    line. Uses the dense NT_FINE set; small L only (overlap is O(|Nt|·N)/sweep).
    Reduced sweeps (small-L corroboration curve, not the headline FSS)."""
    assert q == 2, '⟨R^W⟩ vs N_t figure is Ising-only (q=2) by scope'
    # (L, Nt_set, sweeps, color): small L dense (NT_FINE), large L uses NT_LARGE
    # (the canonical large-L run; overlap is O(|Nt|·N)/sweep so |Nt| stays small).
    # (L, Nt_set, sweeps, color, all_up): contiguous L=64…4096. Small L dense
    # (NT_FINE), L=512 uses NT_CANONICAL (Phase A cache), L>=1024 NT_LARGE.
    configs = [
        (64,   NT_FINE,     20000, '#1b9e77', True),
        (128,  NT_FINE,     20000, '#36a3c4', True),
        (256,  NT_FINE,     20000, '#7570b3', True),
        (512,  NT_CANONICAL,20000, '#e7298a', True),
        (1024, NT_LARGE,    20000, '#66a61e', True),
        (2048, NT_LARGE,    10000, _OKABE['yellow'], True),
        (4096, NT_LARGE,    10000, _OKABE['orange'], True),
    ]
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        for L, nt_set, sweeps, color, all_up in configs:
            ds = _collect_wolff_overlap_fss(q=q, L_list=(L,), Nt_list=nt_set,
                                             sweeps=sweeps, nt_run=nt_set, all_up=all_up)
            nts = np.array(nt_set, dtype=float)
            means = np.array([ds[n]['mean'][0] for n in nt_set])
            sems = np.array([ds[n]['mean_sem'][0] for n in nt_set])
            # Independent-limit asymptote (χ/N)^2 = (⟨|C|⟩/N)^2 — the paper's χ/N
            # IS ⟨|C|⟩/N (FK identity with β absorbed; verified by P_coin^W(1)=1−χ/N).
            # cluster_only=True matches the Wolff .npz cache key (avoids a re-run).
            th = _therm_for_L(L)
            cs = np.mean([_collect_overlap_series('wolff', q, L, s, nt_set, sweeps, th,
                                                  all_up, cluster_only=True)['cluster_size_mean']
                          for s in SEEDS])
            chi_over_N = cs / (L * L)
            ax.plot(nts, means, 'o-', color=color, mfc='white', mec=color, ms=3,
                    lw=1.0, label=rf'$L={L}$', zorder=3)
            ax.errorbar(nts, means, yerr=sems, fmt='none', ecolor=color,
                        elinewidth=1.2, capsize=3, zorder=4)
            ax.axhline(chi_over_N ** 2, color=color, ls='--', lw=0.9, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'separation $N_t$')
        ax.set_ylabel(r'$\langle\mathcal{R}^W(N_t)\rangle$')
        ax.legend(fontsize=6.5, loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1)
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        outpath = Path('pics/paper/overlap_rw_vs_nt.png')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        _paper_savefig(fig, outpath)


def diagonal_r2_table():
    r"""Compact 'diagonal' R^2 (the matching axis per Nt) for all overlap
    observables × q∈{2,3,4}, recomputed from the cached FSS data (no new sims).
    Prints LaTeX table rows for Supplementary §C.

    Diagonal (the axis each Nt-row should collapse on, = highest-R^2 column):
      Wolff ⟨R^W⟩ & Var(R^W): Nt=1,2 -> L^{-0.42} (short-time); Nt=100,400 -> L^{-2η}.
      SW Var(P_coin):         Nt=1 -> L^{-(2-γ/ν)} (exact χ/N); Nt=2 -> L^{-ψ^SW}; Nt=100 -> L^{-2η}.
      Metropolis Var(P_coin): reference, no analytic axis (R^2 on L^{-2η} shown, uniformly low)."""
    print('\n=== R^2 diagonal table (compact, all q) — for Supplementary §C ===')
    print(r'observable (algorithm) & $q$ & $N_t{=}1$ & $N_t{=}2$ & $N_t{=}100$ & $N_t{=}400$ \\')
    for q in (2, 3, 4):
        indep = overlap_indep_exp(q)        # 2η
        naive = overlap_naive_a_exp(q)      # 2-γ/ν  (SW Nt=1 exact axis)
        swlit = sw_lit_exp(q)               # ψ^SW   (SW Nt=2 lit axis)
        # --- Wolff (mean ⟨R^W⟩ and Var(R^W) share one sim/dataset) ---
        collects = [
            _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(1, 2, 100)),
            _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(400,)),
            _collect_wolff_overlap_fss(q=q, L_list=(512,), Nt_list=(1, 2, 100, 400), all_up=True),
        ]
        if q == 2:
            collects.append(_collect_wolff_overlap_fss(q=q, L_list=(1024,),
                                                       Nt_list=(1, 2, 100, 400), all_up=True))
            collects.append(_collect_wolff_overlap_fss(q=q, L_list=(2048, 4096),
                                                       Nt_list=(1, 2, 100, 400), sweeps=10000, all_up=True))
        ds = _merge_L(*collects)

        def cell(d, which, exp):
            m, s = _r2_mean_std(d, which, exp)
            return f'{m:.6f}' if s == 0.0 else f'{m:.6f}$\\pm${s:.6f}'

        for label, which in [(r'$\langle\mathcal{R}^W\rangle$, Wolff', 'mean'),
                             (r'$\mathrm{Var}(\mathcal{R}^W)$, Wolff', 'var')]:
            cells = [label, str(q),
                     cell(ds[1], which, 0.42), cell(ds[2], which, 0.42),
                     cell(ds[100], which, indep), cell(ds[400], which, indep)]
            print(' & '.join(cells) + r' \\')
        # --- SW Var(P_coin): Nt=1 -> naive ; Nt=2 -> ψ^SW ; Nt=100 -> 2η (no 400) ---
        dssw = _merge_L(
            _collect_config_overlap_fss('swendsen_wang', q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(1, 2, 100)),
            _collect_config_overlap_fss('swendsen_wang', q=q, L_list=(512,), Nt_list=(1, 2, 100)),
        )
        print(' & '.join([r'$\mathrm{Var}(P_{\mathrm{coin}})$, SW', str(q),
                          cell(dssw[1], 'var', naive), cell(dssw[2], 'var', swlit),
                          cell(dssw[100], 'var', indep), '---']) + r' \\')
        # --- Metropolis Var(P_coin): reference, R^2 on L^{-2η} (no analytic axis) ---
        dsm = _collect_config_overlap_fss('metropolis', q=q, L_list=[16, 32, 64, 128], Nt_list=(1, 2, 100))
        print(' & '.join([r'$\mathrm{Var}(P_{\mathrm{coin}})$, Metropolis', str(q),
                          cell(dsm[1], 'var', indep), cell(dsm[2], 'var', indep),
                          cell(dsm[100], 'var', indep), '---']) + r' \\')


def fitted_exponent_table():
    r"""Fitted effective exponents (per-seed log-log slope ± spread) for the
    headline overlap observables at Tc, recomputed from cached FSS data (no new
    sims). Prints LaTeX rows: observable & q & separation & fitted e (this work)
    & analytic/literature. This is the field-standard exponent point-estimate
    (with uncertainty) that the R^2 collapse-test grids do not provide; it
    complements Table 1 (which lists only analytic/literature values)."""
    print('\n=== fitted effective exponents (per-seed log-log slope ± spread) ===')
    print(r'observable (algorithm) & $q$ & separation & fitted $e$ (this work) & analytic/literature \\')

    def frac(x):
        return {0.5: r'$1/2$', 0.25: r'$1/4$', 8.0 / 15: r'$8/15$', 4.0 / 15: r'$4/15$'}.get(round(x, 4), f'${x:g}$')

    for q in (2, 3, 4):
        indep = overlap_indep_exp(q)      # 2η   (Nt→∞ analytic)
        naive = overlap_naive_a_exp(q)    # 2-γ/ν (SW Nt=1 exact axis)
        swlit = sw_lit_exp(q)             # ψ^SW  (SW Nt=2 literature)
        # --- Wolff (⟨R^W⟩ and Var(R^W) share one dataset) ---
        collects = [_collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(1, 2, 100)),
                    _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(400,)),
                    _collect_wolff_overlap_fss(q=q, L_list=(512,), Nt_list=(1, 2, 100, 400), all_up=True)]
        if q == 2:
            collects += [_collect_wolff_overlap_fss(q=q, L_list=(1024,), Nt_list=(1, 2, 100, 400), all_up=True),
                         _collect_wolff_overlap_fss(q=q, L_list=(2048, 4096), Nt_list=(1, 2, 100, 400), sweeps=10000, all_up=True)]
        ds = _merge_L(*collects)

        def cell(d, which):
            m, s = _slope_mean_std(d, which)
            return f'${m:.3f} \\pm {s:.3f}$'

        for label, which in [(r'$\langle\mathcal{R}^W\rangle$, Wolff', 'mean'),
                             (r'$\mathrm{Var}(\mathcal{R}^W)$, Wolff', 'var')]:
            print(' & '.join([label, str(q), r'$N_t{=}1$', cell(ds[1], which), '---']) + r' \\')
            print(' & '.join([label, str(q), r'$N_t{=}2$', cell(ds[2], which), r'$0.42$ (lit.)']) + r' \\')
            print(' & '.join([label, str(q), r'$N_t{=}100$', cell(ds[100], which),
                              frac(indep) + r' ($2\eta$, analytic)']) + r' \\')
        # --- SW Var(P_coin): Nt=1 -> exact χ/N ; Nt=2 -> ψ^SW ; Nt=100 -> 2η ---
        dssw = _merge_L(
            _collect_config_overlap_fss('swendsen_wang', q=q, L_list=[16, 32, 64, 128, 256], Nt_list=(1, 2, 100)),
            _collect_config_overlap_fss('swendsen_wang', q=q, L_list=(512,), Nt_list=(1, 2, 100)),
        )
        print(' & '.join([r'$\mathrm{Var}(P_{\mathrm{coin}})$, SW', str(q), r'$N_t{=}1$',
                          cell(dssw[1], 'var'), frac(naive) + r' (analytic)']) + r' \\')
        print(' & '.join([r'$\mathrm{Var}(P_{\mathrm{coin}})$, SW', str(q), r'$N_t{=}2$',
                          cell(dssw[2], 'var'), rf'$\psi^{{\mathrm{{SW}}}}={swlit:g}$ (lit.)']) + r' \\')
        print(' & '.join([r'$\mathrm{Var}(P_{\mathrm{coin}})$, SW', str(q), r'$N_t{=}100$',
                          cell(dssw[100], 'var'), frac(indep) + r' ($2\eta$, analytic)']) + r' \\')


def plot_overlap_mean_fss_fullnt(q: int = 2):
    r"""Wolff ⟨𝓡^W⟩ 8-N_t collapse grid (Supplementary §C): N_t={1,2,10,50,100,200,400,1000}.
    Shows the gradual column-shift from the short-time L^{-0.42} axis (small N_t) to the
    independent-limit L^{-2η} axis (large N_t) — the grid-form of the N_t-convergence
    figure. Same 3 exponent columns as the §5 figure."""
    Nt8 = (1, 2, 10, 50, 100, 200, 400, 1000)
    collects = [
        _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=Nt8),
        _collect_wolff_overlap_fss(q=q, L_list=(512,), Nt_list=Nt8, all_up=True),
    ]
    if q == 2:
        collects.append(_collect_wolff_overlap_fss(q=q, L_list=(1024,), Nt_list=Nt8, all_up=True))
        collects.append(_collect_wolff_overlap_fss(q=q, L_list=(2048, 4096), Nt_list=Nt8, sweeps=10000, all_up=True))
    datasets = _merge_L(*collects)
    naive = overlap_naive_a_exp(q)
    indep = overlap_indep_exp(q)
    exp_cols = [(naive, r'$L^{-%.3f}$ ($\chi/N$ naive)' % naive),
                (0.42, r'$L^{-0.42}$ (lit., $N_t=2$)'),
                (indep, r'$L^{-%.3f}$ (indep. $(\chi/N)^2$)' % indep)]
    ylab = lambda Nt: r'$\langle\mathcal{R}^W(N_t=%s)\rangle$' % _nt_tex(Nt)
    _collapse_grid(datasets, list(Nt8), exp_cols, q, 'mean', ylab,
                   f'pics/paper/overlap_mean_fss_fullNt{overlap_suffix(q)}.png',
                   suptitle=r'FSS of $\langle\mathcal{R}^W\rangle$ at $T_c$ (full $N_t$)')


def plot_overlap_variance_fss_fullnt(q: int = 2):
    r"""Wolff Var(𝓡^W) 8-N_t collapse grid (Supplementary §C). Same axes as the mean
    full-N_t grid; mirrors the column-shift from L^{-0.42} to L^{-2η}."""
    Nt8 = (1, 2, 10, 50, 100, 200, 400, 1000)
    collects = [
        _collect_wolff_overlap_fss(q=q, L_list=[16, 32, 64, 128, 256], Nt_list=Nt8),
        _collect_wolff_overlap_fss(q=q, L_list=(512,), Nt_list=Nt8, all_up=True),
    ]
    if q == 2:
        collects.append(_collect_wolff_overlap_fss(q=q, L_list=(1024,), Nt_list=Nt8, all_up=True))
        collects.append(_collect_wolff_overlap_fss(q=q, L_list=(2048, 4096), Nt_list=Nt8, sweeps=10000, all_up=True))
    datasets = _merge_L(*collects)   # shares the mean grid's sim/cache
    naive = overlap_naive_a_exp(q)
    indep = overlap_indep_exp(q)
    exp_cols = [(naive, r'$L^{-%.3f}$ ($\chi/N$ naive)' % naive),
                (0.42, r'$L^{-0.42}$ (lit., $N_t=2$)'),
                (indep, r'$L^{-%.3f}$ (indep. $(\chi/N)^2$)' % indep)]
    ylab = lambda Nt: r'$\mathrm{Var}\,\mathcal{R}^W(N_t=%s)$' % _nt_tex(Nt)
    _collapse_grid(datasets, list(Nt8), exp_cols, q, 'var', ylab,
                   f'pics/paper/overlap_variance_fss_fullNt{overlap_suffix(q)}.png',
                   suptitle=r'FSS of $\mathrm{Var}(\mathcal{R}^W)$ at $T_c$ (full $N_t$)')


def plot_overlap_sw_variance_fss_fullnt(q: int = 2):
    r"""SW Var(P_coin) 8-N_t collapse grid (Supplementary §C): N_t={1,2,10,50,100,200,400,1000}.
    Column-shift from the exact L^{-(2-γ/ν)} axis (N_t=1) through ψ^SW (N_t=2) to the
    independent L^{-2η} axis (large N_t)."""
    Nt8 = (1, 2, 10, 50, 100, 200, 400, 1000)
    sw_collects = [
        _collect_config_overlap_fss('swendsen_wang', q=q, L_list=[16, 32, 64, 128, 256], Nt_list=Nt8),
        _collect_config_overlap_fss('swendsen_wang', q=q, L_list=(512,), Nt_list=Nt8),
    ]
    if q == 2:
        sw_collects.append(_collect_config_overlap_fss('swendsen_wang', q=q, L_list=(1024,), Nt_list=Nt8))
        sw_collects.append(_collect_config_overlap_fss('swendsen_wang', q=q, L_list=(2048,), Nt_list=Nt8, sweeps=10000))
    datasets = _merge_L(*sw_collects)
    nt1 = overlap_naive_a_exp(q)
    sw_lit = sw_lit_exp(q)
    indep = overlap_indep_exp(q)
    exp_cols = [(nt1, r'$L^{-%.3f}$ (exact, $N_t=1$)' % nt1),
                (sw_lit, r'$L^{-%.3f}$ (lit., $N_t=2$)' % sw_lit),
                (indep, r'$L^{-%.3f}$ (indep.)' % indep)]
    ylab = lambda Nt: r'$\mathrm{Var}(P_{\mathrm{coin}})$, $N_t=%s$' % _nt_tex(Nt)
    _collapse_grid(datasets, list(Nt8), exp_cols, q, 'var', ylab,
                   f'pics/paper/overlap_sw_variance_fss_fullNt{overlap_suffix(q)}.png',
                   suptitle=r'FSS of SW $\mathrm{Var}(P_{\mathrm{coin}})$ at $T_c$ (full $N_t$)')


if __name__ == '__main__':
    # --- Lecture-1 (slide) figures: regenerate separately when needed ---
    # kagome_kondo_ising(); kagome_kondo_ising_v2()
    # plot_metropolis_mt(); plot_thermalization(); plot_sw_mt(); plot_wolff_mt()
    # plot_dynamic_exponent()
    # plot_heisenberg_metropolis_mt(); plot_heisenberg_wolff_mt()

    # --- Paper figures: §4 schematic + §5 verification, q ∈ {2,3,4} → pics/paper/ ---
    for q in (2, 3, 4):
        print(f'===== paper figures, q={q} =====', flush=True)
        plot_overlap_mean_fss(q)
        plot_overlap_variance_fss(q)
        plot_overlap_sw_variance_fss(q)
        plot_overlap_metropolis_variance_fss(q)
        plot_overlap_vs_temperature(q)
        plot_overlap_cluster_mean(q)
        plot_overlap_rw1_temperature(q)
        plot_overlap_wolff_chi_relation(q)
        plot_overlap_self_consistency(q)
        plot_overlap_quick_checks(q)
    # --- R5: N_t convergence (Wolff Ising) + full-N_t supplementary grids + R^2 table ---
    plot_overlap_nt_convergence(2)
    plot_overlap_rw_vs_nt(2)   # dense ⟨R^W⟩ vs N_t curve (Ising, small L)
    for q in (2, 3, 4):
        plot_overlap_mean_fss_fullnt(q)
        plot_overlap_variance_fss_fullnt(q)
        plot_overlap_sw_variance_fss_fullnt(q)
    diagonal_r2_table()

    plot_overlap_schematic(2)
    plot_overlap_schematic_cb(2)
    print('All paper figures generated into pics/paper/.', flush=True)
