"""Phase B driver — run ONE (L, seed) canonical multi-Nt trajectory at q=2.

Called by the PBS job as one process per seed (7 in parallel). Writes the .npz
into overlap_series/ (next to generatePic.py). Usage:
    python3 run_phase_b.py --L 4096 --seed 42 [--algo wolff|swendsen_wang]

Wolff uses cluster_only=True (FSS reads cluster overlap); SW uses cluster_only=False
(config overlap is its primary — needs the bit-packed spin_history_).
"""
import sys
import generatePic as g


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


L = int(_arg('--L'))
seed = int(_arg('--seed'))
algo = _arg('--algo', 'wolff')
SWEEPS = 20000 if L <= 1024 else 10000   # match the figure chunks (L<=1024 default; L>=2048 reduced)
therm = g._therm_for_L(L)   # ramp: 10000 + 1000*log2(L/256)
cluster_only = (algo == 'wolff')   # Wolff cluster-only; SW needs config overlap

print(f'Phase B: algo={algo} q=2 L={L} seed={seed} sweeps={SWEEPS} therm={therm} '
      f'cluster_only={cluster_only} all_up=True Nt=NT_LARGE({g.NT_LARGE})', flush=True)
g._collect_overlap_series(algo, q=2, L=L, seed=seed, Nt_list=g.NT_LARGE,
                          sweeps=SWEEPS, therm=therm, all_up=True, cluster_only=cluster_only)
print(f'DONE {algo} L={L} seed={seed}', flush=True)
