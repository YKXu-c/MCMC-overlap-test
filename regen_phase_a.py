"""Phase A regeneration driver — 7-seed FSS figures with error bars.

Runs only the 7-seed FSS family (the headline collapse grids + full-Nt grids +
Nt-convergence + R² table). The single-seed sanity plots (vs_temperature, etc.)
are unchanged and not re-run here. full-Nt grids + diagonal_r2_table reuse the
headline .npz cache (cache hits → fast)."""
import generatePic as g

print('=== Phase A 7-seed regeneration ===', flush=True)
for q in (2, 3, 4):
    print(f'--- q={q} headline FSS ---', flush=True)
    g.plot_overlap_mean_fss(q)
    g.plot_overlap_variance_fss(q)
    g.plot_overlap_sw_variance_fss(q)
    g.plot_overlap_metropolis_variance_fss(q)

print('--- Nt-convergence (Ising) ---', flush=True)
g.plot_overlap_nt_convergence(2)

print('--- full-Nt grids (reuse .npz cache) ---', flush=True)
for q in (2, 3, 4):
    g.plot_overlap_mean_fss_fullnt(q)
    g.plot_overlap_variance_fss_fullnt(q)
    g.plot_overlap_sw_variance_fss_fullnt(q)

print('--- R² diagonal table ---', flush=True)
g.diagonal_r2_table()
print('=== Phase A 7-seed regeneration DONE ===', flush=True)
