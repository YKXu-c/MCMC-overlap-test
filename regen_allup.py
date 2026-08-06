"""Local regen with all_up=True (Part D). Re-runs all small/medium-L .npz with
ordered init. SKIPS plot_overlap_sw_variance_fss(2) + fullnt(2) + diagonal_r2_table
— those need the SW large-L .npz from zju (job.phase_b_sw); regenerate them after
the download. Schematics unchanged (skipped)."""
import generatePic as g

print('=== local regen (all_up=True) — SW q=2 FSS + R² table deferred to zju ===', flush=True)
for q in (2, 3, 4):
    print(f'--- q={q} ---', flush=True)
    g.plot_overlap_mean_fss(q)
    g.plot_overlap_variance_fss(q)
    if q != 2:                      # SW q=2 FSS needs zju large-L
        g.plot_overlap_sw_variance_fss(q)
    g.plot_overlap_metropolis_variance_fss(q)
    g.plot_overlap_vs_temperature(q)
    g.plot_overlap_cluster_mean(q)
    g.plot_overlap_rw1_temperature(q)
    g.plot_overlap_wolff_chi_relation(q)
    g.plot_overlap_self_consistency(q)
    g.plot_overlap_quick_checks(q)
g.plot_overlap_nt_convergence(2)
g.plot_overlap_rw_vs_nt(2)
for q in (2, 3, 4):
    g.plot_overlap_mean_fss_fullnt(q)
    g.plot_overlap_variance_fss_fullnt(q)
    if q != 2:                      # SW fullnt q=2 needs zju
        g.plot_overlap_sw_variance_fss_fullnt(q)
# diagonal_r2_table deferred (needs SW q=2 large-L from zju)
print('=== local regen DONE (SW q=2 FSS + R² table pending zju) ===', flush=True)
