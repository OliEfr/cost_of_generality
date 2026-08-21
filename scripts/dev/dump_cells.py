import json

cells = json.load(open('/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis/experiments/genbias_link_cells.json'))
for c in cells:
    if c['level'] == 'L0':
        continue
    print(f"== {c['task']} {c['level']} n{c['n']}  SR={c['sr']} fail={c.get('n_fail')} "
          f"rej={c['rejection_rate']:.2f} power_lim={c.get('power_limited')}")
    for d, v in c.get('per_dim', {}).items():
        rb = v['rb']; mw = v['mw_p']; kd = v['ks_d']; kp = v['ks_p']
        f = lambda x: 'nan' if x is None else f"{x:+.3f}"
        g = lambda x: 'nan' if x is None else f"{x:.4f}"
        print(f"   {d:8} rb={f(rb)} mw_p={g(mw)} ks_d={f(kd)} ks_p={g(kp)}")
    for k in ('local_rej_rb', 'local_rej_p', 'sr_gen_easy', 'sr_gen_hard',
              'split_fisher_p', 'nnd_rb', 'nnd_p',
              'variant_gen_eval_spearman', 'variant_gen_eval_p'):
        if c.get(k) is not None:
            v = c[k]
            print(f"   {k} = {v:.4f}" if isinstance(v, float) else f"   {k} = {v}")
    if 'logit' in c:
        print('   logit:', {d: (v['beta'], v['p']) for d, v in c['logit'].items()},
              'conv=', c.get('logit_converged'))
    if 'variant_gen_sr' in c:
        print('   variant gen_sr :', c['variant_gen_sr'])
        print('   variant eval_sr:', c['variant_eval_sr'])
