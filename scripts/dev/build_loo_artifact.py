#!/usr/bin/env python3
"""Emit paper/loo_report/artifact/index.html -- the web version of the leave-one-out report.

Every number is pulled from cog.analysis.loo, the same module the PDF uses, so the two cannot
disagree. Figures are referenced as figures/<name>.png and published alongside the page.
"""
from __future__ import annotations

import datetime as dt
import html
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cog.analysis.loo import ARMS, NDEMOS, REMOVE, TASKS, TASK_LABEL, build  # noqa: E402

OUT = REPO / "paper" / "loo_report" / "artifact" / "index.html"

FIGS = {
    1: ("fig01_surface_curves.png",
        "Success rate against demonstration budget for all six arms of each task. "
        "Error bars are Wilson 95&thinsp;% intervals on 200 episodes."),
    2: ("fig02_surface_heatmap.png",
        "The same 108 measurements as a grid. Cell values are success rates over 200 episodes."),
    3: ("fig03_loo_delta_vs_n.png",
        "Leave-one-out delta against demonstration budget. Shaded bands are Newcombe 95&thinsp;% "
        "intervals; the dashed line is no change."),
    4: ("fig04_loo_forest.png",
        "All 54 leave-one-out deltas with their intervals. Faded rows are those whose interval "
        "covers zero."),
    5: ("fig05_slice_dispersion.png",
        "The ten slice success rates behind every cell (dots) and the pooled cell value (bar)."),
    6: ("fig06_t2_stage_funnel.png",
        "Absolute milestone rates per arm and budget, with success overlaid."),
    7: ("fig07_t2_conditional.png",
        "Conditional rate of reaching each milestone given the previous one, computed from joint "
        "counts."),
    8: ("fig08_time_to_success.png",
        "Distribution of the first-success step per arm. Boxes are quartiles, whiskers 1.5&thinsp;IQR."),
    9: ("fig09_t2_progress_magnitudes.png",
        "Continuous progress measures over all 1,200 episodes of each arm. Dashed lines mark the "
        "milestone thresholds."),
    10: ("fig10_generation.png",
         "Demonstration yield per arm: fraction of generation attempts that produced a usable "
         "demonstration, and attempts required."),
}

DATASET_FRAMES = {
    ("T1", "L0"): 82916, ("T1", "L1"): 75366, ("T1", "L2"): 74860, ("T1", "L3b"): 75438,
    ("T1", "AC"): 75294, ("T1", "BC"): 82960,
    ("T2", "L0"): 281987, ("T2", "L1"): 277661, ("T2", "L2"): 270744, ("T2", "L3b"): 269650,
    ("T2", "AC"): 277842, ("T2", "BC"): 271895,
    ("T3", "L0"): 126735, ("T3", "L1"): 127609, ("T3", "L2"): 124472, ("T3", "L3b"): 123934,
    ("T3", "AC"): 126671, ("T3", "BC"): 124093,
}

CAPS = {"T1": 600, "T2": 1200, "T3": 800}

sys.path.insert(0, str(REPO / "scripts" / "dev"))
from build_loo_report import FLAWS, TODOS  # noqa: E402


def esc(x):
    return html.escape(str(x), quote=False)


def rev():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def fig(n):
    name, cap = FIGS[n]
    return (f'<figure class="plate">\n'
            f'  <img src="figures/{name}" alt="Figure {n}. {esc(cap)}" loading="lazy">\n'
            f'  <figcaption><span class="fignum">Figure {n}</span> {cap}</figcaption>\n'
            f'</figure>')


def tbl(head, rows, cls="", note=None):
    th = "".join(f"<th>{c}</th>" for c in head)
    body = "\n".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    n = f'<p class="note">{note}</p>' if note else ""
    return (f'<div class="scroll"><table class="{cls}">\n<thead><tr>{th}</tr></thead>\n'
            f'<tbody>\n{body}\n</tbody></table></div>{n}')


def num(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return '<span class="na">--</span>'
    return f'<span class="n">{x:+.{nd}f}</span>' if sign else f'<span class="n">{x:.{nd}f}</span>'


def build_html():
    d = build()
    today = dt.date.today().isoformat()
    P = []
    A = P.append

    A('<title>Leave-one-out disturbance ablation</title>')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&'
      'family=IBM+Plex+Serif:wght@500;600&display=swap">')
    A(CSS)

    # ---------------- masthead
    A('<header class="masthead">')
    A('  <p class="eyebrow">Quantitative report &middot; simulated manipulation policies</p>')
    A('  <h1>Leave-one-out disturbance ablation</h1>')
    A('  <p class="standfirst">Which single simplification of the full disturbance set changes '
      'policy success rate, measured on three manipulation tasks at six demonstration budgets.</p>')
    A('  <dl class="facts">')
    for k, v in (("cells", "108 trained policies"), ("scored episodes", "21,600"),
                 ("arms per task", "6"), ("budgets", "10 &rarr; 400 demos"),
                 ("generated", today), ("revision", rev())):
        A(f'    <div><dt>{k}</dt><dd>{v}</dd></div>')
    A('  </dl>')
    A('  <p class="scopenote"><strong>Scope.</strong> This document reports measurements only. '
      'It contains no interpretation, no ranking of axes, and no comparison against any other '
      'study or previously published surface.</p>')
    A('</header>')

    # ---------------- toc
    toc = [("s1", "1", "Design and definitions"), ("s2", "2", "Data provenance"),
           ("s3", "3", "Measured surface"), ("s4", "4", "Leave-one-out results"),
           ("s5", "5", "Within-cell dispersion"), ("s6", "6", "T2 milestone instrumentation"),
           ("s7", "7", "Time to success"), ("s8", "8", "Implementation flaws"),
           ("s9", "9", "Open items"), ("sa", "A", "Budget-pooled deltas"),
           ("sb", "B", "Files behind this report")]
    A('<nav class="toc" aria-label="Contents"><ol>')
    for sid, num_, title in toc:
        A(f'<li><a href="#{sid}"><span class="tnum">{num_}</span>{title}</a></li>')
    A('</ol></nav>')

    A('<main>')

    # ---------------- 1
    A('<section id="s1"><h2><span class="snum">1</span>Design and definitions</h2>')
    A('<p class="lede">The full disturbance set is <b>L3b = {A, B, C}</b>. Each leave-one-out arm '
      'removes exactly one axis from that set and keeps the other two. Every delta in this report '
      'is <span class="mono">SR(arm with one axis removed) &minus; SR(L3b)</span> at the same '
      'demonstration budget, so a positive delta means the simplified arm scored higher.</p>')
    rows = [
        ['<b class="arm arm-L0">L0</b>', "none", "&mdash;", "1 flat", "<code>&lt;TASK&gt;-L0</code>"],
        ['<b class="arm arm-L1">L1</b>', "A", "&mdash;", "1 flat", "<code>&lt;TASK&gt;-L1</code>"],
        ['<b class="arm arm-L2">L2</b>', "A, B", "L3b &#92; C &mdash; variant axis removed",
         "1 flat", "<code>&lt;TASK&gt;-L2</code>"],
        ['<b class="arm arm-L3b">L3b</b>', "A, B, C &nbsp;(full set)", "reference for every delta",
         "10 variants", "<code>&lt;TASK&gt;-L3v0<i>s</i></code>"],
        ['<b class="arm arm-AC">AC</b>', "A, C", "L3b &#92; B &mdash; goal / fixture pose removed",
         "10 variants", "<code>&lt;TASK&gt;-ACv0<i>s</i></code>"],
        ['<b class="arm arm-BC">BC</b>', "B, C", "L3b &#92; A &mdash; manipulandum pose removed",
         "10 variants", "<code>&lt;TASK&gt;-BCv0<i>s</i></code>"],
    ]
    A(tbl(["arm", "disturbances active", "leave-one-out meaning", "sub-environments",
           "evaluation environment, slice <i>s</i>"], rows))
    A('<h3>1.1&nbsp; What each axis randomises</h3>')
    A(tbl(["axis", "T1 cup_place", "T2 drawer_stow", "T3 push_target"], [
        ['<b class="ax ax-A">A</b> manipulandum pose', "cup XY over 30&times;40&nbsp;cm, yaw &plusmn;90&deg;",
         "stow-object XY, yaw &plusmn;45&deg;", "puck XY over 12&times;12&nbsp;cm"],
        ['<b class="ax ax-B">B</b> goal / fixture pose', "goal disk XY over 20&times;20&nbsp;cm",
         "cabinet XY &plusmn;5&nbsp;cm, yaw &plusmn;7.5&deg;", "target bearing &plusmn;25&deg;"],
        ['<b class="ax ax-C">C</b> object variant', "10 cup variants (2 sizes &times; 5 colours)",
         "10 box variants", "10 puck variants (radius / height)"],
    ]))
    A('<h3>1.2&nbsp; Evaluation protocol</h3>')
    A(tbl(["property", "value"], [
        ["episodes scored per cell", "<b>200</b> &mdash; 10 slices &times; 20 parallel environments"],
        ["slices per cell", "10 independent processes, one scored batch each"],
        ["warm-up", "1 unscored batch per slice, reset seed 4900+<i>s</i>, outcomes discarded"],
        ["scored reset seed", "5000+<i>s</i> for slice <i>s</i> = 0&hellip;9"],
        ["variant arms", "diagonal &mdash; slice <i>s</i> evaluates sub-environment v0<i>s</i>"],
        ["checkpoint", "step 080000 of each cell, deterministic action selection"],
        ["episode cap", "T1 600, T2 1,200, T3 800 steps &mdash; equal to the environment time-out"],
        ["success signal", "<code>terminated &amp; termination_manager.get_term('success')</code>, per step"],
        ["evaluation sets", "frozen snapshots, batches 0&ndash;9 per level, never regenerated"],
        ["total", "21,600 scored episodes, plus 21,600 discarded warm-up episodes"],
    ]))
    A('<h3>1.3&nbsp; Training configuration, identical for all 108 cells</h3>')
    cfg = [("policy", "LeRobot 0.4.4 diffusion policy"), ("training steps", "80,000, fixed"),
           ("seed", "0, one per cell"), ("batch size", "64"),
           ("optimiser", "Adam, lr 1e-4, weight decay 1e-6"),
           ("betas / grad clip", "0.95, 0.999 / 10.0"),
           ("scheduler", "diffuser cosine, 500 warm-up steps"),
           ("diffusion", "DDPM, 100 train steps, squaredcos_cap_v2"),
           ("inference steps", "10"), ("vision backbone", "ResNet-18, no pretrained weights"),
           ("RGB encoders", "one per camera"), ("cameras", "table_cam, wrist_cam, 3&times;128&times;128"),
           ("crop / norm", "112&times;112, GroupNorm"), ("observation horizon", "2 steps"),
           ("action horizon / exec", "16 / 8"), ("state / action dim", "9 / 7"),
           ("normalisation", "visual MEAN_STD, state+action MIN_MAX"),
           ("norm statistics", "full 400-episode pool, every budget")]
    A('<div class="kv">' + "".join(
        f'<div><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in cfg)
      + '</div>')
    A('<p class="note">All 108 cells reached step 80,000 in a single allocation with no resume. '
      'The 126 resolved configuration keys are identical across the matrix apart from dataset '
      'path, episode list, job name, output directory and run id.</p>')
    A('</section>')

    # ---------------- 2
    A('<section id="s2"><h2><span class="snum">2</span>Data provenance</h2>')
    rows = []
    for t in TASKS:
        for a in ARMS:
            g = [r for r in d["gen"] if r["task"] == t and r["arm"] == a][0]
            rows.append([f'<span class="tsk">{TASK_LABEL[t]}</span>', f'<b class="arm arm-{a}">{a}</b>',
                         f'<span class="n">{g["legs"]}</span>', f'<span class="n">{g["demos"]:,}</span>',
                         f'<span class="n">{g["failures"]:,}</span>',
                         f'<span class="n">{g["attempts"]:,}</span>',
                         f'<span class="n">{g["gen_sr"]*100:.1f}&thinsp;%</span>',
                         f'<span class="n">{g["mean_ep_len"]:.0f}</span>',
                         f'<span class="n">{DATASET_FRAMES[(t, a)]:,}</span>'])
    A(tbl(["task", "arm", "generation legs", "demonstrations kept", "failed attempts", "attempts",
           "generation SR", "mean demo length (steps)", "dataset frames"], rows, cls="wide",
          note="Every dataset holds exactly 400 episodes at 20&nbsp;fps. Smaller budgets are "
               "contiguous prefixes of the same episode order, which is round-robin over the ten "
               "variants, so every budget of a variant arm is variant-balanced. BC in T1 produced "
               "404 demonstrations; the four surplus episodes fall outside the 400-episode prefix "
               "and are unused."))
    A(fig(10))
    A('</section>')

    # ---------------- 3
    A('<section id="s3"><h2><span class="snum">3</span>Measured surface</h2>')
    A('<p class="lede">All 108 cells, each 200 scored episodes. Intervals are Wilson score '
      'intervals at 95&thinsp;% on the 200 episodes of that cell.</p>')
    A(fig(1))
    A(fig(2))
    A('<h3>3.1&nbsp; Cell table</h3>')
    for t in TASKS:
        rows = []
        for a in ARMS:
            row = [f'<b class="arm arm-{a}">{a}</b>']
            for n in NDEMOS:
                c = [r for r in d["surface"] if r["task"] == t and r["arm"] == a and r["n"] == n][0]
                row.append(f'<span class="n big">{c["sr"]:.3f}</span>'
                           f'<span class="sub">{c["k"]}/200 &nbsp;'
                           f'[{c["wilson_lo"]:.3f}, {c["wilson_hi"]:.3f}]</span>')
            rows.append(row)
        A(f'<h4 class="tasklabel">{TASK_LABEL[t]}</h4>')
        A(tbl(["arm"] + [f"N&thinsp;=&thinsp;{n}" for n in NDEMOS], rows, cls="cells"))
    A('</section>')

    # ---------------- 4
    A('<section id="s4"><h2><span class="snum">4</span>Leave-one-out results</h2>')
    A('<p class="lede">Each row removes one axis from the full set and reports the change in '
      'success rate at a matched demonstration budget. Intervals are Newcombe 95&thinsp;% '
      'intervals for the difference of two independent proportions. Axis C is the one comparison '
      'whose two sides are episode-matched, so the paired McNemar statistic is given for it as '
      'well.</p>')
    A(fig(3))
    A(fig(4))
    A('<h3>4.1&nbsp; Delta table</h3>')
    A('<div class="controls" role="group" aria-label="Filter the delta table">'
      '<label class="chk"><input type="checkbox" id="onlysig"> show only intervals that exclude zero</label>'
      '<span class="filters">'
      + "".join(f'<button class="chip" data-task="{t}" type="button">{TASK_LABEL[t]}</button>'
                for t in TASKS)
      + "".join(f'<button class="chip" data-axis="{ax}" type="button">axis {ax}</button>'
                for ax in REMOVE)
      + '<button class="chip clear" id="clearf" type="button">all</button></span></div>')
    rows = []
    for r in d["loo"]:
        mc = r.get("mcnemar")
        mcs = (f'<span class="mono sm">b={mc["b"]} c={mc["c"]} '
               f'{mc["diff"]:+.3f} [{mc["lo"]:+.3f}, {mc["hi"]:+.3f}] '
               f'&chi;&sup2;={mc["chi2_cc"]:.2f}</span>') if mc else ""
        sig = "sig" if r["significant"] else "ns"
        bar = delta_bar(r["delta"], r["ci_lo"], r["ci_hi"], r["axis"], r["significant"])
        rows.append([
            f'<span class="tsk">{TASK_LABEL[r["task"]]}</span>',
            f'<b class="ax ax-{r["axis"]}">{r["axis"]}</b>',
            f'<b class="arm arm-{r["removed_arm"]}">{r["removed_arm"]}</b>',
            f'<span class="n">{r["n"]}</span>',
            f'<span class="n">{r["sr_full"]:.3f}</span>',
            f'<span class="n">{r["sr_reduced"]:.3f}</span>',
            f'<span class="n {sig}">{r["delta"]:+.3f}</span>',
            f'<span class="n sm">[{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}]</span>',
            bar, mcs])
    body = "\n".join(
        f'<tr data-task="{r["task"]}" data-axis="{r["axis"]}" '
        f'data-sig="{int(r["significant"])}">' + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for r, row in zip(d["loo"], rows))
    head = ["task", "axis", "arm", "N", "SR L3b", "SR reduced", "&Delta;", "Newcombe 95&thinsp;%",
            "", "paired McNemar (axis C)"]
    A('<div class="scroll"><table class="wide" id="lootable"><thead><tr>'
      + "".join(f"<th>{c}</th>" for c in head) + f'</tr></thead><tbody>{body}</tbody></table></div>')
    A('<p class="note">Bold deltas are those whose interval excludes zero. The bar column draws '
      'each delta and its interval on a common scale spanning &minus;0.8 to +0.8.</p>')
    A('</section>')

    # ---------------- 5
    A('<section id="s5"><h2><span class="snum">5</span>Within-cell dispersion</h2>')
    A('<p class="lede">Every cell is ten independent processes of 20 episodes. The spread of those '
      'ten slice rates is the sampling variability behind each pooled cell value.</p>')
    A(fig(5))
    over = [r for r in d["surface"] if r["slice_overdispersed"]]
    rows = [[f'<span class="tsk">{TASK_LABEL[r["task"]]}</span>',
             f'<b class="arm arm-{r["arm"]}">{r["arm"]}</b>', f'<span class="n">{r["n"]}</span>',
             f'<span class="n">{r["sr"]:.3f}</span>', f'<span class="n">{r["slice_min"]:.2f}</span>',
             f'<span class="n">{r["slice_max"]:.2f}</span>',
             f'<span class="n">{r["slice_chi2"]:.1f}</span>',
             '<span class="flag yes">yes</span>' if r["slice_overdispersed"]
             else '<span class="flag no">no</span>']
            for r in sorted(d["surface"], key=lambda r: -r["slice_chi2"])[:14]]
    A(tbl(["task", "arm", "N", "pooled SR", "slice min", "slice max", "&chi;&sup2; (df 9)",
           "rejects a common rate"], rows,
          note=f"{len(over)} of the 108 cells reject a common rate across their ten slices at "
               f"95&thinsp;% (critical value 16.92 on 9 degrees of freedom). The fourteen largest "
               f"statistics are listed."))
    A('</section>')

    # ---------------- 6
    A('<section id="s6"><h2><span class="snum">6</span>T2 drawer_stow milestone instrumentation</h2>')
    A('<p class="lede">T2 is the only task instrumented with intermediate milestones; T1 and T3 '
      'record binary success only. A milestone latches when its condition is met at any point in '
      'the episode: drawer opened past 0.15&nbsp;m, object lifted past 0.05&nbsp;m, object above '
      'the drawer cavity.</p>')
    A('<p class="note">The three milestones are <b>not nested</b> &mdash; an episode can lift the '
      'object without ever opening the drawer past threshold &mdash; so every conditional rate is '
      'computed from the joint count, not from the ratio of two marginal rates.</p>')
    A(fig(6))
    A(fig(7))
    A(fig(9))
    rows = [[f'<b class="arm arm-{r["arm"]}">{r["arm"]}</b>', f'<span class="n">{r["n"]}</span>',
             f'<span class="n">{r["p_opened"]:.3f}</span>', f'<span class="n">{r["p_lifted"]:.3f}</span>',
             f'<span class="n">{r["p_over"]:.3f}</span>',
             f'<span class="n big">{r["p_success"]:.3f}</span>',
             num(r["lift_given_open"]), num(r["over_given_lift"]), num(r["success_given_over"]),
             f'<span class="n">{r["max_open_median"]:.4f}</span>',
             f'<span class="n">{r["max_lift_median"]:.4f}</span>'] for r in d["stages"]]
    A(tbl(["arm", "N", "drawer opened", "object lifted", "object over drawer", "success",
           "lift | opened", "over | lifted", "success | over", "median max opening (m)",
           "median max lift (m)"], rows, cls="wide"))
    A('</section>')

    # ---------------- 7
    A('<section id="s7"><h2><span class="snum">7</span>Time to success</h2>')
    A('<p class="lede">Step index at which a successful episode first satisfied the success '
      'condition, over successful episodes only, all budgets pooled.</p>')
    A(fig(8))
    rows = [[f'<span class="tsk">{TASK_LABEL[r["task"]]}</span>',
             f'<b class="arm arm-{r["arm"]}">{r["arm"]}</b>',
             f'<span class="n">{r["n_success"]:,}</span>', f'<span class="n">{r["min"]:.0f}</span>',
             f'<span class="n">{r["p25"]:.0f}</span>', f'<span class="n big">{r["median"]:.0f}</span>',
             f'<span class="n">{r["p75"]:.0f}</span>', f'<span class="n">{r["max"]:.0f}</span>',
             f'<span class="n dim">{CAPS[r["task"]]}</span>'] for r in d["timing"]]
    A(tbl(["task", "arm", "successful episodes", "min", "p25", "median", "p75", "max",
           "episode cap"], rows))
    A('</section>')

    # ---------------- 8
    A('<section id="s8"><h2><span class="snum">8</span>Implementation flaws</h2>')
    A('<p class="lede">Each entry is a property of how this study was implemented or measured, '
      'with the quantity that establishes it. They are listed, not weighted.</p>')
    A('<ol class="flaws">')
    for fid, title, body in FLAWS:
        A(f'<li><div class="fid">{fid}</div><div class="fbody"><h4>{esc(title)}</h4>'
          f'<p>{esc(body)}</p></div></li>')
    A('</ol></section>')

    # ---------------- 9
    A('<section id="s9"><h2><span class="snum">9</span>Open items</h2><ul class="todos">')
    for tid, text, status in TODOS:
        done = status.startswith("done")
        A(f'<li class="{"done" if done else "open"}"><span class="tid">{tid}</span>'
          f'<span class="ttext">{esc(text)}</span>'
          f'<span class="status">{esc(status)}</span></li>')
    A('</ul></section>')

    # ---------------- appendix A
    A('<section id="sa"><h2><span class="snum">A</span>Budget-pooled deltas</h2>')
    A('<p class="lede">The same deltas with all six budgets summed into one 1,200-episode '
      'proportion per side. Reported for completeness only: the six budgets are six different '
      'policies, and the homogeneity statistic beside each row tests whether they share one '
      'success rate (critical value 11.07 on 5 degrees of freedom). Where that test rejects, the '
      'pooled interval describes a mixture whose value depends on which budgets were included, '
      'not a single parameter.</p>')
    rows = [[f'<span class="tsk">{TASK_LABEL[r["task"]]}</span>',
             f'<b class="ax ax-{r["axis"]}">{r["axis"]}</b>',
             f'<b class="arm arm-{r["removed_arm"]}">{r["removed_arm"]}</b>',
             f'<span class="n">{r["k_full"]:,}</span>', f'<span class="n">{r["k_reduced"]:,}</span>',
             f'<span class="n">{r["episodes"]:,}</span>',
             f'<span class="n big">{r["delta"]:+.3f}</span>',
             f'<span class="n sm">[{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}]</span>',
             f'<span class="n">{r["chi2_full"]:.1f}</span>',
             f'<span class="n">{r["chi2_reduced"]:.1f}</span>',
             '<span class="flag yes">yes</span>' if r["overdispersed"]
             else '<span class="flag no">no</span>'] for r in d["pooled"]]
    A(tbl(["task", "axis", "arm", "successes L3b", "successes reduced", "episodes / side",
           "pooled &Delta;", "Newcombe 95&thinsp;%", "&chi;&sup2; L3b (df 5)",
           "&chi;&sup2; reduced (df 5)", "homogeneity rejected"], rows, cls="wide"))
    A('</section>')

    # ---------------- appendix B
    A('<section id="sb"><h2><span class="snum">B</span>Files behind this report</h2>')
    files = [
        ("results/eval_&lt;TASK&gt;_&lt;ARM&gt;_n&lt;N&gt;_080000_u200d32.json",
         "the 108 pooled cells, each carrying its 200 per-episode outcome records"),
        ("paper/loo_report/tables/surface.csv",
         "108 cells: successes, Wilson interval, slice spread, homogeneity"),
        ("paper/loo_report/tables/loo.csv", "54 leave-one-out rows, Newcombe and McNemar statistics"),
        ("paper/loo_report/tables/pooled.csv", "9 budget-pooled rows with homogeneity statistics"),
        ("paper/loo_report/tables/stages.csv", "36 T2 milestone rows, marginal and joint counts"),
        ("paper/loo_report/tables/timing.csv", "18 first-success timing summaries"),
        ("paper/loo_report/tables/gen.csv", "18 generation-yield rows"),
        ("src/cog/analysis/loo.py", "every statistic in this report"),
        ("src/cog/analysis/loo_figures.py", "every figure in this report"),
        ("scripts/dev/build_loo_report.py", "the PDF edition"),
        ("scripts/dev/build_loo_artifact.py", "this page"),
    ]
    A(tbl(["path", "contents"],
          [[f'<code>{p}</code>', c] for p, c in files]))
    A('</section>')
    A('</main>')
    A(f'<footer><p>Leave-one-out disturbance ablation &middot; repository '
      f'<code>cost_of_generality</code> @ <code>{rev()}</code> &middot; generated {today}</p></footer>')
    A(JS)
    return "\n".join(P)


def delta_bar(delta, lo, hi, axis, sig):
    """A small inline scale drawing: interval as a rule, point as a mark, zero as a tick."""
    def pct(v):
        return max(0.0, min(100.0, (v + 0.8) / 1.6 * 100.0))
    l, h, p = pct(lo), pct(hi), pct(delta)
    cls = "on" if sig else "off"
    return (f'<span class="bar {cls} ax-{axis}" role="img" '
            f'aria-label="delta {delta:+.3f}, interval {lo:+.3f} to {hi:+.3f}">'
            f'<span class="zero"></span>'
            f'<span class="rule" style="left:{l:.2f}%;width:{max(h-l,0.6):.2f}%"></span>'
            f'<span class="dot" style="left:{p:.2f}%"></span></span>')


CSS = """<style>
:root{
  --paper:#f6f7f9; --card:#ffffff; --ink:#15181d; --ink-2:#3d4650; --muted:#6b7580;
  --line:#dde2e8; --line-2:#eceff3; --accent:#1f4f7a; --accent-soft:#e8eff6;
  --ax-a:#c2453f; --ax-b:#c96a11; --ax-c:#3f7f39;
  --good:#3f7f39; --flag:#a8560d;
  --on-accent:#ffffff;
  --shadow:0 1px 2px rgba(20,28,38,.06), 0 8px 24px -16px rgba(20,28,38,.28);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#101317; --card:#171b21; --ink:#e7ebf0; --ink-2:#b9c2cc; --muted:#8a95a1;
    --line:#2a313a; --line-2:#212730; --accent:#7fb3de; --accent-soft:#1b2833;
    --ax-a:#e8706a; --ax-b:#e59a4a; --ax-c:#77bd6f; --good:#77bd6f; --flag:#e0a163;
    --on-accent:#0d1116;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --paper:#101317; --card:#171b21; --ink:#e7ebf0; --ink-2:#b9c2cc; --muted:#8a95a1;
  --line:#2a313a; --line-2:#212730; --accent:#7fb3de; --accent-soft:#1b2833;
  --ax-a:#e8706a; --ax-b:#e59a4a; --ax-c:#77bd6f; --good:#77bd6f; --flag:#e0a163;
  --on-accent:#0d1116;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  background:var(--paper); color:var(--ink); margin:0;
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.masthead,nav.toc,main,footer{max-width:1180px; margin-inline:auto; padding-inline:16px}
.masthead{padding-block:52px 22px}
.eyebrow{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin:0 0 14px;
}
h1{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:clamp(30px,5.4vw,50px);
  line-height:1.06; letter-spacing:-.015em; margin:0 0 14px; text-wrap:balance;
}
.standfirst{font-size:clamp(16px,2.2vw,19px); line-height:1.5; color:var(--ink-2); max-width:62ch; margin:0 0 26px}
.facts{display:flex; flex-wrap:wrap; gap:0; margin:0 0 22px; border-top:1px solid var(--line); border-bottom:1px solid var(--line)}
.facts>div{padding:11px 22px 11px 0; margin-right:22px; border-right:1px solid var(--line-2); flex:0 0 auto}
.facts>div:last-child{border-right:0}
.facts dt{font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin:0 0 3px}
.facts dd{margin:0; font-weight:500; font-variant-numeric:tabular-nums; font-size:15px}
.scopenote{
  background:var(--accent-soft); border-left:3px solid var(--accent); padding:12px 16px;
  margin:0; font-size:14px; color:var(--ink-2); max-width:78ch;
}
nav.toc{padding-block:10px 30px}
nav.toc ol{list-style:none; display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:2px 20px; margin:0; padding:0}
nav.toc a{display:flex; gap:10px; align-items:baseline; text-decoration:none; color:var(--ink-2); padding:5px 0; border-bottom:1px solid transparent; font-size:14px}
nav.toc a:hover{color:var(--accent); border-bottom-color:var(--line)}
.tnum{font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--muted); min-width:14px}
main{padding-block:0 40px}
section{padding-block:34px; border-top:1px solid var(--line)}
h2{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:clamp(21px,3vw,28px);
  letter-spacing:-.01em; margin:0 0 16px; display:flex; gap:14px; align-items:baseline; scroll-margin-top:20px;
}
.snum{font-family:"IBM Plex Mono",monospace; font-size:13px; color:var(--accent); font-weight:500; letter-spacing:.04em}
h3{font-size:16px; font-weight:600; margin:30px 0 10px; letter-spacing:-.005em}
h4.tasklabel{font-family:"IBM Plex Mono",monospace; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin:22px 0 7px; font-weight:500}
p{margin:0 0 12px}
.lede{max-width:78ch; color:var(--ink-2)}
.note{font-size:13px; color:var(--muted); max-width:86ch; margin-top:9px}
.mono,code{font-family:"IBM Plex Mono",ui-monospace,monospace}
code{font-size:.9em; background:var(--line-2); padding:1px 5px; border-radius:3px}
.sm{font-size:12px}
/* tables */
.scroll{overflow-x:auto; margin:14px 0 4px; border:1px solid var(--line); border-radius:6px; background:var(--card)}
table{border-collapse:collapse; width:100%; font-size:13.5px}
thead th{
  text-align:left; font-weight:600; font-size:11.5px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--muted); padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap;
  background:var(--card); position:sticky; top:0;
}
td{padding:8px 12px; border-bottom:1px solid var(--line-2); vertical-align:middle}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--line-2)}
table.wide{font-size:12.5px}
table.wide td,table.wide th{padding:6px 10px; white-space:nowrap}
.n{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums}
.n.big{font-weight:500}
.n.dim{color:var(--muted)}
.n.sig{font-weight:600}
.n.ns{color:var(--muted)}
.na{color:var(--muted)}
table.cells td{vertical-align:top}
table.cells .sub{display:block; font-family:"IBM Plex Mono",monospace; font-size:10.5px; color:var(--muted); margin-top:1px}
.tsk{color:var(--ink-2)}
.arm{font-family:"IBM Plex Mono",monospace; font-weight:500}
.arm-L3b{color:var(--ink)}
.arm-AC{color:var(--ax-a)} .arm-BC{color:var(--ax-b)} .arm-L2{color:var(--ax-c)}
.arm-L0,.arm-L1{color:var(--muted)}
.ax{font-family:"IBM Plex Mono",monospace}
.ax-A{color:var(--ax-a)} .ax-B{color:var(--ax-b)} .ax-C{color:var(--ax-c)}
.flag{font-size:11.5px; font-family:"IBM Plex Mono",monospace}
.flag.yes{color:var(--flag)} .flag.no{color:var(--muted)}
/* delta bar */
.bar{position:relative; display:block; width:150px; height:15px}
.bar .zero{position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--line); }
.bar .rule{position:absolute; top:7px; height:2px; border-radius:1px; background:currentColor; opacity:.45}
.bar .dot{position:absolute; top:4px; width:8px; height:8px; margin-left:-4px; border-radius:50%; background:currentColor}
.bar.ax-A{color:var(--ax-a)} .bar.ax-B{color:var(--ax-b)} .bar.ax-C{color:var(--ax-c)}
.bar.off{opacity:.42}
/* kv grid */
.kv{display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); border-radius:6px; overflow:hidden; margin:14px 0}
.kv>div{background:var(--card); padding:9px 13px; display:flex; flex-direction:column; gap:1px}
.kv .k{font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
.kv .v{font-size:13.5px}
/* figures */
figure.plate{margin:20px 0 6px; background:#ffffff; border:1px solid var(--line); border-radius:6px; padding:14px; box-shadow:var(--shadow)}
figure.plate img{display:block; width:100%; height:auto}
figcaption{font-size:12.5px; color:#5c6570; margin-top:10px; line-height:1.45; max-width:92ch}
.fignum{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#1f4f7a; margin-right:7px}
/* controls */
.controls{display:flex; flex-wrap:wrap; gap:14px 18px; align-items:center; margin:16px 0 4px; font-size:13px}
.chk{display:flex; gap:7px; align-items:center; color:var(--ink-2); cursor:pointer}
.filters{display:flex; flex-wrap:wrap; gap:6px}
.chip{
  font:inherit; font-size:12px; padding:3px 11px; border-radius:99px; cursor:pointer;
  border:1px solid var(--line); background:var(--card); color:var(--ink-2);
}
.chip:hover{border-color:var(--accent); color:var(--accent)}
.chip[aria-pressed="true"]{background:var(--accent); border-color:var(--accent); color:var(--on-accent)}
.chip.clear{border-style:dashed}
/* flaws + todos */
ol.flaws{list-style:none; margin:16px 0 0; padding:0; display:flex; flex-direction:column; gap:0}
ol.flaws li{display:grid; grid-template-columns:52px 1fr; gap:14px; padding:15px 0; border-top:1px solid var(--line-2)}
ol.flaws li:first-child{border-top:0}
.fid{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--flag); font-weight:500; padding-top:2px}
.fbody h4{margin:0 0 5px; font-size:15px; font-weight:600}
.fbody p{margin:0; color:var(--ink-2); font-size:14px; max-width:92ch}
ul.todos{list-style:none; margin:14px 0 0; padding:0}
ul.todos li{display:grid; grid-template-columns:46px 1fr auto; gap:12px; align-items:baseline; padding:9px 0; border-top:1px solid var(--line-2)}
ul.todos li:first-child{border-top:0}
.tid{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--muted)}
.ttext{font-size:14px}
ul.todos .status{font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--flag); white-space:nowrap}
ul.todos li.done .status{color:var(--good)}
ul.todos li.done .ttext{color:var(--muted)}
footer{border-top:1px solid var(--line); padding-block:22px 40px; color:var(--muted); font-size:12.5px}
footer code{background:none; padding:0}
@media (max-width:640px){
  .masthead{padding-block:34px 16px}
  .facts>div{padding-right:14px; margin-right:14px}
  .bar{width:100px}
  ol.flaws li{grid-template-columns:1fr; gap:4px}
  ul.todos li{grid-template-columns:1fr; gap:2px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important}}
</style>"""

JS = """<script>
(function(){
  var table = document.getElementById('lootable');
  if(!table) return;
  var rows = Array.prototype.slice.call(table.querySelectorAll('tbody tr'));
  var onlysig = document.getElementById('onlysig');
  var chips = Array.prototype.slice.call(document.querySelectorAll('.chip[data-task],.chip[data-axis]'));
  var clear = document.getElementById('clearf');
  var state = {task:null, axis:null};
  function apply(){
    rows.forEach(function(tr){
      var ok = true;
      if(state.task && tr.dataset.task !== state.task) ok = false;
      if(state.axis && tr.dataset.axis !== state.axis) ok = false;
      if(onlysig.checked && tr.dataset.sig !== '1') ok = false;
      tr.hidden = !ok;
    });
    chips.forEach(function(c){
      var on = (c.dataset.task && c.dataset.task === state.task) ||
               (c.dataset.axis && c.dataset.axis === state.axis);
      c.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }
  chips.forEach(function(c){
    c.setAttribute('aria-pressed','false');
    c.addEventListener('click', function(){
      if(c.dataset.task) state.task = (state.task === c.dataset.task) ? null : c.dataset.task;
      if(c.dataset.axis) state.axis = (state.axis === c.dataset.axis) ? null : c.dataset.axis;
      apply();
    });
  });
  clear.addEventListener('click', function(){ state.task = null; state.axis = null;
    onlysig.checked = false; apply(); });
  onlysig.addEventListener('change', apply);
  apply();
})();
</script>"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
