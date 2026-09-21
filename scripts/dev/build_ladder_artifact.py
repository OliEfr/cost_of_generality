#!/usr/bin/env python3
"""Emit paper/ladder_report/artifact/index.html -- the additive-ladder study, web edition."""
from __future__ import annotations

import datetime as dt
import html
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from cog.analysis.loo import NDEMOS, TASKS, TASK_LABEL  # noqa: E402
from cog.analysis import ladder_figures  # noqa: E402
from cog.analysis.ladder import AXIS_PLAIN, RUNGS, SETUP_NAME, SETUPS  # noqa: E402
from build_loo_artifact import CSS  # noqa: E402
from build_ladder_report import CAVEATS, TODO, WHAT  # noqa: E402

OUT = REPO / "paper" / "ladder_report" / "artifact" / "index.html"

RUNG_CSS = """<style>
.r-L0{color:#7b93a9} .r-L1{color:#6d9cc4} .r-L2{color:#3f6f9e} .r-L3b{color:#1d3f5e}
.a-A{color:var(--ax-a)} .a-B{color:var(--ax-b)} .a-C{color:var(--ax-c)}
.bar.a-A{color:var(--ax-a)} .bar.a-B{color:var(--ax-b)} .bar.a-C{color:var(--ax-c)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .r-L0{color:#a9c0d6}
  :root:not([data-theme="light"]) .r-L1{color:#8fb6d6}
  :root:not([data-theme="light"]) .r-L2{color:#7fa8cc}
  :root:not([data-theme="light"]) .r-L3b{color:#cdd9e4}}
:root[data-theme="dark"] .r-L0{color:#a9c0d6} :root[data-theme="dark"] .r-L1{color:#8fb6d6}
:root[data-theme="dark"] .r-L2{color:#7fa8cc} :root[data-theme="dark"] .r-L3b{color:#cdd9e4}
.rung{font-weight:600}
</style>"""

FIGS = {
    1: ("ladder02_delta.png",
        "Change in success rate when a disturbance is switched on, given the rungs below it. "
        "Bands are 95&thinsp;% intervals; below the dashed line it made things worse."),
    2: ("ladder01_success.png",
        "Success rate at each rung. Error bars are 95&thinsp;% intervals on 200 episodes."),
    3: ("ladder03_funnel.png",
        "How far each episode got before it stopped. Numbers are percentages of 200 episodes."),
}


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
    return (f'<figure class="plate">\n  <img src="figures/{name}" alt="{esc(cap)}" '
            f'loading="lazy">\n  <figcaption>{cap}</figcaption>\n</figure>')


def tbl(head, rows, cls=""):
    th = "".join(f"<th>{c}</th>" for c in head)
    body = "\n".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<div class="scroll"><table class="{cls}">\n<thead><tr>{th}</tr></thead>\n'
            f'<tbody>\n{body}\n</tbody></table></div>')


def bar(delta, lo, hi, axis, sig):
    def pct(v):
        return max(0.0, min(100.0, (v + 0.8) / 1.6 * 100.0))
    l, h, p = pct(lo), pct(hi), pct(delta)
    return (f'<span class="bar {"on" if sig else "off"} a-{axis}" role="img" '
            f'aria-label="{delta:+.3f}, interval {lo:+.3f} to {hi:+.3f}">'
            f'<span class="zero"></span>'
            f'<span class="rule" style="left:{l:.2f}%;width:{max(h-l,0.6):.2f}%"></span>'
            f'<span class="dot" style="left:{p:.2f}%"></span></span>')


def build_html():
    d, _ = ladder_figures.build_all()
    today = dt.date.today().isoformat()
    P = []
    A = P.append

    A('<title>Additive disturbance ladder</title>')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&'
      'family=IBM+Plex+Serif:wght@500;600&display=swap">')
    A(CSS)
    A(RUNG_CSS)

    A('<header class="masthead">')
    A('  <h1>Additive disturbance ladder</h1>')
    A('  <p class="standfirst">Start with nothing varying, switch the disturbances on one at a '
      'time, and see what each one costs.</p>')
    A('  <dl class="facts">')
    for k, v in (("rungs per task", "4"), ("policies", "72"), ("episodes scored", "14,400"),
                 ("budgets", "10 &rarr; 400 demos"), ("generated", today)):
        A(f'    <div><dt>{k}</dt><dd>{v}</dd></div>')
    A('  </dl></header>')

    A('<main>')

    A('<section><h2>What was run</h2>')
    A('<p class="lede">Four rungs per task, each trained at six demonstration budgets and scored '
      'on 200 episodes. Same training recipe everywhere, one seed, 80,000 steps.</p>')
    rows = []
    for r in WHAT[1:]:
        arm = r[4]
        rows.append([f'<b class="rung r-{arm}">{r[0]}</b>'] +
                    [f'<span class="fixed">{c}</span>' if c == "FIXED" else c for c in r[1:4]] +
                    [f'<code>{arm}</code>'])
    A(tbl(["rung", "object start position", "goal position", "object type", "name on disk"], rows))
    A('<p class="note">Tasks: <b>T1 cup_place</b> put a cup on a goal disk &middot; '
      '<b>T2 drawer_stow</b> open a drawer and stow an object in it &middot; '
      '<b>T3 push_target</b> push a puck to a target. The cost reported for a disturbance is '
      'always <b>success rate with it on, minus success rate one rung below, at the same '
      'demonstration budget</b>. Negative means switching it on made the policy worse. All four '
      'rungs&rsquo; demonstrations were generated on the same GPU, so no comparison here crosses '
      'a hardware boundary.</p>')
    A('</section>')

    A('<section><h2>What each disturbance costs</h2>')
    A(fig(1))
    for t in TASKS:
        rows = []
        for n in NDEMOS:
            row = [f'<span class="n">{n}</span>']
            for axis in RUNGS:
                r = [x for x in d["ladder"] if x["task"] == t and x["axis"] == axis
                     and x["n"] == n][0]
                sig = "sig" if r["significant"] else "ns"
                tag = "" if r["significant"] else '<span class="ns sm"> not clear of zero</span>'
                row.append(f'<span class="n {sig}">{r["delta"]:+.3f}</span>'
                           f'<span class="sub">[{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}]{tag}</span>')
                row.append(bar(r["delta"], r["ci_lo"], r["ci_hi"], axis, r["significant"]))
            rows.append(row)
        A(f'<h3>{TASK_LABEL[t]}</h3>')
        A(tbl(["demos"] + [c for a in RUNGS for c in (f'switch on {AXIS_PLAIN[a]}', "")],
              rows, cls="cells"))
    A('<p class="note">Read each budget on its own line. The same disturbance can cost a lot at '
      '25 demonstrations and nothing at 400.</p>')
    A('</section>')

    A('<section><h2>The success rates behind those costs</h2>')
    A(fig(2))
    for t in TASKS:
        rows = []
        for n in NDEMOS:
            row = [f'<span class="n">{n}</span>']
            for a in SETUPS:
                c = [x for x in d["surface"] if x["task"] == t and x["arm"] == a
                     and x["n"] == n][0]
                row.append(f'<span class="n big">{c["sr"]:.3f}</span>'
                           f'<span class="sub">{c["k"]}/200</span>')
            rows.append(row)
        A(f'<h3>{TASK_LABEL[t]}</h3>')
        A(tbl(["demos"] + [f'<span class="r-{a}">{SETUP_NAME[a]}</span>' for a in SETUPS],
              rows, cls="cells"))
    A('</section>')

    A('<section><h2>Where T2 policies got stuck</h2>')
    A('<p class="lede">T2 is a four-part task: open the drawer, pick up the object, carry it over '
      'the open drawer, drop it in. Every episode records how far it got, so a failure can be '
      'placed instead of just counted. Each bar is 200 episodes split by the furthest point '
      'reached, so the five shares add to 100.</p>')
    A(fig(3))
    rows = []
    for a in SETUPS:
        for n in NDEMOS:
            r = [x for x in d["furthest"] if x["arm"] == a and x["n"] == n][0]
            rows.append([f'<b class="rung r-{a}">{SETUP_NAME[a]}</b>',
                         f'<span class="n">{n}</span>',
                         f'<span class="n">{r["got nowhere"]}</span>',
                         f'<span class="n">{r["opened drawer"]}</span>',
                         f'<span class="n">{r["lifted object"]}</span>',
                         f'<span class="n">{r["held over drawer"]}</span>',
                         f'<span class="n big">{r["succeeded"]}</span>'])
    A(tbl(["rung", "demos", "never opened", "opened, stopped", "lifted, stopped",
           "held over, stopped", "stowed it"], rows))
    A('<p class="note">Counts out of 200 episodes.</p>')
    A('</section>')

    A('<section><h2>What could be wrong with these numbers</h2><ol class="flaws">')
    for i, (title, body) in enumerate(CAVEATS):
        A(f'<li><div class="fid">{i + 1}</div><div class="fbody"><h4>{esc(title)}</h4>'
          f'<p>{esc(body)}</p></div></li>')
    A('</ol></section>')

    A('<section><h2>What would fix them</h2><ul class="todos">')
    for i, t in enumerate(TODO):
        A(f'<li class="open"><span class="tid">{i + 1}</span>'
          f'<span class="ttext">{esc(t)}</span></li>')
    A('</ul></section>')

    A('</main>')
    A(f'<footer><p>Every number from <code>src/cog/analysis/ladder.py</code> over the pooled '
      f'result files. Tables as CSV in <code>paper/ladder_report/tables/</code>. Repository '
      f'<code>cost_of_generality</code> @ <code>{rev()}</code> &middot; {today}</p></footer>')
    return "\n".join(P)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
