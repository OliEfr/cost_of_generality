#!/usr/bin/env python3
"""Emit paper/loo_report/artifact/index.html -- the short web edition.

Four setups, three numbers per task per budget, plain words. Every number comes from
cog.analysis.loo, the same module the PDF uses.
"""
from __future__ import annotations

import datetime as dt
import html
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from cog.analysis.loo import NDEMOS, TASKS, TASK_LABEL, build  # noqa: E402
from cog.analysis import loo_figures  # noqa: E402
from build_loo_simple import CAVEATS, TODO, WHAT  # noqa: E402

OUT = REPO / "paper" / "loo_report" / "artifact" / "index.html"
SETUPS = loo_figures.SETUPS
NAME = loo_figures.SETUP_NAME
PLAIN = loo_figures.AXIS_PLAIN
CLS = {"L3b": "L3b", "BC": "AC", "AC": "BC", "L2": "L2"}  # colour classes, see CSS

FIGS = {
    1: ("simple02_delta.png",
        "Change in success rate against keeping all three disturbances. Bands are 95&thinsp;% "
        "intervals; above the dashed line the simpler setup did better."),
    2: ("simple01_success.png",
        "Success rate of each setup. Error bars are 95&thinsp;% intervals on 200 episodes."),
    3: ("simple03_funnel.png",
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
    return (f'<figure class="plate">\n'
            f'  <img src="figures/{name}" alt="{esc(cap)}" loading="lazy">\n'
            f'  <figcaption>{cap}</figcaption>\n</figure>')


def tbl(head, rows, cls=""):
    th = "".join(f"<th>{c}</th>" for c in head)
    body = "\n".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<div class="scroll"><table class="{cls}">\n<thead><tr>{th}</tr></thead>\n'
            f'<tbody>\n{body}\n</tbody></table></div>')


def bar(delta, lo, hi, arm, sig):
    def pct(v):
        return max(0.0, min(100.0, (v + 0.8) / 1.6 * 100.0))
    l, h, p = pct(lo), pct(hi), pct(delta)
    return (f'<span class="bar {"on" if sig else "off"} s-{CLS[arm]}" role="img" '
            f'aria-label="{delta:+.3f}, interval {lo:+.3f} to {hi:+.3f}">'
            f'<span class="zero"></span>'
            f'<span class="rule" style="left:{l:.2f}%;width:{max(h-l,0.6):.2f}%"></span>'
            f'<span class="dot" style="left:{p:.2f}%"></span></span>')


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

    A('<header class="masthead">')
    A('  <h1>Leave-one-out disturbance ablation</h1>')
    A('  <p class="standfirst">Train on everything, then drop one disturbance at a time and see '
      'what the success rate does.</p>')
    A('  <dl class="facts">')
    for k, v in (("setups per task", "4"), ("policies", "72"), ("episodes scored", "14,400"),
                 ("budgets", "10 &rarr; 400 demos"), ("generated", today)):
        A(f'    <div><dt>{k}</dt><dd>{v}</dd></div>')
    A('  </dl></header>')

    A('<main>')

    # ---- what was run
    A('<section><h2>What was run</h2>')
    A('<p class="lede">Four setups per task, each trained at six demonstration budgets and scored '
      'on 200 episodes. Same training recipe everywhere, one seed, 80,000 steps.</p>')
    rows = []
    for r in WHAT[1:]:
        arm = r[4]
        rows.append([f'<b class="setup s-{CLS[arm]}">{r[0]}</b>'] +
                    [f'<span class="fixed">{c}</span>' if c == "FIXED" else c for c in r[1:4]] +
                    [f'<code>{r[4]}</code>'])
    A(tbl(["setup", "object start position", "goal position", "object type", "name on disk"], rows))
    A('<p class="note">Tasks: <b>T1 cup_place</b> put a cup on a goal disk &middot; '
      '<b>T2 drawer_stow</b> open a drawer and stow an object in it &middot; '
      '<b>T3 push_target</b> push a puck to a target. The change reported for a disturbance is '
      'always <b>success rate without it minus success rate with all three on, at the same '
      'demonstration budget</b>. Positive means the simpler setup did better.</p>')
    A('</section>')

    # ---- the answer
    A('<section><h2>What dropping each disturbance does</h2>')
    A(fig(1))
    for t in TASKS:
        rows = []
        for n in NDEMOS:
            row = [f'<span class="n">{n}</span>']
            for axis in ("A", "B", "C"):
                r = [x for x in d["loo"] if x["task"] == t and x["axis"] == axis
                     and x["n"] == n][0]
                sig = "sig" if r["significant"] else "ns"
                tag = "" if r["significant"] else '<span class="ns sm"> not clear of zero</span>'
                row.append(f'<span class="n {sig}">{r["delta"]:+.3f}</span>'
                           f'<span class="sub">[{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}]{tag}</span>')
                row.append(bar(r["delta"], r["ci_lo"], r["ci_hi"], r["removed_arm"],
                               r["significant"]))
            rows.append(row)
        A(f'<h3>{TASK_LABEL[t]}</h3>')
        A(tbl(["demos", "minus object start", "", "minus goal", "", "minus object type", ""],
              rows, cls="cells"))
    A('<p class="note">Read each budget on its own line. The same disturbance can matter at 25 '
      'demonstrations and not at 400.</p>')
    A('</section>')

    # ---- raw rates
    A('<section><h2>The success rates behind those changes</h2>')
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
        A(tbl(["demos"] + [f'<span class="s-{CLS[a]}">{NAME[a]}</span>' for a in SETUPS],
              rows, cls="cells"))
    A('</section>')

    # ---- T2 funnel
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
            rows.append([f'<b class="setup s-{CLS[a]}">{NAME[a]}</b>',
                         f'<span class="n">{n}</span>',
                         f'<span class="n">{r["got nowhere"]}</span>',
                         f'<span class="n">{r["opened drawer"]}</span>',
                         f'<span class="n">{r["lifted object"]}</span>',
                         f'<span class="n">{r["held over drawer"]}</span>',
                         f'<span class="n big">{r["succeeded"]}</span>'])
    A(tbl(["setup", "demos", "never opened", "opened, stopped", "lifted, stopped",
           "held over, stopped", "stowed it"], rows))
    A('<p class="note">Counts out of 200 episodes.</p>')
    A('</section>')

    # ---- caveats
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
    A(f'<footer><p>Every number from <code>src/cog/analysis/loo.py</code> over the 108 pooled '
      f'result files. The long edition &mdash; all six arms of the sweep, per-slice dispersion, '
      f'timing distributions, the full flaw list &mdash; is '
      f'<code>paper/loo_report/loo_report_full.pdf</code>; the tables are CSV in '
      f'<code>paper/loo_report/tables/</code>. Repository <code>cost_of_generality</code> @ '
      f'<code>{rev()}</code> &middot; {today}</p></footer>')
    return "\n".join(P)


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
.setup{font-weight:600}
.s-L3b{color:var(--ink)} .s-AC{color:var(--ax-a)} .s-BC{color:var(--ax-b)} .s-L2{color:var(--ax-c)}
.bar.s-AC{color:var(--ax-a)} .bar.s-BC{color:var(--ax-b)} .bar.s-L2{color:var(--ax-c)}
.fixed{font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.06em; color:var(--flag)}
table.cells .sub .ns{color:var(--muted)}
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


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
