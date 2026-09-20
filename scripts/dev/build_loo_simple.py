#!/usr/bin/env python3
"""Build paper/loo_report/loo_report.pdf -- the short edition.

Four setups, three numbers per task per budget, plain words. The long edition with every
diagnostic is build_loo_report.py -> loo_report_full.pdf.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph,
                                Spacer)

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from cog.analysis.loo import NDEMOS, REMOVE, TASKS, TASK_LABEL  # noqa: E402
from cog.analysis import loo_figures  # noqa: E402
from build_loo_report import (CONTENT_W, MARGIN, MUTED, PAGE, RULE, S, figure, git_rev,  # noqa: E402
                              hdr, para, rule, table)

PDF = REPO / "paper" / "loo_report" / "loo_report.pdf"

SETUPS = loo_figures.SETUPS
NAME = loo_figures.SETUP_NAME
PLAIN = loo_figures.AXIS_PLAIN

WHAT = [
    ["setup", "object start position", "goal position", "object type", "name on disk"],
    ["all on", "varies", "varies", "10 types", "L3b"],
    ["minus object start", "FIXED", "varies", "10 types", "BC"],
    ["minus goal", "varies", "FIXED", "10 types", "AC"],
    ["minus object type", "varies", "varies", "1 type", "L2"],
]

CAVEATS = [
    ("The two \"minus\" setups that need new demonstrations were generated on different GPUs.",
     "Demonstrations for minus-object-start and minus-goal came off A100s; the other two setups' "
     "demonstrations came off an RTX 4090. A control policy trained on A100-generated data scored "
     "0.750 where its 4090-generated twin scored 0.860 (difference -0.110, 95 % CI -0.208 to "
     "-0.012). That control changed the GPU and the random seed together, so it bounds the effect "
     "rather than pinning it down. Any change in this report smaller than about 0.11 sits inside "
     "that boundary."),
    ("One training seed.",
     "Every policy is seed 0. The intervals here cover episode sampling only; they contain no "
     "training-run variance."),
    ("Repeating an evaluation does not give the identical episodes back.",
     "Re-running one 20-episode slice with the same seed and checkpoint reproduced the slice total "
     "but flipped two individual episodes. Expect about +/-1 success per 20 episodes of jitter."),
    ("\"Minus object type\" is not a newly generated setup.",
     "Removing the object axis from the full set gives exactly the existing single-object setup, "
     "verified identical over 300 matched resets. It is re-used rather than regenerated, which is "
     "also why it is the only comparison whose two sides see the same starting positions."),
    ("Do not average a change across demonstration budgets.",
     "The six budgets are six different policies. A test for a shared success rate is rejected in "
     "all 18 setup-task pairs, so an average over budgets describes the mix of budgets chosen, not "
     "the setup."),
]

TODO = [
    "Re-run the GPU control with the demonstration seed held fixed, so the GPU is the only thing "
    "that changes.",
    "Train more than one seed on whichever comparisons matter.",
    "Record the evaluator version in the results ledger, and which git revision each run used.",
    "Measure whether a second warm-up batch moves the scored success rate.",
    "Guard the pooling script against mixing evaluator versions, and assert the termination-term "
    "ordering the success signal depends on.",
]


def build():
    d, _ = loo_figures.build_simple()
    doc = BaseDocTemplate(str(PDF), pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=16 * mm,
                          title="Leave-one-out disturbance ablation", author="cost_of_generality")
    frame = Frame(MARGIN, 16 * mm, CONTENT_W, PAGE[1] - MARGIN - 16 * mm, id="f")

    def footer(canvas, docu):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 9 * mm,
                          f"Leave-one-out disturbance ablation   |   cost_of_generality @ "
                          f"{git_rev()}   |   {dt.date.today().isoformat()}")
        canvas.drawRightString(PAGE[0] - MARGIN, 9 * mm, f"page {docu.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN, 12 * mm, PAGE[0] - MARGIN, 12 * mm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
    st = []

    # -------- 1 what was done
    st += [Spacer(1, 6),
           para("Leave-one-out disturbance ablation", "title"),
           para("Train on everything, then drop one disturbance at a time and see what the "
                "success rate does.", "subtitle"),
           rule(1.0), Spacer(1, 8),
           para("What was run", "h1"),
           para("Four setups per task. Every one is trained at six demonstration budgets "
                "(10, 25, 50, 100, 200, 400) and scored on 200 episodes, so 4 x 6 x 3 tasks = 72 "
                "policies and 14,400 episodes. Same training recipe everywhere, one seed, 80,000 "
                "steps.", "body"),
           table(WHAT, widths=[44 * mm, 42 * mm, 34 * mm, 30 * mm, CONTENT_W - 150 * mm],
                 font=8.6, pad=4.2),
           Spacer(1, 6),
           para("The three tasks are T1 cup_place (put a cup on a goal disk), T2 drawer_stow "
                "(open a drawer and stow an object in it) and T3 push_target (push a puck to a "
                "target). The change reported for a disturbance is always "
                "<b>success rate without it, minus success rate with all three on, at the same "
                "demonstration budget</b>. Positive means the simpler setup did better.", "body"),
           PageBreak()]

    # -------- 2 the answer
    st += [para("What dropping each disturbance does", "h1"),
           figure("simple02_delta.png",
                  "Change in success rate against keeping all three disturbances. Bands are 95 % "
                  "intervals. Above the dashed line the simpler setup did better.")]
    for t in TASKS:
        rows = [["demonstrations"] + [f"minus {PLAIN[a]}" for a in ("A", "B", "C")]]
        for n in NDEMOS:
            row = [str(n)]
            for axis in ("A", "B", "C"):
                r = [x for x in d["loo"] if x["task"] == t and x["axis"] == axis
                     and x["n"] == n][0]
                mark = "" if r["significant"] else "  (not clear of zero)"
                row.append(f"{r['delta']:+.3f}   [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{mark}")
            rows.append(row)
        st.append(para(TASK_LABEL[t], "h2"))
        st.append(table(rows, widths=[32 * mm] + [(CONTENT_W - 32 * mm) / 3] * 3, font=8.0,
                        pad=3.0))
        st.append(Spacer(1, 4))
    st.append(para("Read each budget on its own line. The same disturbance can matter at 25 "
                   "demonstrations and not at 400.", "small"))
    st.append(PageBreak())

    # -------- 3 the raw rates
    st += [para("The success rates behind those changes", "h1"),
           figure("simple01_success.png",
                  "Success rate of each setup. Error bars are 95 % intervals on 200 episodes.")]
    for t in TASKS:
        rows = [["demonstrations"] + [NAME[a] for a in SETUPS]]
        for n in NDEMOS:
            row = [str(n)]
            for a in SETUPS:
                c = [x for x in d["surface"] if x["task"] == t and x["arm"] == a
                     and x["n"] == n][0]
                row.append(f"{c['sr']:.3f}   ({c['k']}/200)")
            rows.append(row)
        st.append(para(TASK_LABEL[t], "h2"))
        st.append(table(rows, widths=[32 * mm] + [(CONTENT_W - 32 * mm) / 4] * 4, font=8.0,
                        pad=3.0))
        st.append(Spacer(1, 4))
    st.append(PageBreak())

    # -------- 4 T2 milestones
    st += [para("Where T2 policies got stuck", "h1"),
           para("T2 is the only task that records intermediate progress, so a failure can be "
                "placed rather than just counted. The milestones do not nest: a policy can lift "
                "the object without having opened the drawer far enough.", "body"),
           figure("simple03_stages.png",
                  "Fraction of 200 episodes reaching each milestone, per setup.")]
    rows = [["setup", "demonstrations", "opened drawer", "lifted object", "held over drawer",
             "succeeded"]]
    for a in SETUPS:
        for n in NDEMOS:
            r = [x for x in d["stages"] if x["arm"] == a and x["n"] == n][0]
            rows.append([NAME[a], str(n), f"{r['p_opened']:.3f}", f"{r['p_lifted']:.3f}",
                         f"{r['p_over']:.3f}", f"{r['p_success']:.3f}"])
    st.append(table(hdr(rows), widths=[44 * mm, 30 * mm] + [(CONTENT_W - 74 * mm) / 4] * 4,
                    font=7.8, align=[((1, 5), "RIGHT")], pad=2.4))
    st.append(PageBreak())

    # -------- 5 caveats
    st += [para("What could be wrong with these numbers", "h1")]
    rows = [["", "", ""]]
    rows = [[str(i + 1), Paragraph(f"<b>{t}</b>", S["cell"]), Paragraph(b, S["cell"])]
            for i, (t, b) in enumerate(CAVEATS)]
    st.append(table([["#", "issue", "what was measured"]] + rows,
                    widths=[8 * mm, 74 * mm, CONTENT_W - 82 * mm], font=8.0, pad=4.0))
    st.append(Spacer(1, 10))
    st.append(para("What would fix them", "h1"))
    st.append(table([["#", "next step"]]
                    + [[str(i + 1), Paragraph(t, S["cell"])] for i, t in enumerate(TODO)],
                    widths=[8 * mm, CONTENT_W - 8 * mm], font=8.0, pad=4.0))
    st.append(Spacer(1, 10))
    st.append(para("Every number here comes from src/cog/analysis/loo.py, over the 108 pooled "
                   "result files in results/. The long edition, with all six arms of the sweep, "
                   "the per-slice dispersion, timing distributions and the full flaw list, is "
                   "paper/loo_report/loo_report_full.pdf. The tables are also plain CSV in "
                   "paper/loo_report/tables/.", "small"))

    doc.build(st)
    print("wrote", PDF)


if __name__ == "__main__":
    build()
