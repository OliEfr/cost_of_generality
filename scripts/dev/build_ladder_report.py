#!/usr/bin/env python3
"""Build paper/ladder_report/ladder_report.pdf -- the additive-ladder study, short edition."""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

from reportlab.lib.pagesizes import A4, landscape  # noqa: F401
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer)

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "dev"))

from cog.analysis.loo import NDEMOS, TASKS, TASK_LABEL  # noqa: E402
from cog.analysis import ladder_figures  # noqa: E402
from cog.analysis.ladder import AXIS_PLAIN, RUNGS, SETUP_NAME, SETUPS  # noqa: E402
from build_loo_report import (CONTENT_W, MARGIN, MUTED, PAGE, RULE, S, git_rev, hdr,  # noqa: E402
                              para, rule, table)

PDF = REPO / "paper" / "ladder_report" / "ladder_report.pdf"
FIGDIR = REPO / "paper" / "ladder_report" / "figures"


def figure(name, caption, width=None):
    from PIL import Image as PILImage
    p = FIGDIR / name
    w, h = PILImage.open(p).size
    width = width or CONTENT_W
    return KeepTogether([Image(str(p), width=width, height=width * h / w),
                         para(caption, "cap")])


WHAT = [
    ["rung", "object start position", "goal position", "object type", "name on disk"],
    ["nothing varies", "FIXED", "FIXED", "1 type", "L0"],
    ["+ object start", "varies", "FIXED", "1 type", "L1"],
    ["+ goal", "varies", "varies", "1 type", "L2"],
    ["+ object type", "varies", "varies", "10 types", "L3b"],
]

CAVEATS = [
    ("Each rung is measured on top of the rungs below it.",
     "The cost of switching on the goal is measured with the object start already varying, and "
     "the cost of object type with both of the others already on. A different order would give "
     "different numbers. This report measures one order, the one the ladder defines."),
    ("One training seed.",
     "Every policy is seed 0. The intervals here cover episode sampling only; they contain no "
     "training-run variance."),
    ("Repeating an evaluation does not give the identical episodes back.",
     "Re-running one 20-episode slice with the same seed and checkpoint reproduced the slice "
     "total but flipped two individual episodes. Expect about +/-1 success per 20 episodes of "
     "jitter."),
    ("Do not average a change across demonstration budgets.",
     "The six budgets are six different policies. A test for a shared success rate is rejected "
     "in all 18 rung-task pairs, so an average over budgets describes the mix of budgets chosen, "
     "not the rung."),
    ("These numbers replace an earlier scoring of the same policies.",
     "The success counter used to read a flag that stayed latched across episode boundaries, "
     "which credited successes that never happened. All 108 cells were re-scored after the fix "
     "and this report uses only the corrected results. Older surfaces in the repository for the "
     "same cells are superseded."),
]

TODO = [
    "Train more than one seed on whichever rungs matter.",
    "Measure the ladder in a second order, so the conditioning in caveat 1 can be quantified.",
    "Record the evaluator version in the results ledger, and which git revision each run used.",
    "Measure whether a second warm-up batch moves the scored success rate.",
    "Mark the superseded surface files in the repository so they cannot be picked up by mistake.",
]


def build():
    d, _ = ladder_figures.build_all()
    doc = BaseDocTemplate(str(PDF), pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=16 * mm,
                          title="Additive disturbance ladder", author="cost_of_generality")
    frame = Frame(MARGIN, 16 * mm, CONTENT_W, PAGE[1] - MARGIN - 16 * mm, id="f")

    def footer(canvas, docu):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 9 * mm, f"Additive disturbance ladder   |   cost_of_generality "
                                          f"@ {git_rev()}   |   {dt.date.today().isoformat()}")
        canvas.drawRightString(PAGE[0] - MARGIN, 9 * mm, f"page {docu.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN, 12 * mm, PAGE[0] - MARGIN, 12 * mm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
    st = []

    st += [Spacer(1, 6),
           para("Additive disturbance ladder", "title"),
           para("Start with nothing varying, switch the disturbances on one at a time, and see "
                "what each one costs.", "subtitle"),
           rule(1.0), Spacer(1, 8),
           para("What was run", "h1"),
           para("Four rungs per task. Every rung is trained at six demonstration budgets "
                "(10, 25, 50, 100, 200, 400) and scored on 200 episodes, so 4 x 6 x 3 tasks = 72 "
                "policies and 14,400 episodes. Same training recipe everywhere, one seed, 80,000 "
                "steps.", "body"),
           table(WHAT, widths=[40 * mm, 42 * mm, 34 * mm, 30 * mm, CONTENT_W - 146 * mm],
                 font=8.6, pad=4.2),
           Spacer(1, 6),
           para("The three tasks are T1 cup_place (put a cup on a goal disk), T2 drawer_stow "
                "(open a drawer and stow an object in it) and T3 push_target (push a puck to a "
                "target). The cost reported for a disturbance is always <b>success rate with it "
                "on, minus success rate one rung below, at the same demonstration budget</b>. "
                "Negative means switching it on made the policy worse. All four rungs' "
                "demonstrations were generated on the same GPU, so no comparison here crosses a "
                "hardware boundary.", "body"),
           PageBreak()]

    st += [para("What each disturbance costs", "h1"),
           figure("ladder02_delta.png",
                  "Change in success rate when a disturbance is switched on, given the rungs "
                  "below it. Bands are 95 % intervals. Below the dashed line it made things "
                  "worse.")]
    for t in TASKS:
        rows = [["demonstrations"] + [f"switch on {AXIS_PLAIN[a]}" for a in RUNGS]]
        for n in NDEMOS:
            row = [str(n)]
            for axis in RUNGS:
                r = [x for x in d["ladder"] if x["task"] == t and x["axis"] == axis
                     and x["n"] == n][0]
                mark = "" if r["significant"] else "  (not clear of zero)"
                row.append(f"{r['delta']:+.3f}   [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{mark}")
            rows.append(row)
        st.append(para(TASK_LABEL[t], "h2"))
        st.append(table(rows, widths=[32 * mm] + [(CONTENT_W - 32 * mm) / 3] * 3, font=8.0,
                        pad=3.0))
        st.append(Spacer(1, 4))
    st.append(para("Read each budget on its own line. The same disturbance can cost a lot at 25 "
                   "demonstrations and nothing at 400.", "small"))
    st.append(PageBreak())

    st += [para("The success rates behind those costs", "h1"),
           figure("ladder01_success.png",
                  "Success rate at each rung. Error bars are 95 % intervals on 200 episodes.")]
    for t in TASKS:
        rows = [["demonstrations"] + [SETUP_NAME[a] for a in SETUPS]]
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

    st += [para("Where T2 policies got stuck", "h1"),
           para("T2 is a four-part task: open the drawer, pick up the object, carry it over the "
                "open drawer, drop it in. Every episode records how far it got, so a failure can "
                "be placed instead of just counted. Each bar below is 200 episodes split by the "
                "furthest point reached, so the five shares add to 100.", "body"),
           figure("ladder03_funnel.png",
                  "How far each episode got before it stopped. Numbers are percentages of 200 "
                  "episodes."),
           Spacer(1, 4)]
    rows = [["rung", "demos", "never opened", "opened, stopped", "lifted, stopped",
             "held over, stopped", "stowed it"]]
    for a in SETUPS:
        for n in NDEMOS:
            r = [x for x in d["furthest"] if x["arm"] == a and x["n"] == n][0]
            rows.append([SETUP_NAME[a], str(n), str(r["got nowhere"]), str(r["opened drawer"]),
                         str(r["lifted object"]), str(r["held over drawer"]),
                         str(r["succeeded"])])
    st.append(table(hdr(rows), widths=[40 * mm, 22 * mm] + [(CONTENT_W - 62 * mm) / 5] * 5,
                    font=7.8, align=[((1, 6), "RIGHT")], pad=2.4))
    st.append(para("Counts out of 200 episodes.", "small"))
    st.append(PageBreak())

    st += [para("What could be wrong with these numbers", "h1")]
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
    st.append(para("Every number here comes from src/cog/analysis/ladder.py, over the pooled "
                   "result files in results/. The tables are also plain CSV in "
                   "paper/ladder_report/tables/.", "small"))

    doc.build(st)
    print("wrote", PDF)


if __name__ == "__main__":
    build()
