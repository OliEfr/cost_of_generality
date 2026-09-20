#!/usr/bin/env python3
"""Build paper/loo_report/loo_report.pdf -- the standalone quantitative report.

Run with the cog_isaac interpreter and reportlab on PYTHONPATH:
    PYTHONPATH=src:$SCRATCHLIBS python -m scripts.dev.build_loo_report
Figures are rebuilt from cog.analysis.loo_figures, tables from cog.analysis.loo, so the PDF
cannot drift from the CSVs beside it.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import subprocess
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cog.analysis.loo import ARMS, NDEMOS, REMOVE, TASKS, TASK_LABEL, build  # noqa: E402
from cog.analysis import loo_figures  # noqa: E402

OUTDIR = REPO / "paper" / "loo_report"
FIGDIR = OUTDIR / "figures"
PDF = OUTDIR / "loo_report.pdf"

PAGE = landscape(A4)
MARGIN = 14 * mm
CONTENT_W = PAGE[0] - 2 * MARGIN

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#c8c8c8")
BAND = colors.HexColor("#f2f4f7")
ACCENT = colors.HexColor("#2f5d8a")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontSize=23, leading=27,
                            textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=12.5, leading=16,
                               textColor=MUTED, spaceAfter=16),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=18, textColor=ACCENT,
                         spaceBefore=6, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, leading=14, textColor=INK,
                         spaceBefore=9, spaceAfter=4),
    "body": ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.2, leading=12.6,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=5),
    "small": ParagraphStyle("small", parent=ss["BodyText"], fontSize=8.0, leading=10.6,
                            textColor=MUTED, spaceAfter=4),
    "cap": ParagraphStyle("cap", parent=ss["BodyText"], fontSize=8.2, leading=10.8,
                          textColor=MUTED, spaceBefore=2, spaceAfter=9),
    "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontSize=7.6, leading=9.6,
                           textColor=INK, spaceAfter=0),
    "hcell": ParagraphStyle("hcell", parent=ss["BodyText"], fontSize=7.2, leading=8.8,
                            textColor=INK, spaceAfter=0, fontName="Helvetica-Bold"),
}


def hdr(rows):
    """Wrap the header row in Paragraphs so long column titles wrap instead of colliding."""
    rows = [list(r) for r in rows]
    rows[0] = [Paragraph(str(c), S["hcell"]) for c in rows[0]]
    return rows


def cells(rows, cols):
    """Wrap the named column indices in Paragraphs so long values wrap."""
    out = []
    for i, r in enumerate(rows):
        r = list(r)
        if i:
            for c in cols:
                if isinstance(r[c], str):
                    r[c] = Paragraph(r[c], S["cell"])
        out.append(r)
    return out


def para(t, s="body"):
    return Paragraph(t, S[s])


def rule(h=0.7):
    t = Table([[""]], colWidths=[CONTENT_W], rowHeights=[h])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), h, RULE)]))
    return t


def table(rows, widths=None, font=7.4, align=None, header_rows=1, zebra=True,
          highlight=(), pad=2.6, wrap_all=False):
    if wrap_all:
        rows = [[Paragraph(str(c), S["cell"]) if isinstance(c, str) else c for c in r]
                for r in rows]
    t = Table(rows, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    st = [
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font + 2.2),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e6ebf2")),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.7, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dde2e8")),
    ]
    if align:
        for cols, a in align:
            st.append(("ALIGN", (cols[0], header_rows), (cols[1], -1), a))
    if zebra:
        for i in range(header_rows, len(rows)):
            if (i - header_rows) % 2 == 1:
                st.append(("BACKGROUND", (0, i), (-1, i), BAND))
    for r in highlight:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fdf0d5")))
    t.setStyle(TableStyle(st))
    return t


def figure(name, caption, width=None):
    p = FIGDIR / name
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    width = width or CONTENT_W
    return KeepTogether([Image(str(p), width=width, height=width * h / w),
                         para(caption, "cap")])


def fmt(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s


# ---------------------------------------------------------------- content


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


ARM_DEF = [
    ["arm", "disturbances active", "leave-one-out meaning", "sub-environments", "eval env per slice s"],
    ["L0", "none", "--", "1 flat", "<TASK>-L0"],
    ["L1", "A", "--", "1 flat", "<TASK>-L1"],
    ["L2", "A, B", "L3b \\ C  (variant axis removed)", "1 flat", "<TASK>-L2"],
    ["L3b", "A, B, C   (full set)", "reference for every delta", "10 variants", "<TASK>-L3v0s"],
    ["AC", "A, C", "L3b \\ B  (goal/fixture pose removed)", "10 variants", "<TASK>-ACv0s"],
    ["BC", "B, C", "L3b \\ A  (manipulandum pose removed)", "10 variants", "<TASK>-BCv0s"],
]

AXIS_DEF = [
    ["axis", "T1 cup_place", "T2 drawer_stow", "T3 push_target"],
    ["A  manipulandum pose", "cup XY over 30x40 cm, yaw +/-90 deg",
     "stow-object XY, yaw +/-45 deg", "puck XY over 12x12 cm"],
    ["B  goal / fixture pose", "goal disk XY over 20x20 cm",
     "cabinet XY +/-5 cm, yaw +/-7.5 deg", "target bearing +/-25 deg"],
    ["C  object variant", "10 cup variants (2 sizes x 5 colours)",
     "10 box variants", "10 puck variants (radius / height)"],
]

TRAIN_CFG = [
    ["setting", "value", "setting", "value"],
    ["policy", "LeRobot 0.4.4 diffusion policy", "optimiser", "Adam, lr 1e-4, wd 1e-6"],
    ["training steps", "80,000 (fixed, no early stop)", "betas / grad clip", "0.95, 0.999 / 10.0"],
    ["seed", "0  (one seed per cell)", "scheduler", "diffuser cosine, 500 warm-up steps"],
    ["batch size", "64", "diffusion", "DDPM, 100 train steps, squaredcos_cap_v2"],
    ["vision backbone", "ResNet-18, no pretrained weights", "inference steps (eval)", "10"],
    ["RGB encoders", "one per camera (2 cameras)", "observation horizon", "2 steps"],
    ["cameras", "table_cam, wrist_cam, 3x128x128", "action horizon / exec", "16 / 8"],
    ["crop / norm", "112x112, GroupNorm", "state / action dim", "9 / 7"],
    ["normalisation", "visual MEAN_STD, state+action MIN_MAX", "norm statistics", "full 400-episode pool, every budget"],
]

PROTOCOL = [
    ["property", "value"],
    ["episodes scored per cell", "200  (10 slices x 20 parallel environments)"],
    ["slices per cell", "10 independent processes, one scored batch each"],
    ["warm-up", "1 unscored batch per slice, reset seed 4900+s, outcomes discarded"],
    ["scored reset seed", "5000+s  for slice s = 0..9"],
    ["variant arms", "diagonal: slice s evaluates sub-environment v0s"],
    ["checkpoint", "step 080000 of each cell, deterministic action selection"],
    ["episode cap", "T1 600 steps, T2 1,200 steps, T3 800 steps (equals the env time-out)"],
    ["success signal", "terminated AND termination_manager.get_term('success'), evaluated per step"],
    ["eval sets", "frozen snapshots, batches 0-9 per level, never regenerated"],
    ["total scored episodes", "21,600  (plus 21,600 discarded warm-up episodes)"],
]

DATASET_FRAMES = {
    ("T1", "L0"): 82916, ("T1", "L1"): 75366, ("T1", "L2"): 74860, ("T1", "L3b"): 75438,
    ("T1", "AC"): 75294, ("T1", "BC"): 82960,
    ("T2", "L0"): 281987, ("T2", "L1"): 277661, ("T2", "L2"): 270744, ("T2", "L3b"): 269650,
    ("T2", "AC"): 277842, ("T2", "BC"): 271895,
    ("T3", "L0"): 126735, ("T3", "L1"): 127609, ("T3", "L2"): 124472, ("T3", "L3b"): 123934,
    ("T3", "AC"): 126671, ("T3", "BC"): 124093,
}

FLAWS = [
    ("F1", "Episode-level eval is not bit-reproducible",
     "Re-running one slice with an identical environment, seed, checkpoint, warm-up and code "
     "reproduced the slice total (16/20) but flipped two individual environments, and the "
     "first-success step differed on 14 of 20. Four of five independently re-run verification "
     "slices differ from their originals by one success (17 vs 16, 17 vs 16, 5 vs 4, 17 vs 18). "
     "Per-cell totals are therefore reproducible only up to binomial-scale noise of roughly "
     "+/-1 success per 20-episode slice; no single slice is an exact regression test."),
    ("F2", "One training seed per cell",
     "Every cell is seed 0. All intervals in this report are episode-sampling intervals only; "
     "they contain no training-seed variance, which is unmeasured."),
    ("F3", "Demonstrations for AC and BC were generated on different GPUs than L0-L3b",
     "AC and BC demonstrations were generated on A100-SXM-64GB (driver 535.274.02); L0, L1, L2 "
     "and L3b demonstrations were generated on an RTX 4090 (driver 580.173.02). Training and "
     "evaluation are common to all arms (same A100 partition, identical configuration). The A "
     "and B leave-one-out rows therefore compare arms whose demonstrations were rendered on "
     "different hardware; the C row does not (L2 and L3b are both 4090). A dedicated control "
     "cell (L1 demonstrations regenerated on A100, seed 1100, retrained, re-evaluated under this "
     "report's protocol) scored 0.750 [100 episodes] against the 4090-generated L1 cell's 0.860 "
     "[200 episodes]; difference -0.110, 95% CI [-0.208, -0.012]. That control varies both the "
     "GPU and the demonstration RNG stream, so it bounds rather than isolates the effect."),
    ("F4", "The C arm is not an independently generated arm",
     "L3b \\ C is bit-identical to L2 by construction: L2's fixed object is a member of the "
     "variant set, and the two environments were verified to produce identical initial states "
     "over 300 matched resets in T1 and T3 and identical configuration fields in all three "
     "tasks. The C row therefore re-uses the existing L2 cells rather than a separately "
     "generated arm, and is the only row whose two sides are episode-matched."),
    ("F5", "Pooling across demonstration budgets is statistically invalid",
     "The six budgets within one arm are six different policies on a rising curve. A "
     "homogeneity test across the six cells of an arm rejects a common proportion in most "
     "arms (see Appendix A). The budget-pooled deltas in Appendix A are reported only with "
     "their homogeneity statistic beside them; the per-budget rows in Section 4 are the "
     "primary result."),
    ("F6", "Residual first-batch depression after the warm-up batch is unmeasured",
     "One unscored warm-up batch is run in every slice of every cell, so the setting cannot "
     "differ between arms. Measured on one cell, the scored success rate rises from 0.72 with "
     "no warm-up to 0.85 with one full warm-up batch. Whether a second warm-up batch would "
     "raise it further has not been measured. One structural asymmetry remains: flat arms warm "
     "up in the same sub-environment in all ten slices, variant arms in a different one each."),
    ("F7", "The pooling guard cannot detect a mixed-generation pool",
     "scripts/ops/pool_variant_eval.py keys its MIXED_PROTOCOL check on (num_inference_steps, "
     "warm-up block), which is identical across all three evaluator generations produced during "
     "this study. The two fields that do differ (success_signal, phantom_guard_steps) are not "
     "checked, and the pooled payload drops both, so a pooled file's only provenance is its "
     "filename suffix. All 108 cells in this report were verified slice-by-slice to carry the "
     "corrected success signal; no mixed pool occurred."),
    ("F8", "The corrected success signal depends on an unasserted term ordering",
     "The per-step success flag is exact because 'success' is the last declared termination "
     "term in all three task configurations, so it always wins the manager's one-hot record. "
     "Nothing asserts this. Appending a termination term after 'success' would silently begin "
     "dropping successes."),
    ("F9", "T2 milestone flags are evaluated one step after the auto-reset boundary",
     "For the single step on which an environment terminates, the milestone tensors are read "
     "after the vectorised environment has already reset that row. Measured over all 7,180 T2 "
     "episodes the effect is not observable: zero episodes carry drawer_opened with a maximum "
     "opening below threshold, zero carry object_lifted below threshold, and zero successes "
     "are missing a milestone."),
    ("F10", "Provenance gaps in the ledgers",
     "No git revision is recorded in any run artefact (the cluster checkout carries no .git). "
     "Registry rows do not record which evaluator generation produced their success rate. "
     "Three development scripts still read the raw termination latch without the per-step "
     "guard (scripts/dev/t2_smoke.py:57, t3_smoke.py:81, sm_diag.py:69); no number in this "
     "report comes from them. Four T1 BC generation legs overshot to 41 demonstrations each "
     "(trimmed to 400 at conversion). A failed first eval-set freeze wave remains in "
     "experiments/cluster_jobs.csv beside the successful one, with no status column."),
    ("F11", "A docstring claim that does not hold for BC",
     "pool_variant_eval.py and slurm/eval.sbatch state that every arm has the same spatial "
     "coverage. BC covers 20 distinct manipulandum XY poses across its 200 episodes rather "
     "than 200, because axis A is frozen. This is by construction, not a measurement error."),
]

TODOS = [
    ("T1", "Assert the termination-term ordering that F8 depends on, in the eval entry point.",
     "open"),
    ("T2", "Extend the pooling guard to key on success_signal and phantom_guard_steps, and carry "
     "both into the pooled payload.", "open"),
    ("T3", "Write the evaluator generation into experiments/registry.csv rows and notes.", "open"),
    ("T4", "Measure the residual first-batch effect with a two-warm-up-batch slice, flat and "
     "variant arm.", "open"),
    ("T5", "Re-run the GPU-portability control with the demonstration RNG stream held fixed, so "
     "the GPU is the only varying factor (F3).", "open"),
    ("T6", "Record a git revision in every run artefact.", "open"),
    ("T7", "Add the per-step guard to the three development scripts named in F10.", "open"),
    ("T8", "Correct the spatial-coverage docstrings for BC (F11).", "open"),
    ("T9", "Merge the AC/BC generation rows into experiments/gen_stats.csv.", "done 2026-09-20"),
    ("T10", "Move the leave-one-out computation out of a scratch script into the repository "
     "(src/cog/analysis/loo.py).",
     "done 2026-09-20"),
]


def build_pdf():
    d, _ = loo_figures.build_all()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(str(PDF), pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=16 * mm,
                          title="Leave-one-out disturbance ablation -- quantitative report",
                          author="cost_of_generality")
    frame = Frame(MARGIN, 16 * mm, CONTENT_W, PAGE[1] - MARGIN - 16 * mm, id="f")

    def footer(canvas, docu):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 9 * mm,
                          "Leave-one-out disturbance ablation -- quantitative report   |   "
                          f"repo cost_of_generality @ {git_rev()}   |   "
                          f"generated {dt.date.today().isoformat()}")
        canvas.drawRightString(PAGE[0] - MARGIN, 9 * mm, f"page {docu.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN, 12 * mm, PAGE[0] - MARGIN, 12 * mm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
    st = []

    # ---------------- title
    st += [Spacer(1, 34 * mm),
           para("Leave-one-out disturbance ablation", "title"),
           para("Which single simplification of the full disturbance set changes policy success "
                "rate, measured on three manipulation tasks at six demonstration budgets", "subtitle"),
           rule(1.0), Spacer(1, 7)]
    st.append(table([
        ["scope", "3 tasks x 6 arms x 6 demonstration budgets = 108 trained policies"],
        ["evidence", "21,600 scored episodes (200 per cell), plus 21,600 discarded warm-up episodes"],
        ["question", "SR(full set minus one axis) - SR(full set), at a matched demonstration budget"],
        ["contents", "measured surface, leave-one-out deltas, within-cell dispersion, T2 milestone "
                     "instrumentation, time-to-success, generation and training provenance, "
                     "implementation flaws and open items"],
        ["excluded", "no interpretation, no ranking, no comparison against any other study or "
                     "previously published surface"],
        ["generated", f"{dt.date.today().isoformat()} from repository revision {git_rev()}"],
    ], widths=[34 * mm, CONTENT_W - 34 * mm], font=8.6, header_rows=0, zebra=True, pad=4,
       wrap_all=True))
    st.append(PageBreak())

    # ---------------- 1 design
    st += [para("1  Design and definitions", "h1"),
           para("The full disturbance set is L3b = {A, B, C}. Each leave-one-out arm removes "
                "exactly one axis from that set and keeps the other two. Every delta reported "
                "here is SR(arm with one axis removed) minus SR(L3b) at the same demonstration "
                "budget, so a positive delta means the simplified arm scored higher.", "body"),
           table(ARM_DEF, widths=[16 * mm, 40 * mm, 66 * mm, 26 * mm, CONTENT_W - 148 * mm],
                 font=8.0),
           Spacer(1, 7),
           para("1.1  What each axis randomises", "h2"),
           table(AXIS_DEF, widths=[40 * mm] + [(CONTENT_W - 40 * mm) / 3] * 3, font=8.0),
           Spacer(1, 7),
           para("1.2  Evaluation protocol", "h2"),
           table(PROTOCOL, widths=[52 * mm, CONTENT_W - 52 * mm], font=8.0),
           Spacer(1, 7),
           para("1.3  Training configuration, identical for all 108 cells", "h2"),
           table(TRAIN_CFG, widths=[38 * mm, (CONTENT_W - 76 * mm) / 2 + 38 * mm - 38 * mm,
                                    38 * mm, CONTENT_W - 38 * mm * 2 - ((CONTENT_W - 76 * mm) / 2)],
                 font=8.0),
           para("All 108 cells reached step 80,000 in a single Slurm allocation with no resume. "
                "The 126 resolved configuration keys are identical across the matrix apart from "
                "dataset path, episode list, job name, output directory and run id.", "small")]
    st.append(PageBreak())

    # ---------------- 2 provenance
    st += [para("2  Data provenance", "h1"),
           para("2.1  Demonstration generation and dataset size", "h2")]
    rows = [["task", "arm", "generation legs", "demonstrations kept", "failed attempts",
             "attempts", "generation SR", "mean demo length (steps)", "dataset episodes",
             "dataset frames"]]
    for t in TASKS:
        for a in ARMS:
            g = [r for r in d["gen"] if r["task"] == t and r["arm"] == a][0]
            rows.append([TASK_LABEL[t], a, str(g["legs"]), f"{g['demos']:,}",
                         f"{g['failures']:,}", f"{g['attempts']:,}", f"{g['gen_sr']*100:.1f} %",
                         f"{g['mean_ep_len']:.0f}", "400",
                         f"{DATASET_FRAMES[(t, a)]:,}"])
    st.append(table(hdr(rows), widths=[27 * mm, 13 * mm, 20 * mm, 27 * mm, 21 * mm, 17 * mm,
                                  21 * mm, 33 * mm, 23 * mm, CONTENT_W - 202 * mm], font=7.6,
                    align=[((2, 9), "RIGHT")]))
    st.append(para("Every dataset holds exactly 400 episodes at 20 fps. Smaller budgets are "
                   "contiguous prefixes of the same episode order, which is round-robin over the "
                   "ten variants, so every budget of a variant arm is variant-balanced "
                   "(N=10 gives 1 episode per variant, N=400 gives 40). BC in T1 produced 404 "
                   "demonstrations across its ten legs; the four surplus episodes fall outside "
                   "the 400-episode prefix and are unused.", "small"))
    st.append(figure("fig10_generation.png",
                     "Figure 10. Demonstration yield per arm. Left: fraction of generation "
                     "attempts that produced a usable demonstration. Right: attempts required. "
                     "Every cell was filled to exactly 400 demonstrations."))
    st.append(PageBreak())

    # ---------------- 3 surface
    st += [para("3  Measured surface", "h1"),
           para("All 108 cells, each 200 scored episodes. Intervals are Wilson score intervals "
                "at 95 % on the 200 episodes of that cell.", "body"),
           figure("fig01_surface_curves.png",
                  "Figure 1. Success rate against demonstration budget for all six arms of each "
                  "task. Error bars are Wilson 95 % intervals on 200 episodes."),
           figure("fig02_surface_heatmap.png",
                  "Figure 2. The same 108 measurements as a grid. Cell values are success "
                  "rates over 200 episodes."),
           PageBreak(),
           para("3.1  Cell table", "h2")]
    for t in TASKS:
        rows = [["arm"] + [f"N={n}" for n in NDEMOS]]
        for a in ARMS:
            row = [a]
            for n in NDEMOS:
                c = [r for r in d["surface"] if r["task"] == t and r["arm"] == a and r["n"] == n][0]
                row.append(f"{c['sr']:.3f}   {c['k']}/200<br/>"
                           f"<font size=6 color='#5a5a5a'>[{c['wilson_lo']:.3f}, {c['wilson_hi']:.3f}]</font>")
            rows.append(row)
        rows = [[Paragraph(str(x), S["cell"]) if i else Paragraph(f"<b>{x}</b>", S["cell"])
                 for i, x in enumerate(r)] for r in rows]
        st.append(para(TASK_LABEL[t], "h2"))
        st.append(table(rows, widths=[16 * mm] + [(CONTENT_W - 16 * mm) / 6] * 6, font=7.6))
        st.append(Spacer(1, 5))
    st.append(PageBreak())

    # ---------------- 4 loo
    st += [para("4  Leave-one-out results", "h1"),
           para("Each row removes one axis from the full set L3b and reports the change in "
                "success rate at a matched demonstration budget. Intervals are Newcombe 95 % "
                "intervals for the difference of two independent proportions, except where the "
                "two sides are episode-matched (axis C), where the paired McNemar statistic is "
                "given as well.", "body"),
           figure("fig03_loo_delta_vs_n.png",
                  "Figure 3. Leave-one-out delta against demonstration budget. Shaded bands are "
                  "Newcombe 95 % intervals. The dashed line is no change."),
           PageBreak(),
           figure("fig04_loo_forest.png",
                  "Figure 4. All 54 leave-one-out deltas with their intervals. Faded rows are "
                  "those whose interval covers zero."),
           PageBreak(),
           para("4.1  Delta table", "h2")]
    rows = [["task", "axis removed", "arm", "N", "SR L3b", "SR reduced", "delta",
             "Newcombe 95 %", "covers 0", "paired McNemar (axis C only)"]]
    hl = []
    for i, r in enumerate(d["loo"]):
        mc = r.get("mcnemar")
        mcs = (f"b={mc['b']}  c={mc['c']}  d={mc['diff']:+.3f}  "
               f"[{mc['lo']:+.3f}, {mc['hi']:+.3f}]  chi2={mc['chi2_cc']:.2f}") if mc else ""
        rows.append([TASK_LABEL[r["task"]], r["axis"], r["removed_arm"], str(r["n"]),
                     f"{r['sr_full']:.3f}", f"{r['sr_reduced']:.3f}", fmt(r["delta"], 3, True),
                     f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]",
                     "no" if r["significant"] else "yes", mcs])
        if r["significant"]:
            hl.append(i + 1)
    st.append(table(hdr(rows), widths=[26 * mm, 17 * mm, 13 * mm, 11 * mm, 17 * mm, 20 * mm,
                                  16 * mm, 32 * mm, 16 * mm, CONTENT_W - 168 * mm],
                    font=7.0, align=[((3, 8), "RIGHT")], highlight=hl, pad=1.9))
    st.append(para("Shaded rows are those whose 95 % interval excludes zero.", "small"))
    st.append(PageBreak())

    # ---------------- 5 dispersion
    st += [para("5  Within-cell dispersion", "h1"),
           para("Every cell is ten independent processes of 20 episodes. The spread of those ten "
                "slice rates is the sampling variability behind each pooled cell value.", "body"),
           figure("fig05_slice_dispersion.png",
                  "Figure 5. The ten slice success rates behind every cell (dots) and the pooled "
                  "cell value (black bar)."),
           para("5.1  Homogeneity of the ten slices within a cell", "h2")]
    over = [r for r in d["surface"] if r["slice_overdispersed"]]
    rows = [["task", "arm", "N", "pooled SR", "slice min", "slice max", "chi2 (df 9)",
             "rejects common rate at 95 %"]]
    for r in sorted(d["surface"], key=lambda r: -r["slice_chi2"])[:14]:
        rows.append([TASK_LABEL[r["task"]], r["arm"], str(r["n"]), f"{r['sr']:.3f}",
                     f"{r['slice_min']:.2f}", f"{r['slice_max']:.2f}", f"{r['slice_chi2']:.1f}",
                     "yes" if r["slice_overdispersed"] else "no"])
    st.append(table(hdr(rows), widths=[27 * mm, 14 * mm, 12 * mm, 20 * mm, 18 * mm, 18 * mm,
                                  22 * mm, CONTENT_W - 131 * mm], font=7.6,
                    align=[((2, 6), "RIGHT")]))
    st.append(para(f"{len(over)} of the 108 cells reject a common rate across their ten slices "
                   f"at 95 % (critical value 16.92 on 9 degrees of freedom). The fourteen "
                   f"largest statistics are listed.", "small"))
    st.append(PageBreak())

    # ---------------- 6 stages
    st += [para("6  T2 drawer_stow milestone instrumentation", "h1"),
           para("T2 is the only task instrumented with intermediate milestones; T1 and T3 record "
                "binary success only. A milestone latches when its condition is met at any point "
                "in the episode: drawer opened past 0.15 m, object lifted past 0.05 m, object "
                "above the drawer cavity. All rates are over the same 200 episodes as the "
                "success rate.", "body"),
           figure("fig06_t2_stage_funnel.png",
                  "Figure 6. Absolute milestone rates per arm and budget, with success overlaid."),
           PageBreak(),
           figure("fig07_t2_conditional.png",
                  "Figure 7. Conditional rate of reaching each milestone given the previous one."),
           figure("fig09_t2_progress_magnitudes.png",
                  "Figure 9. Continuous progress measures over all 1,200 episodes of each arm "
                  "(six budgets pooled). Dashed lines mark the milestone thresholds."),
           PageBreak(),
           para("The three milestones are not nested: an episode can lift the object without ever "
                "having opened the drawer past 0.15 m. Every conditional rate below is therefore "
                "computed from the joint count (episodes carrying both flags divided by episodes "
                "carrying the first), not from the ratio of two marginal rates. The joint counts "
                "themselves are in paper/loo_report/tables/stages.csv.", "small"),
           para("6.1  Milestone table", "h2")]
    rows = [["arm", "N", "drawer opened", "object lifted", "object over drawer", "success",
             "lift | opened", "over | lifted", "success | over", "median max opening (m)",
             "median max lift (m)"]]
    for r in d["stages"]:
        rows.append([r["arm"], str(r["n"]), f"{r['p_opened']:.3f}", f"{r['p_lifted']:.3f}",
                     f"{r['p_over']:.3f}", f"{r['p_success']:.3f}",
                     fmt(r["lift_given_open"]), fmt(r["over_given_lift"]),
                     fmt(r["success_given_over"]), f"{r['max_open_median']:.4f}",
                     f"{r['max_lift_median']:.4f}"])
    st.append(table(hdr(rows), widths=[13 * mm, 11 * mm, 24 * mm, 22 * mm, 30 * mm, 18 * mm,
                                  21 * mm, 21 * mm, 24 * mm, 36 * mm, CONTENT_W - 220 * mm],
                    font=7.0, align=[((1, 10), "RIGHT")], pad=1.9))
    st.append(PageBreak())

    # ---------------- 7 timing
    st += [para("7  Time to success", "h1"),
           para("Step index at which a successful episode first satisfied the success condition, "
                "over successful episodes only, all budgets pooled.", "body"),
           figure("fig08_time_to_success.png",
                  "Figure 8. Distribution of the first-success step per arm. Boxes are quartiles, "
                  "whiskers 1.5 IQR, outliers suppressed."),
           Spacer(1, 4)]
    rows = [["task", "arm", "successful episodes", "min", "p25", "median", "p75", "max",
             "episode cap"]]
    caps = {"T1": 600, "T2": 1200, "T3": 800}
    for r in d["timing"]:
        rows.append([TASK_LABEL[r["task"]], r["arm"], f"{r['n_success']:,}", f"{r['min']:.0f}",
                     f"{r['p25']:.0f}", f"{r['median']:.0f}", f"{r['p75']:.0f}",
                     f"{r['max']:.0f}", str(caps[r["task"]])])
    st.append(table(hdr(rows), widths=[27 * mm, 14 * mm, 30 * mm, 16 * mm, 16 * mm, 18 * mm,
                                  16 * mm, 16 * mm, CONTENT_W - 153 * mm], font=7.4,
                    align=[((2, 8), "RIGHT")], pad=2.0))
    st.append(PageBreak())

    # ---------------- 8 flaws
    st += [para("8  Implementation flaws", "h1"),
           para("Each entry is a property of how this study was implemented or measured, with "
                "the quantity that establishes it. They are listed, not weighted.", "body")]
    rows = [["id", "flaw", "evidence"]]
    for i, (fid, title, body) in enumerate(FLAWS):
        rows.append([fid, Paragraph(f"<b>{title}</b>", S["cell"]), Paragraph(body, S["cell"])])
    st.append(table(rows, widths=[11 * mm, 62 * mm, CONTENT_W - 73 * mm], font=7.6, pad=3.4))
    st.append(PageBreak())

    st += [para("9  Open items", "h1")]
    rows = [["id", "item", "status"]]
    for tid, text, status in TODOS:
        rows.append([tid, Paragraph(text, S["cell"]), Paragraph(status, S["cell"])])
    st.append(table(rows, widths=[11 * mm, CONTENT_W - 61 * mm, 50 * mm], font=7.8, pad=3.2))
    st.append(PageBreak())

    # ---------------- appendix
    st += [para("Appendix A  Budget-pooled deltas", "h1"),
           para("The same deltas with all six budgets summed into one 1,200-episode proportion "
                "per side. This pooling is reported for completeness only: the six budgets are "
                "six different policies, and the homogeneity statistic beside each row tests "
                "whether they share one success rate (critical value 11.07 on 5 degrees of "
                "freedom). Where that test rejects, the pooled interval describes a mixture "
                "whose value depends on which budgets were included, not a single parameter.",
                "body")]
    rows = [["task", "axis removed", "arm", "successes L3b", "successes reduced", "episodes/side",
             "pooled delta", "Newcombe 95 %", "chi2 L3b (df 5)", "chi2 reduced (df 5)",
             "homogeneity rejected"]]
    for r in d["pooled"]:
        rows.append([TASK_LABEL[r["task"]], r["axis"], r["removed_arm"], f"{r['k_full']:,}",
                     f"{r['k_reduced']:,}", f"{r['episodes']:,}", fmt(r["delta"], 3, True),
                     f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", f"{r['chi2_full']:.1f}",
                     f"{r['chi2_reduced']:.1f}", "yes" if r["overdispersed"] else "no"])
    st.append(table(hdr(rows), widths=[26 * mm, 17 * mm, 13 * mm, 24 * mm, 28 * mm, 23 * mm,
                                  20 * mm, 32 * mm, 24 * mm, 28 * mm, CONTENT_W - 235 * mm],
                    font=7.0, align=[((3, 10), "RIGHT")], pad=2.2))
    st.append(Spacer(1, 8))
    st.append(para("Appendix B  Files behind this report", "h1"))
    rows = [["path", "contents"]]
    for p, c in [
        ("results/eval_&lt;TASK&gt;_&lt;ARM&gt;_n&lt;N&gt;_080000_u200d32.json",
         "the 108 pooled cells, each carrying its 200 per-episode outcome records"),
        ("paper/loo_report/tables/surface.csv", "108 cells: successes, Wilson interval, slice spread, homogeneity"),
        ("paper/loo_report/tables/loo.csv", "54 leave-one-out rows with Newcombe and McNemar statistics"),
        ("paper/loo_report/tables/pooled.csv", "9 budget-pooled rows with homogeneity statistics"),
        ("paper/loo_report/tables/stages.csv", "36 T2 milestone rows"),
        ("paper/loo_report/tables/timing.csv", "18 first-success timing summaries"),
        ("paper/loo_report/tables/gen.csv", "18 generation-yield rows"),
        ("src/cog/analysis/loo.py", "every statistic in this report"),
        ("src/cog/analysis/loo_figures.py", "every figure in this report"),
        ("scripts/dev/build_loo_report.py", "this document"),
    ]:
        rows.append([Paragraph(f"<font face='Courier'>{p}</font>", S["cell"]),
                     Paragraph(c, S["cell"])])
    st.append(table(rows, widths=[110 * mm, CONTENT_W - 110 * mm], font=7.6, pad=3.0))

    doc.build(st)
    print("wrote", PDF)
    return PDF


if __name__ == "__main__":
    build_pdf()
