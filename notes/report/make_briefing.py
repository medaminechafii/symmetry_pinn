"""Briefing documents for the meeting with the professor:
    professor_handout.pdf   overview of the research, the problems met and how they were solved, open questions
    meeting_prep.pdf        for the student: pitch, likely questions with answers, key numbers, glossary

    <python with reportlab> notes/report/make_briefing.py
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
                                CondPageBreak)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FONTS = next((ROOT / ".venv" / "lib").glob("python*/site-packages/matplotlib/mpl-data/fonts/ttf"))
for n, f in (("DV", "DejaVuSans"), ("DVB", "DejaVuSans-Bold"), ("DVI", "DejaVuSans-Oblique")):
    pdfmetrics.registerFont(TTFont(n, str(FONTS / f"{f}.ttf")))
registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK, MUTED, BLUE = colors.HexColor("#222222"), colors.HexColor("#555555"), colors.HexColor("#2F5F98")
S = {
    "title": ParagraphStyle("t", fontName="DVB", fontSize=18, leading=23, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("s", fontName="DV", fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=13.5, leading=17, textColor=BLUE, spaceBefore=9, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=11, leading=14, textColor=INK, spaceBefore=6, spaceAfter=3),
    "body": ParagraphStyle("b", fontName="DV", fontSize=9.6, leading=13.5, textColor=INK, spaceAfter=4),
    "small": ParagraphStyle("sm", fontName="DV", fontSize=8.3, leading=11.3, textColor=MUTED, spaceAfter=3),
    "eq": ParagraphStyle("eq", fontName="DV", fontSize=10.5, leading=15, alignment=TA_CENTER, spaceBefore=2, spaceAfter=4),
    "cell": ParagraphStyle("c", fontName="DV", fontSize=8.3, leading=10.8, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="DVB", fontSize=8.3, leading=10.8, textColor=INK),
    "box": ParagraphStyle("bx", fontName="DV", fontSize=9.0, leading=12.6, textColor=INK),
}


class Doc:
    def __init__(self):
        self.story = []

    def p(self, t, st="body"):
        self.story.append(Paragraph(t, S[st]))

    def h1(self, t):
        self.story.append(CondPageBreak(50 * mm)); self.p(t, "h1")

    def h2(self, t):
        self.story.append(CondPageBreak(30 * mm)); self.p(t, "h2")

    def eq(self, t):
        self.p(t.replace("  ", "&nbsp;&nbsp;"), "eq")

    def box(self, title, text, col="#2F5F98", bg="#EEF3FA"):
        t = Table([[Paragraph(f"<b>{title}</b><br/>{text}", S["box"])]], colWidths=[170 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                               ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(col)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        self.story.append(t); self.story.append(Spacer(1, 6))

    def table(self, rows, widths, head=True):
        data = [[Paragraph(str(c), S["cellb" if (head and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
              ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if head:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF2")))
        t.setStyle(TableStyle(st)); self.story.append(t); self.story.append(Spacer(1, 7))

    def figure(self, path, caption, w=170):
        from PIL import Image as PILImage                          # only for the aspect ratio
        with PILImage.open(path) as im:
            r = im.height / im.width
        self.story.append(KeepTogether([Image(str(path), width=w * mm, height=w * mm * r), Paragraph(caption, S["small"])]))

    def build(self, name, header):
        def footer(c, d):
            c.saveState(); c.setFont("DV", 7.5); c.setFillColor(MUTED)
            c.drawString(20 * mm, 10 * mm, header); c.drawRightString(190 * mm, 10 * mm, str(d.page)); c.restoreState()
        doc = SimpleDocTemplate(str(HERE / name), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                                topMargin=16 * mm, bottomMargin=18 * mm, title=header)
        doc.build(self.story, onFirstPage=footer, onLaterPages=footer)
        print("wrote", HERE / name)


# =============================================================================================================
# 1. Handout for the professor
# =============================================================================================================
d = Doc()
d.p("Discovering and using symmetries in physics-informed neural networks when the equation is only partly known",
    "title")
d.p("Graduation research (kōsen), status 27 November 2026. Meeting handout: research overview, problems met on the way "
    "to the current method, and open questions on which advice would help.", "sub")

d.h1("1. Research question")
d.p("Symmetry-informed PINNs add a known symmetry of the governing equation as an extra loss and are more accurate with "
    "little data. In practice the equation is often only <b>partly</b> known, so its symmetries are not known either. "
    "<b>Question:</b> can the symmetries be discovered from sparse, noisy data of a partly unknown equation, and can they "
    "then be used to improve the learned equation itself?")

d.h1("2. Test bed")
d.eq("u<sub>t</sub> + u u<sub>x</sub> = ν u<sub>xx</sub> + g(u),    ν = 0.1,  x periodic,  t ∈ [0, 2]")
d.p("Burgers' equation with an <b>unknown source g(u)</b>; ν is also learned. Its Lie symmetries are known (Olver), and "
    "the choice of g switches them on and off, so every result can be checked against the truth:")
d.table([["data set", "true g", "non-trivial symmetries", "expected result"],
         ["g0", "0", "Galilean, scaling, projective", "find 3"],
         ["cubic", "−u³", "scaling", "find 1"],
         ["ε = 0.1", "−u³ + 0.1u", "none (scaling is false but close)", "find 0: false-positive control"],
         ["ε = 0.03", "−u³ + 0.03u", "none (even closer)", "probably below the resolution"]], [22, 26, 64, 58])
d.p("Data: 3 initial conditions per data set; N = 50, 200 or 1000 scattered observations; noise σ = 0, 1 or 3 % of std(u).")
d.p("<b>Learned equation (hybrid PINN):</b> a network u(x, t) is fitted with L<sub>data</sub> + λL<sub>PDE</sub>, where "
    "the PDE residual uses a learned source ĝ (polynomial c<sub>1</sub>u + … + c<sub>5</sub>u<super>5</super>, or a small "
    "MLP) and a learned viscosity ν̂. <b>Symmetry condition:</b> a generator V = ξ∂<sub>x</sub> + τ∂<sub>t</sub> + φ∂<sub>u</sub> "
    "is written in a library of 30 basis fields; the infinitesimal criterion pr V(Δ) = 0 (with u<sub>t</sub> substituted "
    "from the learned equation) becomes a linear system A w = 0, solved by SVD. The smallest singular vectors are "
    "candidate symmetries.")

d.story.append(CondPageBreak(130 * mm))
d.h1("3. Current pipeline")
d.figure(ROOT / "notes" / "figures" / "symmetry_loop_v2.png",
         "Figure 1. Each SVD candidate is cleaned (small entries set to 0) or sparsified (LieNLSD-style rotation), trained "
         "<i>alone</i> on top of the baseline with its non-zero coefficients learnable, and accepted if the error on held-out "
         "data (t &gt; 1.5) does not rise by more than 10 % and the candidate moved less than 10°. Accepted symmetries are "
         "trained together; the new model's SVD may reveal more candidates (second pass).")

d.h1("4. Problems met and how they were solved")
d.table([["#", "problem", "evidence", "what we changed"],
         ["1", "Counting small singular values does not find the symmetries of a <i>learned</i> equation",
          "gap 0.1 – 1.2 instead of 14; a 1.5 % breaking of the cubic term already drops it to 2; the learned ĝ breaks "
          "symmetries by a few % (mostly the u<super>5</super> term)",
          "use the SVD only to <b>propose</b> candidates; decide on each candidate separately"],
         ["2", "Would a known symmetry even help? Or is it just shrinking ĝ?",
          "true generators as a loss: g0 err_g 0.165 → 0.0003, cubic 0.217 → 0.156; weight decay never matches and "
          "damages cubic",
          "symmetry loss kept; the gain is not shrinkage"],
         ["3", "The SVD cannot reject a false symmetry",
          "near-symmetric ε data: the detector prefers the truly symmetric twin in only 11 – 56 % of pairs; but a "
          "candidate-specific measure recovers ε = 0.1",
          "test each candidate with its own measure"],
         ["4", "How to test a candidate without circularity",
          "training loss is insensitive (fits to noise level); a time-window held-out set: true scaling R 0.95 – 0.99 "
          "(1 %), false Galilean R 11 – 12",
          "held-out window t &gt; 1.5, R = E<sub>cand</sub>/E<sub>base</sub>, accept if R ≤ 1.1 (δ fixed from a pilot)"],
         ["5", "Noise-free: the baseline error varies up to 12× between seeds; later steps rejected by luck",
          "true g0 steps compared with the previous model: R 1.16 – 1.49",
          "reference = better of 2 baseline seeds; every candidate tested <b>alone</b> against the baseline"],
         ["6", "True candidates rejected because they are a few degrees off",
          "cubic scaling candidate (2 – 5° off) R 4 – 19; false scaling on ε = 0.1 R 13 – 77: overlap",
          "cleaning (not enough: the ratio −1.04 instead of −1 remains) → <b>learnable coefficients</b>: true R 0.042, "
          "false R 2325"],
         ["7", "Learnable candidates sometimes jump to a different symmetry", "drifts of 40 – 90° (duplicates of scaling)",
          "reject if drift &gt; 10°; skip duplicates"],
         ["8", "g0 candidates are mixtures of the 3 true symmetries", "single candidates 35 – 90° from each true one",
          "LieNLSD sparsifying rotation: exact scaling, Galilean and projective, ĝ = 0, ν error 0.136 % → 0.002 %"]],
        [6, 42, 66, 56])

d.h1("5. Main results so far")
d.table([["experiment", "result"],
         ["learned ĝ, ν̂ (N = 1000, 1 %)", "g within ~5 %, ν within 0.25 %; degrades smoothly with fewer points / more noise"],
         ["true symmetries as a loss (upper bound)", "form of ĝ fixed (spurious terms → 0); size of the cubic term not fixed (couples to ν)"],
         ["latest loop test (noise-free, one draw)", "cubic: exactly scaling accepted · ε = 0.1: nothing accepted · g0: all three exactly"],
         ["running now", "432 chains: 4 data sets × noise 0/1/3 % × N 1000/200/50 × 3 draws × poly/MLP × clean/sparse"]],
        [58, 112])
d.p("Caveat: the latest results are noise-free and one data draw each; the full evaluation is in progress.", "small")

d.h1("6. Questions on which advice would help")
d.table([["#", "question", "context"],
         ["1", "Is “refine a discovered generator until it can be enforced, and keep it only if held-out prediction "
               "does not suffer” a reasonable contribution?",
          "LieNLSD (ICML 2025) finds generators to ~1e-2 but never feeds them back; NeurIPS 2024 work regularises with "
          "discovered symmetries. Our finding: discovered generators are not accurate enough to be enforced as they are."],
         ["2", "How should thresholds be justified? (δ = 0.1, drift 10°, cleaning 0.1)",
          "All fixed before the runs from small pilots; δ calibrated only at N = 1000, noise 0 and 1 %."],
         ["3", "Learnable coefficients rely on known terms (u u<sub>x</sub>) fixing the generator's ratios. How to keep "
               "the test strong when they do not (e.g. u<sub>t</sub> = νu<sub>xx</sub> + g(u))?",
          "There the candidate and ĝ could co-adapt and accept a false symmetry with an odd exponent."],
         ["4", "A cubic source and extra viscosity are hard to tell apart with sparse data (c<sub>3</sub> moves with ν̂). "
               "Is there a principled remedy?", "Scaling fixes the form of ĝ, not its size."],
         ["5", "With noise, the held-out error is dominated by noise (R ≈ 1 even when the true error drops 4–8×). "
               "Is there a better test statistic?", "Noise-free tests separate by factors of 10<super>3</super> – 10<super>5</super>."],
         ["6", "Unexplained: the singular-value gap is 0.6 – 0.8 decades below the prediction from the measured "
               "symmetry breaking, in every setting.", "Not needed for the method anymore, but unexplained."],
         ["7", "What evaluation / baselines would make the result convincing?",
          "Planned: true-generator upper bound, no-symmetry baseline, weight decay, false-positive data (ε)."]],
        [6, 84, 80])
d.build("professor_handout.pdf", "Meeting handout: symmetry discovery in hybrid PINNs, 27 Nov 2026")

# =============================================================================================================
# 2. Preparation sheet for the student
# =============================================================================================================
d = Doc()
d.p("Meeting preparation: what to say and what you may be asked", "title")
d.p("For you, not for handing out. Read together with the two-week report and the handout.", "sub")

d.h1("1. One-minute pitch")
d.box("Say it like this",
      "“My research is about PINNs that use symmetries of the equation. Normally you must know the symmetries in "
      "advance. I study the case where part of the equation is unknown: Burgers' equation with an unknown source term. "
      "First a PINN learns the unknown term and the viscosity from sparse noisy data. From the learned equation I "
      "compute candidate symmetries with a linear-algebra method, like the LieNLSD paper. The problem I found is that "
      "these candidates are a few degrees off, and if you enforce them as they are, they make the model worse, exactly "
      "like a false symmetry. So I make the candidate's few coefficients trainable and test every candidate on held-out "
      "data. In the first tests this accepts the true symmetries exactly and rejects a near-true false one. A large "
      "evaluation is running now.”", "#3B6D11", "#EFF6E8")

d.h1("2. Three-minute walkthrough (follow Figure 1 of the handout)")
for i, t in enumerate([
        "<b>Test bed.</b> Burgers with unknown g(u); four data sets with 3, 1, 0, 0 symmetries; sparse noisy data.",
        "<b>Baseline.</b> Hybrid PINN learns u, ĝ, ν̂. Two seeds, keep the one with the smaller held-out error.",
        "<b>Propose.</b> Symmetry condition → 30-column matrix → SVD → smallest singular vectors are candidates.",
        "<b>Why not just count small singular values?</b> The learned ĝ breaks the symmetry by a few %, the test reacts to 1 %.",
        "<b>Test.</b> Each candidate alone: retrain with its symmetry loss; accept if held-out error rises ≤ 10 %.",
        "<b>The key problem and fix.</b> Candidates are a few degrees off → rejected like false ones. Clean/sparsify, make "
        "the coefficients trainable, reject if they drift more than 10°.",
        "<b>Result so far.</b> Noise-free: cubic → scaling exactly; ε = 0.1 → nothing; g0 → all three exactly."], 1):
    d.p(f"{i}. {t}")

d.h1("3. Questions you may be asked, with short answers")
d.table([["question", "short answer"],
         ["Why Burgers' equation?", "Nonlinear, symmetries known exactly, and the source g lets us choose which symmetries exist, so we can always check."],
         ["Why a PINN and not finite differences like LieNLSD?", "Our data are sparse and noisy; finite differences need dense clean grids. The PINN also gives the learned ĝ, ν̂."],
         ["Isn't using the form u_t + u u_x = ν u_xx + g(u) cheating?", "We assume the known physics (transport + diffusion) and learn only the unknown part. The true symmetries and the true g are never used; they are only for scoring."],
         ["Is the learnable candidate truth leakage?", "No true symmetry enters. It uses the assumed terms of the equation, which fix the allowed ratios. For equations where they don't, the test is weaker; that's a stated limitation."],
         ["Why held-out data and why a time window?", "On training data every model fits to the noise level, so true and false look the same. In t > 1.5 the prediction comes mostly from the learned equation, so a wrong ĝ shows."],
         ["Why δ = 0.1?", "From a pilot before the real runs: chance moves R by ≤ 2.4 %, true symmetries give R ≤ 1, a false one ≥ 11. Fixed before seeing loop results."],
         ["Why not raise the threshold instead of learnable coefficients?", "True-but-inexact and false-but-close overlapped (19 vs 13): no threshold separates them."],
         ["Why substitute u_t in the condition?", "The condition must hold on solutions; substitution makes true symmetries give exactly zero and removes the multiplier."],
         ["How reliable are the numbers?", "3 data draws per setting; the latest results are one draw, noise-free. The full evaluation (432 chains) is running."],
         ["What is new compared with LieNLSD?", "Sparse noisy data via PINN; candidates tested on held-out data; refined so they can be enforced; fed back to improve the learned equation."],
         ["Why both a polynomial and an MLP for g?", "Polynomial: readable coefficients. MLP: no assumed form, closer to a real application."]],
        [60, 110])

d.h1("4. Key numbers to remember")
d.table([["what", "number"],
         ["learned g, ν at N = 1000, 1 %", "g error ≈ 5 %, ν error ≈ 0.25 %"],
         ["gap of an exact equation vs learned", "≈ 14 vs 0.1 – 1.2"],
         ["true symmetries as a loss, g0", "err_g 0.165 → 0.0003"],
         ["δ pilot, 1 % noise", "true R 0.95 – 0.99, false R 11 – 12"],
         ["first loop: true vs false scaling (noise-free)", "R 4 – 19 vs 13 – 77 (overlap)"],
         ["with learnable coefficients", "true R 0.042, false R 2325"],
         ["g0 with sparsification", "all 3 symmetries exact; ν error 0.136 % → 0.002 %"]], [80, 90])

d.h1("5. Mini-glossary")
d.table([["term", "one line"],
         ["generator", "the infinitesimal direction V = ξ∂x + τ∂t + φ∂u of a symmetry transformation"],
         ["prolongation", "how the transformation also changes u_x, u_t, u_xx"],
         ["infinitesimal criterion", "pr V(Δ) = 0 on solutions: the condition for V to be a symmetry (Olver Thm 2.31)"],
         ["basis fields", "ξ, τ, φ each written with 10 monomials → 30 unknown coefficients"],
         ["SVD, singular vector", "small singular value ↔ near-symmetry; its vector = candidate generator"],
         ["res", "sin(angle) to a true symmetry: 0 exact, 0.1 ≈ 6°, 1 unrelated (only for scoring)"],
         ["R, δ", "held-out error with / without the candidate; accept if R ≤ 1 + δ = 1.1"],
         ["cleaning / sparsifying", "zero the small coefficients / rotate several candidates to the sparsest basis"],
         ["drift", "angle between the candidate before and after training; > 10° = rejected"]], [44, 126])
d.p("Bring: the handout (to give), the two-week report (for details), and the loop figure.", "small")
d.build("meeting_prep.pdf", "Meeting preparation (for the student)")
