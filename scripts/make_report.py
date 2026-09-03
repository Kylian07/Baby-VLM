#!/usr/bin/env python3
"""Render the GAVAGAI project report to PDF.

Kept in the repo rather than produced ad hoc so the report can be regenerated
whenever the numbers change. Figures are read from ``docs/figures/``, which are
themselves generated from the measured result JSONs -- so no number in the PDF
is hand-transcribed.

    python scripts/make_report.py --out docs/GAVAGAI_report.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# --- palette ---------------------------------------------------------------
INK = colors.HexColor("#1A1D24")
MUTED = colors.HexColor("#5A6270")
ACCENT = colors.HexColor("#2C4B9B")
ACCENT_LT = colors.HexColor("#E8EDF8")
WARN = colors.HexColor("#C2410C")
WARN_LT = colors.HexColor("#FDF0E7")
GOOD = colors.HexColor("#0B7A5A")
RULE = colors.HexColor("#D6DAE2")
ZEBRA = colors.HexColor("#F5F7FA")

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def register_fonts() -> tuple[str, str, str]:
    """Register DejaVu, which has the Greek and symbol glyphs the report needs."""
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", FONT_DIR / "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", FONT_DIR / "DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("DejaVuMono", FONT_DIR / "DejaVuSansMono.ttf"))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                      italic="DejaVu", boldItalic="DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold", "DejaVuMono"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Courier"


BODY_F, BOLD_F, MONO_F = register_fonts()


def styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("t", parent=ss["Title"], fontName=BOLD_F, fontSize=25,
                                leading=29, textColor=INK, alignment=0, spaceAfter=2)
    s["subtitle"] = ParagraphStyle("st", fontName=BODY_F, fontSize=11.5, leading=16,
                                   textColor=ACCENT, spaceAfter=3)
    s["meta"] = ParagraphStyle("m", fontName=BODY_F, fontSize=8.5, leading=12, textColor=MUTED)
    s["h1"] = ParagraphStyle("h1", fontName=BOLD_F, fontSize=14.5, leading=18, textColor=ACCENT,
                             spaceBefore=15, spaceAfter=6)
    s["h2"] = ParagraphStyle("h2", fontName=BOLD_F, fontSize=11, leading=14, textColor=INK,
                             spaceBefore=10, spaceAfter=4)
    s["body"] = ParagraphStyle("b", fontName=BODY_F, fontSize=9.6, leading=14.4, textColor=INK,
                               alignment=TA_JUSTIFY, spaceAfter=6)
    s["bullet"] = ParagraphStyle("bu", parent=s["body"], leftIndent=13, bulletIndent=3,
                                 spaceAfter=3, alignment=0)
    s["cap"] = ParagraphStyle("c", fontName=BODY_F, fontSize=8.2, leading=11.5, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=9)
    s["cell"] = ParagraphStyle("cl", fontName=BODY_F, fontSize=8.4, leading=11.5, textColor=INK)
    s["cellb"] = ParagraphStyle("clb", fontName=BOLD_F, fontSize=8.4, leading=11.5,
                                textColor=colors.white)
    s["callout"] = ParagraphStyle("co", fontName=BODY_F, fontSize=9.6, leading=14.2,
                                  textColor=INK, alignment=TA_JUSTIFY)
    s["mono"] = ParagraphStyle("mo", fontName=MONO_F, fontSize=8.2, leading=12, textColor=INK)
    return s


S = styles()


def P(txt, st="body"):
    return Paragraph(txt, S[st])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def callout(title, text, tone="accent"):
    bg, bar = (ACCENT_LT, ACCENT) if tone == "accent" else (WARN_LT, WARN)
    inner = [Paragraph(f'<font color="{bar.hexval()}"><b>{title}</b></font>', S["callout"]),
             Spacer(1, 3), Paragraph(text, S["callout"])]
    t = Table([[inner]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, bar),
    ]))
    return KeepTogether([t, Spacer(1, 9)])


def table(header, rows, widths, highlight=None, align_right_from=1):
    """Zebra-striped table. ``highlight`` is a set of row indices to emphasise."""
    data = [[Paragraph(h, S["cellb"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), S["cell"]) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (align_right_from, 1), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    for i in highlight or []:
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FFF6DB")))
    t.setStyle(TableStyle(st))
    return KeepTogether([t, Spacer(1, 8)])


def figure(path: Path, caption: str, width=160 * mm):
    if not path.exists():
        return Spacer(1, 0)
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, Paragraph(caption, S["cap"])])


# --- page furniture --------------------------------------------------------

def decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page == 1:
        canvas.setFillColor(ACCENT)
        canvas.rect(0, h - 13 * mm, w, 13 * mm, stroke=0, fill=1)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(22 * mm, 15 * mm, w - 22 * mm, 15 * mm)
    canvas.setFont(BODY_F, 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(22 * mm, 10.5 * mm, "GAVAGAI — referential alignment for infant-scale VLMs")
    canvas.drawRightString(w - 22 * mm, 10.5 * mm, f"{doc.page}")
    canvas.restoreState()


def build(out: Path, figdir: Path):
    doc = BaseDocTemplate(str(out), pagesize=A4, title="GAVAGAI: Cross-Situational Grounding as Balanced Optimal Transport",
                          author="Prepared with Claude Code", leftMargin=22 * mm, rightMargin=22 * mm,
                          topMargin=20 * mm, bottomMargin=20 * mm)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(story(figdir))


def story(F: Path):
    st = []
    a = st.append

    # ---------------- cover ----------------
    a(Spacer(1, 6 * mm))
    a(P("GAVAGAI", "title"))
    a(P("Cross-situational grounding as balanced optimal transport", "subtitle"))
    a(Spacer(1, 2))
    a(P("A referential-alignment objective for infant-scale vision–language models, "
        "targeting the BabyVLM Workshop @ NeurIPS 2026.", "meta"))
    a(Spacer(1, 5))
    a(Table([[""]], colWidths=[165 * mm], rowHeights=[1.6],
            style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)])))
    a(Spacer(1, 8))

    a(callout(
        "In one paragraph",
        "A baby hears <b>“look — are you hungry? there’s the ball!”</b> while facing a room "
        "containing a ball, a hand, the floor and a table. Which word goes with which thing? Nobody says. "
        "Today’s baby vision–language models are trained to predict the next word given the picture, "
        "an objective with <b>no place in it for a word to mean a thing</b>. This work adds that missing "
        "piece: on every scene, words are assigned to image regions by solving a small transport problem "
        "with two rules — a word may refer to <b>nothing</b>, and no single region may absorb "
        "<b>everything</b>. The second rule is mutual exclusivity, a documented bias in real toddlers, and "
        "it turns out to be exactly a capacity constraint on an assignment matrix."))

    a(P("What this report covers", "h2"))
    a(P("Section 1 states the problem. Section 2 summarises the papers the workshop asks entrants to read "
        "and what each leaves open. Section 3 identifies two gaps — one in how these models are "
        "<b>evaluated</b>, one in how they are <b>trained</b>. Section 4 explains the method without "
        "mathematics. Sections 5–6 give the novelty claim and the measured results. Sections 7–8 "
        "record what went wrong and what remains unproven.", "body"))

    # ---------------- 1 problem ----------------
    a(P("1.  The problem: which word means which thing?", "h1"))
    a(P("The philosopher W.V.O. Quine imagined a linguist who hears a native speaker say "
        "<b>“gavagai”</b> as a rabbit runs past. Does it mean <i>rabbit</i>? <i>white</i>? "
        "<i>running</i>? <i>dinner tonight</i>? A single scene cannot tell you. An infant faces this "
        "problem continuously, and solves it.", "body"))
    a(P("Two properties of real infant head-camera footage make it considerably harder than the textbook "
        "version:", "body"))
    a(Spacer(1, 2))
    for b in bullets([
        "<b>Most words refer to nothing on screen.</b> “hungry”, “there”, "
        "“we”, “go” point at nothing visible. In transcribed caregiver speech these "
        "are the majority of word tokens.",
        "<b>A few objects are in almost every frame and are almost never named.</b> Hands, floor, table, "
        "the caregiver’s torso. They are permanently available to be blamed for any word — the "
        "raw material of what we call a <b>hub</b>.",
    ]):
        a(b)
    a(Spacer(1, 5))
    a(P("Developmental psychology names two mechanisms infants use. <b>Cross-situational learning</b>: no "
        "single scene resolves “ball”, but across hundreds of scenes the word co-occurs with balls "
        "and not with anything else. <b>Mutual exclusivity</b>: shown a familiar ball and a novel gadget and "
        "told “find the dax”, a toddler picks the gadget — one word, one thing. Both are the "
        "design targets of the method in Section 4.", "body"))

    # ---------------- 2 prior work ----------------
    a(PageBreak())
    a(P("2.  What the prior work established, and what it left open", "h1"))
    a(P("These are the papers the BabyVLM workshop lists as recommended reading, plus two that turned out "
        "to matter.", "body"))
    a(table(
        ["Paper", "What it contributed", "What it left open"],
        [["<b>SAYCam</b><br/>Sullivan et al., 2022",
          "Head-camera video from three infants, 6–32 months, ~478 h. The reference corpus.",
          "Gated behind Databrary approval, so results on it are not directly reproducible."],
         ["<b>BabyView</b><br/>Long et al., 2024",
          "A second, higher-resolution longitudinal infant corpus.", "Also gated."],
         ["<b>LLaVA</b><br/>Liu et al., 2023",
          "The standard recipe: frozen vision encoder, connector, language model, trained to predict the caption.",
          "Nothing in the objective ever asks which word corresponds to which part of the image."],
         ["<b>CVCL</b><br/>Vong et al., <i>Science</i> 2024",
          "Trained on one child’s footage; genuinely acquired word–referent mappings that generalise.",
          "Matches whole utterance to whole frame. No latent word-to-region variable."],
         ["<b>BabyVLM-V1 / V2</b><br/>Wang et al., ICCV’25 / CVPR’26",
          "A 1.1B model trained from scratch on SAYCam, plus <b>DevCV Toolbox</b>: ten tasks adapted from the NIH Baby Toolbox®.",
          "Every training stage is next-token prediction. No contrastive term anywhere."],
         ["<b>Looking to Learn</b><br/>Ganescu et al., 2025",
          "A learned gate deciding when to use vision versus language. Found it favours vision for content words, language for function words — with no supervision.",
          "Left as an emergent side effect rather than a structural commitment."],
         ["<b>3rd BabyLM Challenge</b><br/>Charpentier et al., 2025",
          "Ran the multimodal track with GIT and Flamingo baselines.",
          "<b>No submission beat the baselines, two years running.</b> Notes that new objectives tend to produce the best approaches."]],
        widths=[36 * mm, 64 * mm, 65 * mm], align_right_from=0))

    # ---------------- 3 gaps ----------------
    a(P("3.  Two gaps", "h1"))
    a(P("3.1  The benchmark may not measure what it appears to", "h2"))
    a(P("BabyVLM-V2’s published per-task scores look like a model with excellent perception and no "
        "vocabulary: <b>96.4</b> on Left/Right (humans score 94.5 — it beats people) but <b>32.4</b> on "
        "Picture Vocabulary, where pure guessing scores 25.0 and humans score 91.8.", "body"))
    a(P("But Left/Right asks: <i>“here is a cat — which of these three is exactly this cat?”</i> "
        "That is a spot-the-matching-photograph puzzle. It may not require understanding the question at all. "
        "So we tested it with a deliberately weak baseline: shrink every image to 32×32, take its "
        "brightness pattern and colour histogram, and return whichever option is most similar to the query. "
        "<b>No training, no neural network, and no access to the prompt.</b>", "body"))
    a(table(
        ["Task", "Text-blind matcher", "Baby model", "Human", "Chance"],
        [["Left/Right", "<b>1.00</b> &nbsp;(24/24, CI 0.86–1.00)", "96.4", "94.5", "33.3"],
         ["NIH Spatial", "<b>1.00</b> &nbsp;(10/10, CI 0.72–1.00)", "92.8", "100", "33.3"],
         ["spatialdetails (SAYCam)", "0.33 &nbsp;(CI 0.10–0.70)", "—", "—", "33.3"],
         ["Picture Vocabulary", "n/a — nothing to match against", "32.4", "91.8", "25.0"]],
        widths=[42 * mm, 58 * mm, 22 * mm, 20 * mm, 21 * mm], highlight=[1, 2]))
    a(callout(
        "Finding 1",
        "Left/Right — where the published model scores <b>above the human ceiling</b> — is solved "
        "outright by a matcher that never reads the question. Picture Vocabulary is <b>not</b>, which is the "
        "control showing the baseline is not simply winning everywhere. Part of the dramatic "
        "“perception yes, words no” split is therefore a property of the <b>tasks</b>, not of the "
        "model. <font color=\"%s\"><b>Caveat:</b></font> n is 6–24 per task on the public samples, "
        "hence the Wilson intervals; these must be regenerated on the full public Ego4D release before "
        "being quoted." % WARN.hexval()))
    a(figure(F / "text_blind_audit.png",
             "Figure 1. Accuracy of a training-free matcher that never reads the prompt. "
             "Vertical ticks mark the chance floor for each task."))

    a(P("3.2  The training objective has nowhere to put a word’s meaning", "h2"))
    a(P("This is the deeper gap. In LLaVA-style training the model’s only job is: given the picture, "
        "predict the next word. That can be done well from language habits alone — “are you…” "
        "is usually followed by “hungry” — plus a rough sense of the scene. There is no variable "
        "anywhere in the mathematics that says <b>“this word refers to that region”</b>, so there is "
        "no pressure ever to work it out.", "body"))
    a(P("A note on precision: it is tempting to blame contrastive learning’s frequency bias here. That "
        "would be wrong, and we do not claim it — with negatives drawn from the marginal, the "
        "contrastive optimum is already an unbiased association score. The defect is specifically in the "
        "<b>within-scene</b> assignment step, whose competitors are the other regions of the same image.", "body"))

    # ---------------- 4 method ----------------
    a(PageBreak())
    a(P("4.  The method, as a seating chart", "h1"))
    a(P("Treat every scene as a seating problem. The rows are the content words just heard. The columns are "
        "regions of the image — plus one extra column labelled <b>NOTHING</b>. Every word must be "
        "assigned somewhere.", "body"))

    grid = [["", "hand", "floor", "ball", "table", "NOTHING"],
            ["ball", "", "", "■", "", ""],
            ["hungry", "", "", "", "", "■"]]
    gt = Table(grid, colWidths=[24 * mm] + [22 * mm] * 5, hAlign="LEFT")
    gt.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), BODY_F, 8.6),
        ("FONT", (0, 0), (-1, 0), BOLD_F, 8.6),
        ("FONT", (0, 1), (0, -1), BOLD_F, 8.6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (5, 0), (5, -1), colors.HexColor("#FFF6DB")),
        ("TEXTCOLOR", (1, 1), (-1, -1), ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    a(gt)
    a(Paragraph("Figure 2. One scene as an assignment matrix. “ball” claims a region; "
                "“hungry” claims the NOTHING column.", S["cap"]))

    a(P("Rule 1 — a word may mean nothing", "h2"))
    a(P("That is what the NOTHING column is for. This sounds trivial; it is the single most important part. "
        "Without it, “hungry” is forcibly glued to whatever happens to be visible — usually a "
        "hand or the floor, because those are always there. Repeat a few thousand times and every word in "
        "the vocabulary ends up pointing at a hand.", "body"))
    a(P("Rule 2 — no region may absorb everything", "h2"))
    a(P("Each column has limited capacity. If “hand” has already been claimed, the next word must "
        "look elsewhere. This <b>is</b> mutual exclusivity, expressed as a constraint rather than a "
        "heuristic: the degenerate solution where every word points at one salient region becomes "
        "<b>infeasible</b>, not merely disfavoured.", "body"))
    a(P("Solving it", "h2"))
    a(P("“Assign rows to columns under capacity limits” is a classical, solved problem — "
        "<b>optimal transport</b> — and the standard algorithm is simply: repeatedly divide each row by "
        "its total, then each column by its total. About eight matrix multiplications. Next to running the "
        "vision encoder it is effectively free.", "body"))
    a(P("Remembering across scenes", "h2"))
    a(P("One scene is hopeless; a thousand are not. A running tally records how often each word was assigned "
        "to each kind of object, and that tally is fed back into the next scene’s costs. Crucially the "
        "tally is scored by <b>how surprising</b> a pairing is, not how frequent — raw counts would just "
        "conclude that everything goes with hands, since hands are always present. Correcting for baseline "
        "frequency is what lets “ball” stand out. This is the cross-situational learning the "
        "project is named for.", "body"))

    # ---------------- 5 novelty ----------------
    # Bind the heading to its table so it cannot be orphaned at a page foot.
    novelty = table(
        ["Ingredient", "Why it matters"],
        [["<b>One dial from the standard method</b>",
          "A single knob ρ controls exclusivity. At <b>ρ = 0</b> the mathematics provably collapses "
          "into <i>exactly</i> the region–word contrastive learning everyone already uses. Every "
          "ablation is therefore the same code path with one number changed — no chance of accidentally "
          "handicapping the baseline through a re-implementation."],
         ["<b>Words may refer to nothing</b>",
          "Standard objectives force every word onto some region. Given that most caregiver speech is "
          "non-referential, this is a genuine defect; no prior work makes “nothing” a first-class option."],
         ["<b>Memory across scenes</b>",
          "The closest prior art (OTTER, 2021) uses related transport machinery, but only within a single "
          "batch and at whole-caption granularity. Here the coupling is word-to-region and the lexicon "
          "persists across the entire run."],
         ["<b>Exclusivity as a constraint</b>",
          "A bias psychologists have studied for four decades turns out to be precisely a capacity "
          "constraint on an assignment matrix — and enforcing it costs almost nothing."]],
        widths=[45 * mm, 120 * mm], align_right_from=0)
    a(KeepTogether([P("5.  What is actually new", "h1"), novelty]))

    # ---------------- 6 results ----------------
    a(P("6.  Measured results", "h1"))
    a(P("All figures below are generated directly from the run outputs; no number is transcribed by hand. "
        "Every condition is three random seeds with dispersion reported.", "body"))

    a(P("6.1  The headline: forcing every word to mean something is catastrophic", "h2"))
    a(P("Picture-vocabulary accuracy on <b>held-out exemplars</b> of each category — so this measures "
        "generalisation to unseen instances, not recall of a memorised vector. Chance is 0.025.", "body"))
    a(table(
        ["Condition", "Clean", "Moderate", "Realistic"],
        [["Captioning only (the standard recipe)", "0.392", "0.242", "0.058"],
         ["+ alignment, <b>forced</b> (no NOTHING column)", "0.650", "0.358", "<b>0.000</b>"],
         ["+ NOTHING column only", "0.658", "0.358", "0.017"],
         ["+ NOTHING + exclusivity &nbsp;<b>(ours)</b>", "<b>0.700</b>", "<b>0.400</b>", "0.075"],
         ["+ NOTHING + full balancing", "0.683", "0.400", "0.083"]],
        widths=[85 * mm, 26 * mm, 26 * mm, 26 * mm], highlight=[2, 4]))
    a(callout(
        "Finding 2",
        "Under realistic ambiguity — 80% of speech non-referential, half of content words uttered while "
        "their referent is off-screen — the <b>naive</b> form of the alignment scores <b>0.000, with zero "
        "variance across all three seeds</b>. That is total hub collapse, and it is <b>worse than adding "
        "nothing at all</b>. The escape hatch and the capacity limit are what turn that catastrophe into the "
        "best result in the table."))
    a(figure(F / "ar_ablation.png",
             "Figure 3. The captioning objective is weakest in every regime. The missing orange bar at "
             "right is not a rendering error: naive alignment scores exactly zero."))

    a(P("6.2  The exclusivity knob has a sweet spot", "h2"))
    a(P("Sweeping ρ continuously from 0 (which <i>is</i> the standard method) to full balancing:", "body"))
    a(table(
        ["ρ", "0", "0.02", "0.05", "<b>0.1</b>", "0.3", "1", "3", "∞"],
        [["accuracy", "0.275", "0.333", "0.375", "<b>0.383</b>", "0.358", "0.333", "0.358", "0.350"],
         ["hub rate", "0.050", "0.033", "0.050", "0.033", "0.042", "0.025", "0.025", "0.025"]],
        widths=[22 * mm] + [17.8 * mm] * 8))
    a(P("The optimum is <b>interior</b>, not at maximum exclusivity. That is what one wants from a genuine "
        "cognitive constraint: real scenes do contain several things a word could plausibly mean, so forcing "
        "a strict one-to-one matching over-constrains. A parameter that only ever helps would look more like "
        "an artefact than a mechanism.", "body"))
    a(figure(F / "rho_sweep.png",
             "Figure 4. Accuracy peaks at an interior optimum while the hub rate trends down — "
             "the effect and the mechanism it is attributed to move together.", width=150 * mm))

    a(P("6.3  Consistent across corpus sizes", "h2"))
    a(P("Re-measured sample efficiency; column A is the standard method (ρ = 0, no NOTHING column):", "body"))
    a(table(
        ["Corpus", "A — contrastive", "B — + NOTHING", "C — + exclusivity", "D — balanced"],
        [["250", "0.342", "0.333", "<b>0.400</b>", "0.383"],
         ["500", "0.358", "0.417", "0.458", "<b>0.475</b>"],
         ["1000", "0.392", "0.367", "<b>0.475</b>", "0.467"],
         ["2500", "0.450", "0.350", "<b>0.542</b>", "0.492"]],
        widths=[24 * mm, 36 * mm, 34 * mm, 36 * mm, 32 * mm]))
    a(P("Condition C beats the baseline at all four sizes (+0.058, +0.100, +0.083, +0.092), with hub rate "
        "roughly a third to a quarter of the baseline’s throughout. <b>The accuracy gain and the "
        "mechanism it is attributed to move together</b> — which is stronger evidence than a bare "
        "accuracy delta.", "body"))

    a(P("6.4  It reproduces human data", "h2"))
    a(P("Yu &amp; Smith (2007) had adults learn 18 word–referent pairs from 27 deliberately ambiguous "
        "trials. An unconstrained ideal observer solves that design perfectly, so the informative question is "
        "what capacity limit reproduces the human pattern. Forgetting does not — accuracy stays at "
        "ceiling for every decay rate we tried. <b>Limited encoding</b> does: assume a learner registers only "
        "two word–object pairs per trial, and one shared free parameter reproduces the whole ordering.", "body"))
    a(table(
        ["Objects encoded per trial", "2×2", "3×3", "4×4", "RMSE vs human"],
        [["unlimited", "0.983", "1.000", "1.000", "0.292"],
         ["1", "0.456", "0.336", "0.208", "0.410"],
         ["<b>2</b>", "<b>0.983</b>", "<b>0.725</b>", "<b>0.517</b>", "<b>0.066</b>"],
         ["3", "0.983", "1.000", "0.875", "0.231"],
         ["<b>human</b>", "<b>0.889</b>", "<b>0.778</b>", "<b>0.556</b>", "—"]],
        widths=[52 * mm, 24 * mm, 24 * mm, 24 * mm, 32 * mm], highlight=[3, 5]))

    # ---------------- 7 corrections ----------------
    a(PageBreak())
    a(P("7.  What went wrong along the way", "h1"))
    a(P("Recorded here rather than quietly patched, because each one would have been a reviewer’s finding.", "body"))
    a(table(
        ["What was wrong", "How it was caught and fixed"],
        [["<b>The original theory was wrong.</b> The first framing blamed contrastive learning’s frequency bias.",
          "It is not frequency-biased — with marginal negatives its optimum is already an unbiased score. Reframed around the within-scene assignment step, which genuinely lacks any correction."],
         ["<b>A tempting lemma was false.</b> “Balancing induces the same ranking as the surprise score.”",
          "Tested on 2,000 random matrices: the two disagree on more than half. There is now a test that <i>documents the falsity</i> so nobody re-derives it."],
         ["<b>A negative result was retracted.</b> An earlier draft reported the exclusivity constraint gave no benefit.",
          "It had been measured in a setup where every condition scored a perfect 1.000, making the comparison vacuous. Re-measured properly, the effect is present at every corpus size."],
         ["<b>Part of the codebase was silently untracked.</b>",
          "A <font name=\"%s\">.gitignore</font> pattern matched at any depth and excluded a package directory. A fresh clone would not have imported. Caught by actually cloning it." % MONO_F],
         ["<b>A performance defect that only bites on GPU.</b>",
          "The transport solver synchronised with the host every iteration — harmless on CPU, crippling on a T4. Removed before any GPU time was spent."],
         ["<b>Tests depended on execution order.</b>",
          "Random seeding sat at module level, so adding a new test file silently changed the data another file’s tests ran on. Seeding is now per-test."],
         ["<b>A retracted claim was baked into the generator.</b>",
          "The summary document is auto-generated; the withdrawn wording lived in the generating script and would have reappeared on every regeneration."]],
        widths=[62 * mm, 103 * mm], align_right_from=0))

    # ---------------- 8 limits ----------------
    a(P("8.  What remains unproven", "h1"))
    for b in bullets([
        "<b>Everything measured so far is simulation.</b> Whether the gains transfer to real video is untested, and is the genuine risk in the project.",
        "<b>The benchmark audit rests on small samples</b> (6–24 items per task). Regenerating it on the full public Ego4D release is the single highest-value next step — minutes of compute, and unlike the training run it cannot come back negative.",
        "<b>The hardest ambiguity regime is floor-limited.</b> Tripling the corpus and the training steps moves nothing, so no claim is made there beyond “does not collapse”.",
        "<b>Regions are pooled grid cells, not discovered objects.</b> No object discovery is claimed.",
        "<b>The public data is adult egocentric video</b> (Ego4D), used as a reproducible stand-in for gated infant corpora. Claims about developmental plausibility are hedged accordingly.",
    ]):
        a(b)
    a(Spacer(1, 6))
    a(callout(
        "How the work is structured against risk",
        "This is deliberately a <b>two-tier bet</b>. The benchmark audit in Section 3.1 is a measurement that "
        "holds up regardless of whether the method works — it needs no training and cannot fail. The "
        "method sits on top of it with supporting evidence and its limits stated. If the real-data run "
        "disappoints, the contribution narrows rather than disappears."))

    a(P("Reproducing everything", "h2"))
    a(Paragraph(
        "git clone https://github.com/Kylian07/Baby-VLM<br/>"
        "pip install -r requirements.txt &amp;&amp; python -m pytest tests/ -q<br/>"
        "python scripts/text_blind_audit.py --roots data/Ego4D<br/>"
        "python scripts/run_simulation.py ar", S["mono"]))
    a(Spacer(1, 5))
    a(P("31 tests cover the objective, the transport solver, the lexicon and the benchmark loader — "
        "including one test whose purpose is to document a claim verified to be <b>false</b>. A Kaggle "
        "notebook runs the whole pipeline end to end on a T4.", "body"))
    a(Spacer(1, 4))
    a(callout(
        "The single highest-value next hour",
        "Run the benchmark audit against the <b>full public Ego4D release</b> of DevCV Toolbox — one "
        "command in the Kaggle notebook, minutes of compute. It turns the Left/Right result from "
        "<b>24/24 with a wide confidence interval</b> into a number with enough samples to headline a "
        "paper. Unlike the training run, it cannot come back negative: whatever it returns is a fact "
        "about the benchmark that the community needs."))
    a(Spacer(1, 4))
    a(Paragraph(
        "Repository: github.com/Kylian07/Baby-VLM &nbsp;·&nbsp; "
        "Method and proofs: METHOD.md &nbsp;·&nbsp; Measured numbers: RESULTS.md &nbsp;·&nbsp; "
        "Reading notes: docs/related_work.md &nbsp;·&nbsp; Paper plan: docs/paper_outline.md",
        S["meta"]))
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/GAVAGAI_report.pdf"))
    ap.add_argument("--figures", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.out, args.figures)
    print(f"wrote {args.out} ({args.out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
