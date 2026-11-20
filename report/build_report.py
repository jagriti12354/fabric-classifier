"""
Builds the TXD401 Major Project Part I report (.docx) in the department format.
  pip install python-docx pillow
  python report/build_report.py --micro-images <folder with Kaggle class folders>
Microscope CNN numbers are read from research/microscope/outputs_*/cv_results.json.
Open the .docx in Word and press "Yes" to update fields (Contents / List of Figures / Tables).
"""
import argparse
import json
import os

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)

# ----------------------------------------------------------------- fill these
NAME = "[Name of the Candidate]"
ENTRY = "[Entry No.]"
SUPERVISOR = "[Name of the Supervisor]"
MONTH_YEAR = "November 2026"
TITLE = "Low-Cost Identification of Silk, Cotton and Polyester Fabrics using Image-Based Deep Learning and Infrared Spectroscopy"


# ============================================================ docx helpers
def set_font(run, size=12, bold=None, italic=None):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def setup_styles(doc):
    for name in ["Normal", "Heading 1", "Heading 2", "Heading 3", "Caption"]:
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        st.font.color.rgb = None
        st.font.size = Pt(12)
        pf = st.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(6)
    doc.styles["Normal"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    h1 = doc.styles["Heading 1"]
    h1.font.size, h1.font.bold = Pt(16), True
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.space_before, h1.paragraph_format.space_after = Pt(0), Pt(18)
    h2 = doc.styles["Heading 2"]
    h2.font.bold, h2.font.all_caps = True, True
    h2.paragraph_format.space_before = Pt(12)
    h3 = doc.styles["Heading 3"]
    h3.font.bold, h3.font.italic = True, False
    for name in ["Heading 1", "Heading 2", "Heading 3"]:
        doc.styles[name].paragraph_format.keep_with_next = True
        doc.styles[name].font.color.rgb = None
    for cname in ["FigCaption", "TabCaption"]:
        st = doc.styles.add_style(cname, 1)
        st.base_style = doc.styles["Normal"]
        st.font.size = Pt(12)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        st.paragraph_format.space_after = Pt(10)
    doc.styles["TabCaption"].paragraph_format.keep_with_next = True


def add_field(par, instr, placeholder="Right-click and choose Update Field"):
    def fld(t):
        r = OxmlElement("w:r")
        f = OxmlElement("w:fldChar")
        f.set(qn("w:fldCharType"), t)
        r.append(f)
        return r
    par._p.append(fld("begin"))
    r = OxmlElement("w:r")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    r.append(it)
    par._p.append(r)
    par._p.append(fld("separate"))
    run = par.add_run(placeholder)
    set_font(run)
    par._p.append(fld("end"))


def page_number_footer(section, fmt, start=1):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(p, "PAGE", "1")
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))
    section._sectPr.append(pg)


def margins(section):
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)   # A4
    section.top_margin = section.bottom_margin = Inches(1)
    section.left_margin, section.right_margin = Inches(1.5), Inches(1)       # inside / outside


class Writer:
    def __init__(self, doc):
        self.d = doc
        self.chapter = 0
        self.fig = 0
        self.tab = 0
        self.h2 = 0
        self.h3 = 0

    def p(self, text, bold=False, align=None, size=12, italic=False, after=None):
        par = self.d.add_paragraph()
        # simple **bold** markup
        parts = text.split("**")
        for i, part in enumerate(parts):
            if part:
                set_font(par.add_run(part), size, bold or i % 2 == 1, italic)
        if align is not None:
            par.alignment = align
        if after is not None:
            par.paragraph_format.space_after = Pt(after)
        return par

    def bullets(self, items):
        for it in items:
            par = self.p("•  " + it)
            par.paragraph_format.left_indent = Inches(0.3)
            par.paragraph_format.first_line_indent = Inches(-0.2)

    def page_break(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def front_title(self, text):
        par = self.p(text, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, after=18)
        return par

    def chapter_start(self, title, first=False):
        self.chapter += 1
        self.fig = self.tab = self.h2 = 0
        if not first:
            self.page_break()
        self.p(f"Chapter-{self.chapter}", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=16, after=0)
        h = self.d.add_heading(level=1)
        set_font(h.add_run(f"{self.chapter}. {title}" if title not in ("References", "Appendix") else title), 16, True)

    def unnumbered_chapter(self, title):
        self.page_break()
        if title == "Appendix":
            self.chapter, self.fig, self.tab = "A", 0, 0
        h = self.d.add_heading(level=1)
        set_font(h.add_run(title), 16, True)

    def h(self, title):
        self.h2 += 1
        self.h3 = 0
        hd = self.d.add_heading(level=2)
        set_font(hd.add_run(f"{self.chapter}.{self.h2} {title.upper()}"), 12, True)

    def hh(self, title):
        self.h3 += 1
        hd = self.d.add_heading(level=3)
        set_font(hd.add_run(f"{self.chapter}.{self.h2}.{self.h3} {title}"), 12, True)

    def figure(self, path, caption, width=5.6):
        if not os.path.exists(path):
            self.p(f"[figure missing: {os.path.basename(path)}]", italic=True)
            return None
        self.fig += 1
        par = self.d.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True
        par.add_run().add_picture(path, width=Inches(width))
        c = self.d.add_paragraph(style="FigCaption")
        set_font(c.add_run(f"Figure {self.chapter}.{self.fig}: "), 12, True)
        set_font(c.add_run(caption), 12)
        return f"{self.chapter}.{self.fig}"

    def table(self, caption, header, rows, widths=None):
        self.tab += 1
        c = self.d.add_paragraph(style="TabCaption")
        set_font(c.add_run(f"Table {self.chapter}.{self.tab}: "), 12, True)
        set_font(c.add_run(caption), 12)
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htxt in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(cell.paragraphs[0].add_run(htxt), 11, True)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:fill"), "D9E2F3")
            cell._tc.get_or_add_tcPr().append(shd)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                par = cells[i].paragraphs[0]
                par.alignment = WD_ALIGN_PARAGRAPH.LEFT
                par.paragraph_format.line_spacing = 1.15
                set_font(par.add_run(str(v)), 11)
        for r_ in t.rows:
            for c_ in r_.cells:
                c_.paragraphs[0].paragraph_format.line_spacing = 1.15
                c_.paragraphs[0].paragraph_format.space_after = Pt(2)
        if widths:
            for r_ in t.rows:
                for i, w in enumerate(widths):
                    r_.cells[i].width = Inches(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(4)
        return f"{self.chapter}.{self.tab}"


# ============================================================ data helpers
def micro_results():
    out = {}
    for key in ["binary", "3class", "binary_extra"]:
        f = R("research", "microscope", f"outputs_{key}", "cv_results.json")
        if os.path.exists(f):
            d = json.load(open(f))
            cm = np.array(d["confusion_matrix"])
            rec = cm.diagonal() / cm.sum(1)
            prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
            out[key] = {"acc": np.mean(d["fold_acc"]), "acc_sd": np.std(d["fold_acc"]),
                        "f1": np.mean(d["fold_f1"]), "f1_sd": np.std(d["fold_f1"]),
                        "classes": d["classes"], "cm": cm, "rec": rec, "prec": prec,
                        "n": int(cm.sum())}
    return out


def montage(paths, labels, out, tile=260):
    imgs = []
    for p in paths:
        if os.name == "nt" and len(os.path.abspath(p)) > 240:   # Windows long-path limit
            p = "\\\\?\\" + os.path.abspath(p)
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((tile, tile))
        imgs.append(im)
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    from PIL import ImageDraw
    canvas = Image.new("RGB", (cols * (tile + 10) + 10, rows * (tile + 40) + 10), "white")
    dr = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        x = 10 + (i % cols) * (tile + 10)
        y = 10 + (i // cols) * (tile + 40)
        canvas.paste(im, (x, y))
        dr.text((x + 5, y + tile + 8), lab, fill="black")
    canvas.save(out)
    return out


def first_image(folder):
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(folder, f)


# ============================================================ content
def build(args):
    doc = Document()
    setup_styles(doc)
    s1 = doc.sections[0]
    margins(s1)
    s1.different_first_page_header_footer = True
    page_number_footer(s1, "lowerRoman", 1)
    settings = doc.settings.element
    settings.append(OxmlElement("w:mirrorMargins"))
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "true")
    settings.append(uf)

    W = Writer(doc)
    M = micro_results()
    C = WD_ALIGN_PARAGRAPH.CENTER

    # ---------------- title page
    W.p(TITLE, bold=True, align=C, size=20, after=24)
    W.p("TXD 401: Major Project Part I", bold=True, align=C, size=14, after=30)
    W.p(NAME, bold=True, align=C, size=14, after=0)
    W.p(ENTRY, bold=True, align=C, size=14, after=40)
    W.p("Under the Guidance of", align=C, size=14, after=0)
    W.p(SUPERVISOR, bold=True, align=C, size=14, after=30)
    logo = args.logo
    if logo and os.path.exists(logo):
        par = doc.add_paragraph()
        par.alignment = C
        par.add_run().add_picture(logo, width=Inches(1.3))
    W.p("Department of Textile and Fibre Engineering", align=C, size=14, after=0)
    W.p("Indian Institute of Technology Delhi", align=C, size=14, after=0)
    W.p(MONTH_YEAR, align=C, size=14)

    # ---------------- certificate
    W.page_break()
    W.front_title("CERTIFICATE")
    W.p(f"This is to certify that this project report entitled “{TITLE}”, being submitted by "
        f"{NAME}, Entry No. {ENTRY}, to the Department of Textile and Fibre Engineering, Indian "
        "Institute of Technology Delhi, for the award of the degree of Bachelor of Technology, is a "
        "record of bona fide work carried out under my guidance and supervision.")
    W.p("To the best of my knowledge, the results embodied in this project have not been submitted "
        "in part or full to other University or Institute.")
    for line in ["", "", "Signature(s) of Supervisor(s)", f"({SUPERVISOR})",
                 "Department of Textile and Fibre Engineering", "Indian Institute of Technology Delhi"]:
        W.p(line, align=WD_ALIGN_PARAGRAPH.RIGHT, after=0)
    W.p("Date:", after=0)
    W.p("Place: New Delhi")

    # ---------------- acknowledgement
    W.page_break()
    W.front_title("ACKNOWLEDGEMENT")
    W.p(f"I would like to express my sincere gratitude to my supervisor, {SUPERVISOR}, for the "
        "guidance, encouragement and valuable suggestions provided throughout this project.")
    W.p("I thank the Department of Textile and Fibre Engineering, IIT Delhi, for the facilities and "
        "support extended to me. I also acknowledge the open-data efforts of the National Institute of "
        "Standards and Technology (NIR-SORT), King's College London and collaborators (ATR-FTIR textile "
        "fibre dataset), and the authors of the open microscope fabric image dataset, without which the "
        "spectroscopic and microscopic parts of this work would not have been possible.")
    W.p("Finally, I thank my family and friends for their constant support.")
    for line in ["", "Signature(s) of Candidate(s)", f"({NAME})",
                 "Department of Textile and Fibre Engineering", "Indian Institute of Technology Delhi"]:
        W.p(line, align=WD_ALIGN_PARAGRAPH.RIGHT, after=0)
    W.p("Date:", after=0)
    W.p("Place: New Delhi")

    # ---------------- abstract
    mb = M.get("binary")
    micro_line = (f"a transfer-learned ResNet18 separated silk from cotton and polyester with "
                  f"{mb['acc']*100:.1f} ± {mb['acc_sd']*100:.1f} % cross-validated accuracy"
                  if mb else "a transfer-learned ResNet18 was trained for silk vs non-silk classification")
    W.page_break()
    W.front_title("ABSTRACT")
    W.p("Silk is a high-value fibre that is frequently imitated by cheaper fibres such as polyester, "
        "and fast fibre identification is equally important for quality control and for sorting textile "
        "waste for recycling. Conventional identification relies on burning, solubility or microscopy "
        "tests that are destructive or need trained personnel, while spectroscopic methods need "
        "instruments that are not always available. This project investigates low-cost and data-driven "
        "routes to identify three fibres – cotton, polyester and silk – with particular emphasis on "
        "distinguishing silk from non-silk fabrics.")
    W.p("Four complementary approaches were studied. (i) An on-device mobile application was developed "
        "in which a MobileNetV2 network, fine-tuned on about 2,500 phone photographs, classifies a "
        "fabric photograph as cotton, polyester or silk; it reached 70.4 % accuracy on a held-out test "
        f"set. (ii) Using an open dataset of smartphone-microscope images (60×), {micro_line}. "
        "(iii) Since no NIR spectrometer was available, the open NIST NIR-SORT dataset (833–2500 nm) was "
        "used: after SNV and Savitzky–Golay derivative pre-processing, PLS-DA and SVM classified pure "
        "cotton and polyester fabrics with 100 % accuracy under fabric-wise cross-validation, and PLS "
        "regression predicted the cotton content of cotton/polyester blends with an RMSECV of 4.9 % "
        "(R² = 0.98). (iv) As a proof of concept for silk, an open ATR-FTIR dataset of 137 fibre samples "
        "was used; silk was separated from all 30 other fibre types, including the protein fibres wool "
        "and alpaca and the polyamide nylon, for 136 of 137 samples, and a one-class model trained only "
        "on silk rejected 98 % of non-silk spectra.")
    W.p("The spectroscopic models are explained by fibre chemistry – the aromatic C–H and ester C=O "
        "bands of polyester, the O–H/C–O bands of cellulose, and the amide bands of silk fibroin, whose "
        "amide I band at about 1620 cm⁻¹ (β-sheet) distinguishes it from wool and nylon. Image-based "
        "methods are cheaper and portable but depend on surface appearance and are therefore less "
        "reliable. The main limitation is the scarcity of silk data in all open datasets; future work "
        "will measure in-house samples on an infrared spectrometer for external validation.")

    # ---------------- contents / lists
    for title, instr in [("CONTENTS", 'TOC \\o "1-3" \\h \\z \\u'),
                         ("LIST OF FIGURES", 'TOC \\h \\z \\t "FigCaption,1"'),
                         ("LIST OF TABLES", 'TOC \\h \\z \\t "TabCaption,1"')]:
        W.page_break()
        W.front_title(title)
        add_field(doc.add_paragraph(), instr,
                  "Open in MS Word and press F9 (or right-click → Update Field) to generate this list.")

    W.page_break()
    W.front_title("LIST OF SYMBOLS AND ABBREVIATIONS")
    abbr = [("ATR-FTIR", "Attenuated total reflectance Fourier-transform infrared spectroscopy"),
            ("CNN", "Convolutional neural network"), ("CV", "Cross-validation"),
            ("LV", "Latent variable (PLS component)"), ("NIR", "Near infrared (780–2500 nm)"),
            ("PCA", "Principal component analysis"), ("PET", "Poly(ethylene terephthalate), polyester"),
            ("PLS / PLS-DA", "Partial least squares (discriminant analysis)"),
            ("R²", "Coefficient of determination"),
            ("RMSECV", "Root mean square error of cross-validation"),
            ("SG", "Savitzky–Golay smoothing/derivative filter"),
            ("SIMCA", "Soft independent modelling of class analogy"),
            ("SNV", "Standard normal variate"), ("SVM", "Support vector machine"),
            ("TFLite", "TensorFlow Lite (on-device model format)"),
            ("Q", "Squared residual of a spectrum from a PCA model")]
    t = doc.add_table(rows=0, cols=2)
    for a, b in abbr:
        cells = t.add_row().cells
        set_font(cells[0].paragraphs[0].add_run(a), 12, True)
        set_font(cells[1].paragraphs[0].add_run(b), 12)

    # ---------------- main matter: new section, arabic numbering, odd page
    s2 = doc.add_section(WD_SECTION.ODD_PAGE)
    margins(s2)
    s2.different_first_page_header_footer = False
    page_number_footer(s2, "decimal", 1)

    # ===== Chapter 1
    W.chapter_start("Introduction", first=True)
    W.h("Background")
    W.p("Identification of the fibre content of a fabric is a routine requirement in textile testing, "
        "labelling, trade and recycling. Standard laboratory practice combines microscopy, burning "
        "behaviour and solubility tests [1]. These tests are reliable in trained hands but are slow, "
        "partly destructive and difficult to apply in the field. Fibre identification has also become "
        "important for the circular economy: post-consumer textiles must be sorted by fibre before they "
        "can be recycled, and near-infrared (NIR) spectroscopy has emerged as the main technology for "
        "automated sorting [2].")
    W.p("Silk deserves special attention in the Indian context. It is a high-value protein fibre "
        "(mulberry, tussar, eri and muga silks), and silk products are often imitated with polyester "
        "satin or regenerated cellulose (‘art silk’). A quick, low-cost method to tell silk from "
        "non-silk would therefore be useful for consumers, traders and quality-control laboratories.")
    W.h("Spectroscopic and image-based identification")
    W.p("Vibrational spectroscopy identifies a fibre from its chemistry. In the NIR region (780–2500 nm) "
        "the overtone and combination bands of C–H, O–H and N–H bonds give each polymer a characteristic "
        "spectrum [3]. In the mid-infrared region, attenuated total reflectance FTIR (ATR-FTIR) measures "
        "the fundamental vibrations directly and is a standard method for fibre identification [4]. "
        "Image-based methods, on the other hand, use cameras or low-cost microscopes together with "
        "deep learning to recognise fabrics from their appearance – weave, lustre, yarn texture and "
        "colour. They are cheap and portable but do not see chemistry.")
    W.h("Scope of this report")
    W.p("This report (Major Project Part I) compares four such routes for the identification of cotton, "
        "polyester and silk, with emphasis on silk vs non-silk: a phone-photograph classifier deployed "
        "as a mobile application, a smartphone-microscope image classifier, NIR spectroscopy, and "
        "ATR-FTIR spectroscopy. Because an NIR spectrometer was not available in the institute, the "
        "spectroscopic parts use carefully selected open-source datasets.")

    # ===== Chapter 2
    W.chapter_start("Literature Review")
    W.h("Overview of chosen topic")
    W.hh("Near-infrared spectroscopy of textile fibres")
    W.p("NIR absorption bands are overtones and combinations of fundamental molecular vibrations [3]. "
        "Cellulosic fibres such as cotton show strong O–H bands near 1490 and 1930 nm and a cellulose "
        "O–H/C–O combination band near 2100 nm. Polyester (PET) shows a sharp aromatic C–H first overtone "
        "near 1660 nm and ester C=O related combination bands near 2130–2160 nm. Protein fibres such as "
        "silk and wool show N–H bands near 1500–1530 nm and amide combination bands near 2050 and "
        "2180 nm. Because the bands are broad and overlapping, chemometric pre-processing and multivariate "
        "models are needed.")
    W.hh("Mid-infrared (ATR-FTIR) spectroscopy")
    W.p("ATR-FTIR spectra contain sharp fundamental bands: the ester C=O stretch of PET at about "
        "1715 cm⁻¹, the C–O stretching region of cellulose near 1030 cm⁻¹, and the amide I (C=O stretch, "
        "~1620–1650 cm⁻¹) and amide II (N–H bend + C–N stretch, ~1510–1540 cm⁻¹) bands of proteins and "
        "polyamides [4]. In silk fibroin the amide I band lies at about 1620 cm⁻¹, characteristic of its "
        "β-sheet crystalline structure [5], which is a useful marker against wool (largely α-helical "
        "keratin) and nylon.")
    W.hh("Image-based deep learning")
    W.p("Convolutional neural networks pre-trained on ImageNet can be fine-tuned with a few hundred to a "
        "few thousand images per class (transfer learning). For textiles they learn texture, weave and "
        "lustre features from photographs or magnified images.")
    W.h("Work done by previous researchers")
    W.p("Riba et al. classified post-consumer textile waste using infrared spectroscopy and multivariate "
        "methods, demonstrating that the main fibre families can be separated for recycling [2]. "
        "Goodge et al. at NIST released NIR-SORT, an open dataset of benchtop and handheld NIR spectra of "
        "113 fabrics of known and real-world provenance, intended for fibre-content classification and "
        "validation of sorting systems [6]. Peets et al. showed that reflectance FT-IR is a viable "
        "non-destructive option for textile fibre identification and released reference spectra [4]. "
        "Parkar et al. published an open ATR-FTIR dataset of 160 spectra from 137 verified natural and "
        "man-made fibre samples covering 26 fibre subtypes, including silk [7]. On the imaging side, "
        "Zhong et al. created TextileNet, a large material-taxonomy image dataset whose fibre labels are "
        "hard to learn from photographs alone (about 50 % top-1 accuracy) [8], while Gupta and Dubey used "
        "consumer clip-on microscopes (50–100×) with CNNs and reported over 90 % accuracy for 14 fabric "
        "classes on their own dataset [9]. Table 2.1 summarises these studies.")
    W.table("Summary of relevant literature and datasets",
            ["Ref.", "Technique", "Fibres / data", "Key finding"],
            [["[2]", "Infrared + chemometrics", "Post-consumer textiles", "Main fibre families separable for recycling"],
             ["[6]", "NIR (833–2500 nm), open data", "113 fabrics, cotton/PET/nylon/blends", "Benchmark for fibre-content classification"],
             ["[4]", "Reflectance FT-IR", "Reference textile fibres", "Non-destructive identification feasible"],
             ["[7]", "ATR-FTIR, open data", "137 samples, 26 subtypes incl. silk", "Reproducible fibre-class signatures"],
             ["[5]", "FTIR of fibroin", "Silk proteins", "Amide I ~1620 cm⁻¹ indicates β-sheet"],
             ["[8]", "Photographs + CNN", "33 fibre labels, web images", "Fibre from photos is hard (~50 %)"],
             ["[9]", "Clip-on microscope + CNN", "14 fabric classes, 840 images", ">90 % on own data"]],
            widths=[0.5, 1.5, 1.8, 2.0])
    W.h("Identified research gaps")
    W.bullets([
        "No open NIR dataset was found that contains silk; open NIR data cover mainly cotton, polyester, "
        "nylon and their blends [6].",
        "Silk is under-represented in every open dataset (a handful of samples in FTIR [7], tens of images "
        "in microscope data [9]), so silk-specific results are rarely reported.",
        "Image-based studies often split images randomly, so photographs of the same fabric can appear in "
        "both training and test sets, which inflates accuracy.",
        "Few studies compare low-cost imaging and spectroscopy for the same fibre set within one study.",
    ])

    # ===== Chapter 3
    W.chapter_start("Objectives and Motivations")
    W.h("Motivation")
    W.p("A practical silk-authentication or fibre-sorting tool must be inexpensive and quick, and its "
        "decisions should be explainable. A phone application is the cheapest possible tool, while "
        "spectroscopy provides chemical evidence. Understanding how far each approach can go – and with "
        "which data – is the motivation of this project.")
    W.h("Objectives")
    W.bullets([
        "To develop an on-device mobile application that classifies fabric photographs as cotton, "
        "polyester or silk.",
        "To evaluate a smartphone-microscope image classifier for silk vs non-silk and for the three "
        "fibres, using open data.",
        "To build chemometric NIR models for cotton vs polyester identification and blend-ratio "
        "prediction using the open NIST NIR-SORT dataset.",
        "To demonstrate, as a proof of concept, ATR-FTIR based identification of silk against cotton, "
        "polyester and other fibres including wool and nylon.",
        "To compare the approaches in terms of accuracy, cost, portability and limitations.",
    ])

    # ===== Chapter 4
    W.chapter_start("Experimental Work")
    W.h("Overall approach")
    W.p("Table 4.1 lists the four approaches, the data used and the models. All code is available in "
        "the project repository (github.com/jagriti12354/fabric-classifier).")
    W.table("Approaches, datasets and models used in this work",
            ["Approach", "Data", "Pre-processing", "Model / validation"],
            [["Phone photograph app", "Roboflow fabric dataset [16]: 2,229 / 159 / 159 images (train/val/test)",
              "224×224, flip, rotation, brightness", "MobileNetV2 [14] → TFLite; held-out test set"],
             ["Smartphone microscope", "Kaggle microscope images [9]: silk 59, cotton 55, polyester 60 (+ 5 other fabrics)",
              "Resize, random crop, flips, rotation, colour jitter", "ResNet18 [15]; 5-fold stratified CV"],
             ["NIR", "NIST NIR-SORT benchtop [6]: 718 spectra, 113 specimens",
              "nm grid 1100–2450, SNV [10], SG 1st derivative [11]", "PLS-DA [12], SVM, PLS; CV grouped by fabric"],
             ["ATR-FTIR", "KCL dataset [7]: 160 spectra, 137 samples, 31 subtypes",
              "Absorbance, 3700–2700 & 1800–600 cm⁻¹, SNV, SG", "PLS-DA, SVM, SIMCA-type one-class [13]; leave-one-sample-out"]],
            widths=[1.1, 1.9, 1.5, 1.6])

    W.h("Phone-photograph classifier and mobile application")
    W.p("Photographs of cotton, polyester and silk fabrics were organised in a Roboflow project and "
        "exported in folder-classification format at 224×224 pixels, with three augmented copies of each "
        "training image (horizontal flip, ±10° rotation, ±15 % brightness) [16]. A MobileNetV2 network [14] "
        "with ImageNet weights was trained in two stages: the new classification head alone for 8 epochs "
        "(Adam, learning rate 10⁻³), then the upper backbone layers for 6 epochs (learning rate 10⁻⁵). "
        "The model was converted to TensorFlow Lite (8.9 MB) with the input normalisation built into the "
        "graph and integrated into a Flutter application that runs inference entirely on the phone.")

    W.h("Smartphone-microscope image classifier")
    W.p("The open microscope dataset of Gupta and Dubey [9] was captured with an Aopick clip-on "
        "microscope attached to an iPhone at 60× magnification. The Contrado-Silk, Cotton and Polyester "
        "classes were used; for a harder test, nylon, lycra, crepe, jersey and velvet fabrics were added "
        "as further non-silk examples. Satin (a weave of unstated fibre) and the cotton–silk blend were "
        "excluded. A ResNet18 [15] pre-trained on ImageNet was fine-tuned for 12 epochs (AdamW, learning "
        "rate 3×10⁻⁴, cosine schedule, class-weighted cross-entropy). Images were randomly cropped, "
        "flipped, rotated and lightly colour-jittered. Performance was estimated with 5-fold stratified "
        "cross-validation, reporting the last-epoch model of each fold to avoid optimistic selection.")
    mp = args.micro_images
    if mp and os.path.isdir(mp):
        paths, labs = [], []
        for cls, folder in [("Cotton", "Cotton"), ("Polyester", "Polyester"), ("Silk", "Silk")]:
            f = first_image(R("dataset", "test", folder))
            if f:
                paths.append(f)
                labs.append(f"Phone photo – {cls}")
        for cls, folder in [("Cotton", "Cotton"), ("Polyester", "Polyester"), ("Silk", "Contrado-Silk")]:
            f = first_image(os.path.join(mp, folder))
            if f:
                paths.append(f)
                labs.append(f"Microscope 60x – {cls}")
        out = R("report", "fig_samples.png")
        montage(paths, labs, out)
        W.figure(out, "Example images. Top row: phone photographs used for the mobile application; "
                      "bottom row: smartphone-microscope (60×) images of cotton, polyester and silk.", 5.4)

    W.h("NIR spectroscopy using the NIST NIR-SORT dataset")
    W.p("Benchtop spectra (Thermo Nicolet iS50 FT-NIR with integrating sphere, 4 cm⁻¹ resolution, "
        "128 scans, about seven replicates per fabric on non-overlapping yarn areas) were downloaded from "
        "the NIST repository [6]. The fibre composition of each specimen is encoded in its file name "
        "(e.g. a065b035 = cotton 65 %, polyester 35 %); recycled polyester was grouped with polyester. "
        "Spectra were converted from wavenumber to wavelength, interpolated onto a 2 nm grid from 1100 to "
        "2450 nm, scatter-corrected by the standard normal variate (SNV) [10] and differentiated with a "
        "Savitzky–Golay filter (window 15, second-order polynomial, first derivative) [11].")
    W.p("Two tasks were defined: (a) classification of pure cotton (40 fabrics) vs pure polyester "
        "(9 fabrics) with PLS-DA (5 latent variables) [12] and an RBF support vector machine, and "
        "(b) prediction of cotton content in 80 cotton/polyester specimens (0–100 %) with PLS regression, "
        "selecting the number of latent variables by RMSECV. All cross-validation was grouped by fabric, "
        "so the replicate spectra of one fabric were never split between training and test.")

    W.h("ATR-FTIR proof of concept for silk")
    W.p("The KCL ATR-FTIR dataset [7] (PerkinElmer Frontier, diamond ATR, 4000–550 cm⁻¹, 4 cm⁻¹) contains "
        "7 silk samples (Bombyx mori and tussah), 10 cotton, 15 PET, 19 wool, 2 alpaca and 17 nylon "
        "samples among 31 fibre subtypes. The baseline-corrected transmittance spectra were converted to "
        "absorbance; the 3700–2700 and 1800–600 cm⁻¹ regions were retained (the 2400–1800 cm⁻¹ region "
        "contains diamond and CO₂ artefacts), followed by SNV and a Savitzky–Golay first derivative.")
    W.p("Three tasks were evaluated with leave-one-sample-out validation: (A) silk / cotton / polyester; "
        "(B) silk vs all other 30 fibre types; and (C) a one-class, SIMCA-type model [13] in which a "
        "3-component PCA model is built on silk only and a new spectrum is accepted as silk if its "
        "squared residual Q is below 1.5 times the 95th percentile of cross-validated silk residuals. "
        "Task C reflects the practical authentication question – ‘is this silk?’ – without having to "
        "model every possible imitation.")
    W.h("Evaluation metrics")
    W.p("Classification is reported as accuracy, per-class precision and recall, and confusion matrices; "
        "for spectroscopy both spectrum-level and sample-level (majority vote over replicates) accuracy "
        "are given. Regression is reported as RMSECV and R².")

    # ===== Chapter 5
    W.chapter_start("Results and Discussion")
    W.h("Phone-photograph classifier")
    W.table("Data split and test performance of the MobileNetV2 phone-photograph classifier",
            ["Split", "Cotton", "Polyester", "Silk", "Total"],
            [["Train", 837, 783, 609, "2,229"], ["Validation", 59, 56, 44, 159], ["Test", 60, 56, 43, 159],
             ["Test accuracy", "", "", "", "70.4 %"]], widths=[1.4, 1, 1, 1, 1])
    W.p("The model reached 70.4 % accuracy on the held-out test set, well above the 33 % chance level "
        "for three classes. Validation accuracy stayed at about 72 % during fine-tuning while training "
        "accuracy kept increasing, indicating mild over-fitting. Cotton and polyester were the more "
        "reliably separated pair; silk had the fewest images and the largest visual overlap with "
        "the other classes – a smooth polyester satin looks very similar to silk in a photograph. The "
        "application currently has no ‘other fibre’ output, so wool or denim are forced into one of the "
        "three classes.")

    W.h("Smartphone-microscope classifier")
    if M:
        rows = []
        names = {"binary": "Silk vs cotton + polyester", "3class": "Silk / cotton / polyester",
                 "binary_extra": "Silk vs 7 other fabric types"}
        for k in ["binary", "3class", "binary_extra"]:
            if k in M:
                m = M[k]
                silk_i = m["classes"].index("silk")
                rows.append([names[k], m["n"], f"{m['acc']*100:.1f} ± {m['acc_sd']*100:.1f}",
                             f"{m['f1']:.3f} ± {m['f1_sd']:.3f}",
                             f"{m['rec'][silk_i]*100:.1f}", f"{m['prec'][silk_i]*100:.1f}"])
        W.table("Cross-validated performance of the ResNet18 smartphone-microscope classifier "
                "(5-fold CV, mean ± SD over folds)",
                ["Task", "Images", "Accuracy (%)", "Macro F1", "Silk recall (%)", "Silk precision (%)"],
                rows, widths=[1.9, 0.7, 1.0, 1.0, 0.8, 0.8])
        cmf = R("research", "microscope", "outputs_binary", "confusion_matrix.png")
        W.figure(cmf, "Pooled cross-validation confusion matrix of the silk vs non-silk "
                      "(cotton + polyester) smartphone-microscope classifier.", 3.8)
        W.p("The magnified images give the network access to yarn and weave structure that is not "
            "visible in ordinary photographs, and every image was classified correctly in all five folds. "
            "Such perfect separation should not be over-interpreted. "
            "However, the silk images most likely come from only one or two fabrics, and images were "
            "split randomly, so photographs of the same fabric can occur in both training and test folds. "
            "These numbers are therefore an upper bound; an external test on new fabrics is required "
            "before the method can be considered validated.")
    else:
        W.p("[Results pending – run research/microscope/silk_classifier.py and rebuild the report.]",
            italic=True)

    W.h("NIR spectroscopy (NIST NIR-SORT)")
    W.figure(R("research", "nir", "outputs", "mean_spectra.png"),
             "Mean NIR spectra of pure cotton and polyester fabrics (top: raw absorbance; bottom: after SNV "
             "and Savitzky–Golay first derivative) with the main band assignments.")
    W.p("Figure 5.2 shows the chemical basis of the separation. Polyester has a sharp aromatic C–H band "
        "at about 1660 nm and several sharp combination bands between 2130 and 2300 nm, while cotton "
        "shows broad O–H bands around 1490 and 1930 nm and the cellulose band near 2100 nm. Table 5.3 "
        "summarises the models.")
    W.table("Results of the NIR models (cross-validation grouped by fabric)",
            ["Task", "Data", "Model", "Result"],
            [["Cotton vs polyester", "347 spectra, 40 + 9 fabrics", "PLS-DA (5 LV)", "100 % (spectrum and fabric level)"],
             ["Cotton vs polyester", "347 spectra, 40 + 9 fabrics", "SVM (RBF)", "100 % (spectrum and fabric level)"],
             ["Cotton % in blends", "487 spectra, 80 specimens", "PLS (6 LV)", "RMSECV 4.93 %, R² 0.983"]],
            widths=[1.4, 1.7, 1.1, 1.8])
    W.figure(R("research", "nir", "outputs", "blend_regression.png"),
             "Cotton-content prediction in cotton/polyester blends: RMSECV vs number of PLS latent variables "
             "(left) and predicted vs labelled cotton content (right).")
    W.p("Pure cotton and polyester were perfectly separated, which is expected given the distinct "
        "chemistry; the result should nevertheless be read in the light of the small number of pure "
        "polyester fabrics (9). Blend content was predicted with an error of about 5 % cotton, which is "
        "useful for recycling, where feedstock is usually specified in broad composition bands. A few "
        "specimens deviate strongly (e.g. some 30 % labels predicted near 5 % or 60 %), which may reflect "
        "label uncertainty, knit vs woven structure or finishes. Silk could not be studied with NIR "
        "because no open NIR dataset containing silk was found.")

    W.h("ATR-FTIR proof of concept for silk")
    W.figure(R("research", "ftir", "outputs", "ftir_mean_spectra.png"),
             "Top: mean ATR-FTIR spectra of silk, cotton and polyester with band assignments. Bottom: amide "
             "region of silk, wool and nylon 6.")
    W.p("Figure 5.4 shows that the three target fibres have very different mid-infrared spectra: the "
        "ester C=O (1715 cm⁻¹), C–O (1240 cm⁻¹) and aromatic (725 cm⁻¹) bands of PET; the dominant C–O "
        "region near 1030 cm⁻¹ of cotton; and the amide I and amide II bands of silk. More importantly, "
        "the zoomed amide region shows that silk differs from the other amide-containing fibres: its "
        "amide I maximum is at about 1620 cm⁻¹ (β-sheet fibroin [5]) compared with about 1635 cm⁻¹ for "
        "wool and nylon, and its amide II band is at about 1512 cm⁻¹ compared with about 1535 cm⁻¹.")
    W.table("Results of the ATR-FTIR models (leave-one-sample-out)",
            ["Task", "Samples", "Model", "Result"],
            [["A. Silk / cotton / PET", "32 (7 / 10 / 15)", "PLS-DA (4 LV)", "30/32 samples; spectrum accuracy 92.9 %; silk recall 100 %"],
             ["A. Silk / cotton / PET", "32", "SVM (RBF)", "29/32 samples; spectrum accuracy 90.5 %"],
             ["B. Silk vs all others", "137 (31 subtypes)", "PLS-DA and SVM", "136/137 samples; silk recall 100 %, precision 91.7 %"],
             ["C. One-class silk", "Trained on silk only", "PCA-Q (3 PCs)", "Silk accepted 11/11; non-silk rejected 146/149"]],
            widths=[1.5, 1.3, 1.2, 2.0])
    W.figure(R("research", "ftir", "outputs", "ftir_one_class_silk.png"),
             "One-class silk model: squared residual Q of each spectrum from the silk PCA model; spectra "
             "below the red threshold are accepted as silk.")
    W.p("Silk was never missed in tasks A and B. The single non-silk sample predicted as silk in task B "
        "was an indigo-dyed cotton denim, probably because dye or sizing residues alter its surface "
        "spectrum, and one plain cotton and one PET sample were confused in task A. In the one-class "
        "model (Figure 5.5) wool, nylon, cotton and PET lie well above the threshold, whereas the two "
        "alpaca spectra – another protein fibre – lie on the threshold. This indicates that separating "
        "silk from other protein fibres is the real challenge and that more silk samples are needed "
        "to set a robust threshold. With only seven silk samples these results are a proof of concept.")

    W.h("Comparison of approaches")
    micro_cell = (f"{M['binary']['acc']*100:.0f} % (silk vs non-silk, CV)" if "binary" in M else "see §5.2")
    W.table("Comparison of the four approaches",
            ["Approach", "Cost / portability", "Silk vs non-silk", "Cotton vs polyester", "Basis"],
            [["Phone photo + MobileNetV2", "Phone only; fully portable", "Weakest (most confused class)", "Moderate", "Appearance"],
             ["Clip-on microscope + ResNet18", "₹500–3,000 add-on; portable", micro_cell + " – upper bound", "Not tested separately", "Magnified texture"],
             ["NIR (benchtop)", "Lab instrument; handhelds exist", "Not tested (no open silk data)", "100 %", "Chemistry (overtones)"],
             ["ATR-FTIR", "Lab instrument, common in institutes", "136/137 samples", "Separated in 3-class model", "Chemistry (fundamentals)"]],
            widths=[1.4, 1.3, 1.3, 1.1, 0.9])
    W.p("Image-based methods are the cheapest and most portable but depend on appearance, which can be "
        "imitated; spectroscopic methods identify the polymer itself and are therefore much more reliable, "
        "and in the case of silk the amide band positions provide direct chemical evidence. A practical "
        "system could combine both: a phone or microscope screen, with a confirmatory spectroscopic or "
        "simple chemical test (burning or 5 % NaOH dissolution [1]) when the image model is uncertain.")
    W.h("Limitations")
    W.bullets([
        "Silk data are scarce in all open datasets (7 FTIR samples, about 59 microscope images, no NIR).",
        "No external validation on fabrics measured in-house; spectroscopic results rely on open data.",
        "Random image-level splits in the microscope data may place images of one fabric in both training "
        "and test folds.",
        "Model settings (PLS latent variables, one-class threshold) were chosen with the same "
        "cross-validation that is reported, which is slightly optimistic.",
        "The mobile application cannot reject fibres outside the three trained classes.",
    ])

    # ===== Chapter 6
    W.chapter_start("Conclusions")
    W.h("Conclusions")
    W.bullets([
        "A working on-device mobile application was developed that classifies fabric photographs into "
        "cotton, polyester and silk with 70.4 % test accuracy; silk is the most difficult class from "
        "photographs.",
        "Smartphone-microscope images improve image-based discrimination because they capture yarn and "
        "weave structure, but available silk data are too limited for a validated claim.",
        "Using open NIR data, cotton and polyester were identified with 100 % accuracy and cotton content "
        "in blends was predicted with an RMSECV of 4.9 %, confirming NIR as the method of choice for "
        "sorting these fibres.",
        "ATR-FTIR identified silk against 30 other fibre types for 136 of 137 samples; the amide I band at "
        "~1620 cm⁻¹ (β-sheet) distinguishes silk from wool and nylon.",
        "Spectroscopy provides chemically explainable identification, while imaging provides low cost and "
        "portability; the two are complementary.",
    ])
    W.h("Future work (Part II)")
    W.bullets([
        "Measure in-house silk (mulberry, tussar, eri, muga), cotton and polyester samples on an ATR-FTIR "
        "instrument, or on a UV-Vis-NIR spectrophotometer with an integrating sphere (to 2500 nm), as an "
        "external test set.",
        "Collect a larger clip-on microscope image set with fabric-wise splits.",
        "Add a rejection (‘not cotton/polyester/silk’) option to the mobile application using a "
        "one-class approach.",
        "Combine image screening with a confirmatory chemical or spectroscopic test for silk.",
    ])

    # ===== References
    W.unnumbered_chapter("References")
    refs = [
        "Fibers in textiles: identification, AATCC Test Method 20-2021, American Association of Textile Chemists and Colorists, 2021.",
        "Riba, J.-R., Cantero, R., Canals, T., Puig, R., Circular economy of post-consumer textile waste: classification through infrared spectroscopy. Journal of Cleaner Production, 272, 123011, 2020.",
        "Workman, J., Weyer, L., Practical guide to interpretive near-infrared spectroscopy, CRC Press (Boca Raton), 2008.",
        "Peets, P., Kaupmees, K., Vahur, S., Leito, I., Reflectance FT-IR spectroscopy as a viable option for textile fiber identification. Heritage Science, 7, 93, 2019.",
        "Hu, X., Kaplan, D., Cebe, P., Determining beta-sheet crystallinity in fibrous proteins by thermal analysis and infrared spectroscopy. Macromolecules, 39, 6161-6170, 2006.",
        "Goodge, K.E., Vederman, C.J., Patel, R.V., Wentz, C.M., Tsinas, Z., Landauer, A.K., Forster, A.L., A spectroscopic dataset for known provenance and post-consumer textiles. Scientific Data, 13, 1188, 2026. Data: https://doi.org/10.18126/61v0-rf98 [accessed 16 November 2026].",
        "Parkar, R., Jain, A., et al., Infrared spectral data of natural and man-made textile fibres for material identification and classification. Data in Brief, 2026. Data: Mendeley Data V3, https://doi.org/10.17632/rx3fjgz96x.3 [accessed 16 November 2026].",
        "Zhong, S., Ribul, M., Cho, Y., Obrist, M., TextileNet: a material taxonomy-based fashion textile dataset. arXiv preprint arXiv:2301.06160, 2023.",
        "Gupta, D., Dubey, A., Towards improved sustainability in the textile lifecycle with deep learning. Proceedings of the 7th ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies (COMPASS ’24), ACM (New Delhi), 2024. Data: https://www.kaggle.com/datasets/danikagupta/microscope-images-of-fabric [accessed 20 November 2026].",
        "Barnes, R.J., Dhanoa, M.S., Lister, S.J., Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. Applied Spectroscopy, 43, 772-777, 1989.",
        "Savitzky, A., Golay, M.J.E., Smoothing and differentiation of data by simplified least squares procedures. Analytical Chemistry, 36, 1627-1639, 1964.",
        "Wold, S., Sjöström, M., Eriksson, L., PLS-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58, 109-130, 2001.",
        "Wold, S., Pattern recognition by means of disjoint principal components models. Pattern Recognition, 8, 127-139, 1976.",
        "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C., MobileNetV2: inverted residuals and linear bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 4510-4520, 2018.",
        "He, K., Zhang, X., Ren, S., Sun, J., Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770-778, 2016.",
        "https://universe.roboflow.com/fabric-classification-01qgr/fabric-classification-uimqb [accessed 20 November 2026].",
    ]
    for i, r_ in enumerate(refs, 1):
        par = W.p(f"[{i}]  {r_}", align=WD_ALIGN_PARAGRAPH.LEFT)
        par.paragraph_format.left_indent = Inches(0.45)
        par.paragraph_format.first_line_indent = Inches(-0.45)

    # ===== Appendix
    W.unnumbered_chapter("Appendix")
    W.p("**A. Code and data availability.** All scripts are in the repository "
        "github.com/jagriti12354/fabric-classifier: research/nir/nir_cotton_polyester.py (NIR), "
        "research/ftir/ftir_silk.py (ATR-FTIR), research/microscope/silk_classifier.py (microscope CNN), "
        "train_fiber_classifier.py and lib/ (mobile application). The spectroscopic scripts download the "
        "open datasets automatically.")
    W.p("**B. NIR-SORT file-name fibre codes.** a = cotton, b = polyester, c = nylon, d = acrylic, "
        "e = spandex, f = recycled polyester, g = rayon; the three digits after each letter give the "
        "percentage.")
    W.figure(R("research", "nir", "outputs", "pca_scores.png"),
             "PCA scores of pre-processed NIR spectra of pure cotton and polyester fabrics.", 4.0)
    W.figure(R("research", "ftir", "outputs", "ftir_pca.png"),
             "PCA scores of pre-processed ATR-FTIR spectra of silk, cotton, PET, wool and nylon 6.", 4.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc.save(args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro-images", default=None)
    ap.add_argument("--logo", default=None)
    ap.add_argument("--out", default=R("report", "TXD401_Final_Report.docx"))
    build(ap.parse_args())
