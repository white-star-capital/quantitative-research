const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// ── Colour palette ──────────────────────────────────────────────────────────
const NAVY   = "1F3864";   // WSC deep navy
const GOLD   = "BF9000";   // WSC accent gold
const LIGHT  = "EBF0F7";   // header row fill
const MID    = "D6E4F0";   // alt row fill
const WHITE  = "FFFFFF";
const GRAY   = "595959";

// ── Helpers ──────────────────────────────────────────────────────────────────
const sp = (before, after) => ({ spacing: { before, after } });
const border = (color = "CCCCCC") => ({ style: BorderStyle.SINGLE, size: 1, color });
const allBorders = (color = "CCCCCC") => ({ top: border(color), bottom: border(color), left: border(color), right: border(color) });

function cell(text, { bold = false, fill = WHITE, color = "000000", align = AlignmentType.LEFT, w, italic = false } = {}) {
  return new TableCell({
    borders: allBorders("CCCCCC"),
    width: { size: w, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, bold, italic, color, font: "Calibri", size: 19 })]
    })]
  });
}

function hCell(text, w) {
  return cell(text, { bold: true, fill: NAVY, color: WHITE, w, align: AlignmentType.CENTER });
}

function para(text, { bold = false, size = 22, color = "000000", before = 80, after = 80, align = AlignmentType.LEFT, italic = false } = {}) {
  return new Paragraph({
    alignment: align,
    ...sp(before, after),
    children: [new TextRun({ text, bold, italic, color, font: "Calibri", size })]
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    ...sp(40, 40),
    children: [new TextRun({ text, font: "Calibri", size: 22, color: "000000" })]
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    ...sp(60, 60),
    children: [new TextRun({ text, font: "Calibri", size: 22, color: "000000" })]
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    ...sp(360, 120),
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY, space: 4 } },
    children: [new TextRun({ text, bold: true, font: "Calibri", size: 32, color: NAVY })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    ...sp(240, 80),
    children: [new TextRun({ text, bold: true, font: "Calibri", size: 26, color: NAVY })]
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    ...sp(180, 60),
    children: [new TextRun({ text, bold: true, italic: true, font: "Calibri", size: 23, color: GRAY })]
  });
}

function codeBlock(lines) {
  return lines.map(line => new Paragraph({
    ...sp(0, 0),
    indent: { left: 720 },
    children: [new TextRun({ text: line, font: "Courier New", size: 19, color: "1F3864" })]
  }));
}

function divider() {
  return new Paragraph({
    ...sp(120, 120),
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 1 } },
    children: []
  });
}

// ── Performance Results Table ─────────────────────────────────────────────
// Total content width = 9360 DXA (US Letter, 1" margins)
const colW1 = [3000, 1620, 1620, 1560, 1560]; // Performance table
function performanceTable() {
  const headers = ["Strategy", "Sharpe Ratio", "Annual Return", "Max Drawdown", "Volatility"];
  const rows = [
    ["Equal-Weight Subnets",        "2.59",   "+67%",  "-5.9%",  "17.9%"],
    ["PCA-Optimised (\u03b3=0, K=15)", "2.46", "+90%",  "-9.6%",  "23.9%"],
    ["PCA Min-Variance (\u03b3=1, K=5)","1.46","+76%", "-11.6%", "35.5%"],
    ["Value-Weight (Market Cap)",   "-1.22",  "-20%",  "-16.3%", "22.6%"],
    ["TAO Buy-and-Hold",            "~0.00",  "~0%",   "n/a",    "n/a"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colW1,
    rows: [
      new TableRow({ children: headers.map((h, i) => hCell(h, colW1[i])) }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((v, i) => cell(v, {
          w: colW1[i],
          fill: ri % 2 === 0 ? WHITE : MID,
          bold: i === 0,
          align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          color: v.startsWith("-") && i > 0 ? "C00000" : (v.startsWith("+") || parseFloat(v) > 0 ? "375623" : "000000")
        }))
      }))
    ]
  });
}

// ── Gamma Sensitivity Table ───────────────────────────────────────────────
const colW2 = [2200, 2280, 4880];
function gammaTable() {
  const headers = ["\u03b3 Value", "RP-PCA Tangency Sharpe", "Interpretation"];
  const rows = [
    ["0 (centred PCA)",      "+1.26", "\u2713 Positive \u2014 variance structure only"],
    ["1 (uncentred PCA)",    "+1.40", "\u2713 Positive \u2014 second-moment matrix"],
    ["5",                    "+0.27", "Degrading \u2014 mean noise entering"],
    ["10",                   "-0.14", "Negative \u2014 factor rotation corrupted"],
    ["25",                   "-0.53", "Significantly negative"],
    ["50",                   "-0.49", "Significantly negative"],
    ["T (auto \u2248 398)", "-1.89", "Catastrophic \u2014 full noise amplification"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colW2,
    rows: [
      new TableRow({ children: headers.map((h, i) => hCell(h, colW2[i])) }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((v, i) => cell(v, {
          w: colW2[i],
          fill: ri % 2 === 0 ? WHITE : MID,
          bold: i === 0,
          align: i === 0 ? AlignmentType.CENTER : (i === 1 ? AlignmentType.CENTER : AlignmentType.LEFT),
          color: v.startsWith("-") && i === 1 ? "C00000" : (v.startsWith("+") ? "375623" : "000000")
        }))
      }))
    ]
  });
}

// ── Component Count Table ────────────────────────────────────────────────
const colW3 = [900, 2700, 1440, 2160, 2160];
function componentTable() {
  const headers = ["K", "Best Strategy", "Sharpe", "Annual Return", "Max DD"];
  const rows = [
    ["3",  "PCA Min-Var",       "1.29", "+101%", "-20.6%"],
    ["5",  "PCA Min-Var",       "1.46", "+76%",  "-11.6%"],
    ["7",  "PCA Tangency",      "1.53", "+62%",  "-11.1%"],
    ["10", "PCA Min-Var",       "1.66", "+108%", "-14.5%"],
    ["15", "PCA Tangency",      "2.46", "+90%",  "-9.6%"],
    ["20", "PCA Tangency",      "0.84", "+25%",  "-15.2%"],
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colW3,
    rows: [
      new TableRow({ children: headers.map((h, i) => hCell(h, colW3[i])) }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((v, i) => cell(v, {
          w: colW3[i],
          fill: ri === 4 ? LIGHT : (ri % 2 === 0 ? WHITE : MID),
          bold: ri === 4,
          align: i <= 1 ? AlignmentType.CENTER : AlignmentType.CENTER,
          color: v.startsWith("+") && i >= 3 ? "375623" : (v.startsWith("-") && i >= 3 ? "C00000" : "000000")
        }))
      }))
    ]
  });
}

// ── Header / Footer ──────────────────────────────────────────────────────
function makeHeader() {
  return new Header({
    children: [
      new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 3 } },
        ...sp(0, 60),
        children: [
          new TextRun({ text: "WHITE STAR CAPITAL", bold: true, font: "Calibri", size: 18, color: NAVY }),
          new TextRun({ text: "\t", font: "Calibri", size: 18 }),
          new TextRun({ text: "CONFIDENTIAL \u2014 INTERNAL MEMO", font: "Calibri", size: 18, color: GRAY, italic: true }),
        ]
      })
    ]
  });
}

function makeFooter() {
  return new Footer({
    children: [
      new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 3 } },
        ...sp(60, 0),
        children: [
          new TextRun({ text: "Quantitative Research Desk \u2022 White Star Capital \u2022 March 2026", font: "Calibri", size: 16, color: GRAY, italic: true }),
          new TextRun({ text: "\t", font: "Calibri", size: 16 }),
          new TextRun({ text: "Page ", font: "Calibri", size: 16, color: GRAY }),
          new TextRun({ children: [PageNumber.CURRENT], font: "Calibri", size: 16, color: GRAY }),
        ]
      })
    ]
  });
}

// ── Title block ──────────────────────────────────────────────────────────
function titleBlock() {
  const metaColW = [2200, 7160];
  const metaRows = [
    ["To:",             "Investment Partners"],
    ["From:",           "Quantitative Research Desk"],
    ["Date:",           "March 25, 2026"],
    ["Re:",             "TAO Subnet Portfolio Strategy \u2014 RP-PCA Framework Assessment"],
    ["Classification:", "Confidential"],
  ];
  return [
    para("WHITE STAR CAPITAL", { bold: true, size: 52, color: NAVY, align: AlignmentType.CENTER, before: 0, after: 60 }),
    para("INTERNAL INVESTMENT MEMO", { bold: false, size: 26, color: GOLD, align: AlignmentType.CENTER, before: 0, after: 300, italic: true }),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: metaColW,
      rows: metaRows.map(([label, value], ri) => new TableRow({
        children: [
          cell(label, { bold: true, fill: ri % 2 === 0 ? LIGHT : WHITE, w: metaColW[0], color: NAVY }),
          cell(value, { fill: ri % 2 === 0 ? LIGHT : WHITE, w: metaColW[1], bold: label === "Classification:" }),
        ]
      }))
    }),
    para(" ", { before: 0, after: 0 }),
  ];
}

// ── Document assembly ────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22, color: "000000" } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Calibri", color: NAVY },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: NAVY },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, italic: true, font: "Calibri", color: GRAY },
        paragraph: { spacing: { before: 180, after: 60 }, outlineLevel: 2 }
      },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1260, bottom: 1080, left: 1260 }
      }
    },
    headers: { default: makeHeader() },
    footers: { default: makeFooter() },
    children: [

      // ── TITLE ──────────────────────────────────────────────────────────
      ...titleBlock(),

      // ═══════════════════════════════════════════════════════════════════
      // PART I — PLAIN ENGLISH
      // ═══════════════════════════════════════════════════════════════════
      h1("PART I \u2014 EXECUTIVE SUMMARY"),

      // What We Built
      h2("What We Built and What We Found"),
      para("We set out to test whether an academic machine-learning framework \u2014 Risk-Premium PCA, developed by leading finance professors at Stanford and Vienna \u2014 could be used to construct a superior portfolio of Bittensor (TAO) subnet tokens. The short answer: the specific \u201Crisk-premium\u201D enhancement from the paper does not improve returns in this market. However, in the process of testing it, we uncovered something more interesting and more reliable: simple, systematic diversification across TAO subnets generates exceptional risk-adjusted returns that significantly outperform both TAO buy-and-hold and market-cap-weighted approaches."),

      // The Bittensor Opportunity
      h2("The Bittensor Opportunity in Plain Terms"),
      para("Bittensor is a decentralised AI network where 128 \u201Csubnets\u201D compete to provide different AI services \u2014 some do image generation, some run language models, some handle data storage. Each subnet has its own token whose price fluctuates relative to TAO. Think of it like a portfolio of 128 early-stage tech startups: most will underperform, but a handful will massively outperform, and you don\u2019t know which ones in advance."),
      para("This is exactly the environment where diversification earns its keep. Our analysis found:", { after: 40 }),
      bullet("Only 30% of subnets had positive returns relative to TAO over the study period"),
      bullet("The winners won by far more than the losers lost"),
      bullet("The market systematically over-weights the losers \u2014 because value-weighting concentrates capital in large, established subnets that are already priced in"),
      para(" ", { before: 0, after: 0 }),

      // Numbers
      h2("What the Numbers Show"),
      para("Over the out-of-sample test period (October 2025 to March 2026), a simple equal-weight portfolio of all 128 subnets achieved a Sharpe ratio of 2.59 with a maximum drawdown of only \u22125.9% \u2014 measured in TAO terms. For context, Bitcoin\u2019s Sharpe ratio over the same period was approximately 0.20. A market-cap-weighted portfolio (buying more of the biggest subnets) lost 20% annualised over the same period.", { after: 120 }),

      performanceTable(),

      para("Note: All returns are TAO-denominated. USD performance depends on the cost and availability of a TAO/USD overlay hedge.", { italic: true, size: 19, color: GRAY, before: 80, after: 120 }),

      // Key Insight
      h2("The Key Insight"),
      para("The strategy that beats the market is not the complex one \u2014 it is the disciplined one. By systematically holding all subnets in equal proportion and rebalancing monthly, a portfolio captures the positive skew of the subnet ecosystem: the handful of breakout subnets that generate enormous returns more than compensate for the many that decline. This is the same logic behind venture capital portfolio construction, applied to liquid, daily-rebalanced tokens."),

      // What Doesn't Work
      h2("What Doesn\u2019t Work \u2014 and Why It Matters"),
      para("The academic framework we tested (RP-PCA) attempts to identify \u201Crewarded risk factors\u201D \u2014 clusters of assets that move together and tend to generate positive returns. The idea is theoretically sound and works in US equities. In TAO subnets, it fails because the mean return estimates it relies on are too noisy: in a market where 70% of assets are declining, the model\u2019s \u201Cfind the direction of high expected return\u201D signal gets corrupted, and the portfolios it constructs destroy value. This is an important negative result \u2014 it tells us this market is structurally different from equities, and that simpler approaches are more robust."),

      // Recommendation
      h2("Recommendation"),
      para("Based on this analysis, we recommend the following course of action:", { after: 40 }),
      bullet("The diversification-based TAO subnet strategy merits further investigation and potential pilot deployment"),
      bullet("Equal-weight or lightly PCA-optimised (variance-only) portfolios are the deployable candidates"),
      bullet("A USD overlay (TAO hedge) is required before capital deployment \u2014 this is a relative-value strategy, not an absolute return strategy"),
      bullet("Minimum 12 months of additional live data needed before committing significant capital"),
      bullet("Suggested pilot size: $500K\u2013$2M, TAO-denominated, with full TAO/USD hedge"),
      para(" ", { before: 0, after: 0 }),
      divider(),

      // ═══════════════════════════════════════════════════════════════════
      // PART II — TECHNICAL
      // ═══════════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      h1("PART II \u2014 TECHNICAL DETAILS"),
      h2("Methodology and Implementation"),

      // 1. Data
      h3("1. Data and Universe"),
      para("The analysis uses daily closing prices for all 128 active Bittensor subnets, sourced from Taostats exports in TAO-denominated terms. The full dataset spans February 2025 to March 2026 (398 trading days). All prices are expressed relative to TAO (i.e., what fraction of one TAO token does one subnet token cost). Returns are computed as log(P\u209C / P\u209C\u208B\u2081), winsorised at the 1st/99th percentile to remove data errors, and a minimum 50% observation coverage filter is applied per asset. The resulting return matrix is 398 \u00D7 128."),

      // 2. Framework
      h3("2. The RP-PCA Framework"),
      para("Risk-Premium PCA (Lettau & Pelger, Review of Financial Studies 2020) modifies standard PCA by decomposing a composite matrix:"),
      ...codeBlock(["M = \u03a3 + \u03b3 \u00B7 \u03bc \u00B7 \u03bcᵀ"]),
      para("where \u03a3 is the N\u00D7N sample covariance matrix of returns, \u03bc is the N-vector of mean returns, and \u03b3 is a scalar penalty parameter. When \u03b3 = 0 this reduces to standard centred PCA (variance-maximising factors). When \u03b3 > 1, eigenvectors are tilted toward directions of high expected return \u2014 so-called \u201Crewarded risk factors.\u201D The top K eigenvectors (loadings matrix L of size N\u00D7K) are extracted, factor returns computed as F = X\u00B7L, and portfolio weights constructed via:", { after: 40 }),
      ...codeBlock([
        "Tangency:        w_F = \u03a3_F\u207B\u00B9 \u00B7 (\u03bc_F \u2212 r_f) / (1\u1D40 \u00B7 \u03a3_F\u207B\u00B9 \u00B7 (\u03bc_F \u2212 r_f))",
        "Min-Variance:    w_F = \u03a3_F\u207B\u00B9 \u00B7 1  / (1\u1D40 \u00B7 \u03a3_F\u207B\u00B9 \u00B7 1)",
        "Asset space:     w_asset = L @ w_F",
      ]),
      para(" ", { before: 0, after: 0 }),

      // 3. Backtest
      h3("3. Walk-Forward Backtest Design"),
      para("The backtest uses a rolling walk-forward structure to eliminate look-ahead bias. At each rebalance date t: (1) estimate \u03a3 on the covariance slice and \u03bc on the mean slice; (2) form M; (3) eigendecompose to extract loadings L; (4) construct portfolio weights; (5) hold for 21 days, applying weights to out-of-sample returns only.", { after: 40 }),
      bullet("Covariance estimation window: 252 trading days (rolling)"),
      bullet("Mean estimation window: 63 trading days (shorter, reflecting mean instability)"),
      bullet("Rebalance frequency: every 21 trading days"),
      bullet("OOS period: 24 October 2025 \u2013 18 March 2026 (146 days, 7 rebalances)"),
      bullet("Covariance method: sample covariance; Ledoit-Wolf shrinkage produces identical results, confirming factor structure dominates"),
      bullet("No transaction costs modelled (identified as future work)"),
      para(" ", { before: 0, after: 0 }),

      // 4. Gamma sensitivity
      h3("4. The \u03b3 Sensitivity \u2014 Core Empirical Finding"),
      para("The table below shows how the OOS Sharpe ratio of the RP-PCA Tangency strategy varies monotonically with \u03b3, holding K = 5 and cov_method = sample fixed.", { after: 100 }),
      gammaTable(),
      para("Interpretation: Mean return estimates over 63-day windows are dominated by noise in this market. The TAO subnet universe experienced a broad drawdown throughout the study period \u2014 only 29.7% of assets have a positive mean return; the cross-sectional mean daily return is \u22120.15%. When \u03b3 > 1, the composite matrix M is corrupted by a noisy \u03bc\u03bcᵀ term, rotating eigenvectors toward directions of past (not future) high return. The RP-PCA innovation requires stable, persistent risk premia \u2014 a condition that does not hold in this nascent market.", { before: 100 }),

      // 5. Component count
      h3("5. Component Count Sensitivity (\u03b3 = 0)"),
      para("Results for varying K (number of PCA components) at \u03b3 = 0, sample covariance. The highlighted row (K = 15) represents the best-performing configuration.", { after: 100 }),
      componentTable(),
      para("The sweet spot is K = 10\u201315 at \u03b3 = 0, balancing dimensionality reduction with covariance estimation accuracy. At K = 20 (ratio K/N \u2248 16%), performance degrades as the model begins overfitting the training covariance.", { before: 100 }),

      // 6. Robustness
      h3("6. Robustness Checks"),

      para("Bootstrap Test (Politis-Romano 1994 \u2014 1,000 circular block resamples, block length 5 days)", { bold: true, before: 120, after: 40 }),
      bullet("PCA Tangency vs. Equal-Weight: observed Sharpe difference = \u22121.33; P(PCA > EW) = 21.5%; 95% CI = [\u22124.62, +1.85]. PCA does not statistically dominate equal-weight."),
      bullet("PCA Tangency vs. Value-Weight: P(PCA > VW) = 92.1%; 95% CI = [\u22120.71, +5.02]. PCA significantly outperforms value-weighting, though confidence is limited by the short sample."),
      para(" ", { before: 0, after: 0 }),

      para("Fama-MacBeth Cross-Sectional Tests (Shanken 1992 correction, 50/50 sample split)", { bold: true, before: 120, after: 40 }),
      bullet("RP-PCA: mean cross-sectional R\u00B2 = 9.83%; Factor 1 risk price: \u03bb\u2081 = \u22120.012 (t = \u22121.78). No factor achieves |t| > 2.0."),
      bullet("PCA: mean cross-sectional R\u00B2 = 9.97%. All factor risk prices statistically insignificant."),
      bullet("Interpretation: extracted factors explain \u223C10% of cross-sectional return variation. No priced risk factor is identified \u2014 consistent with a short, noisy dataset rather than evidence of an absent factor structure."),
      para(" ", { before: 0, after: 0 }),

      para("Regime Analysis", { bold: true, before: 120, after: 40 }),
      bullet("Bear regime (182 in-sample days): no OOS observations fall in bear regime \u2014 strategy behaviour in a sustained TAO bear market is untested."),
      bullet("Bull regime (117 OOS days): Equal-weight Sharpe = 3.04; PCA Min-Var = 1.99; RP-PCA strategies negative."),
      bullet("Sideways regime (29 OOS days): Equal-weight Sharpe = 0.75; PCA Tangency = 1.74."),
      para(" ", { before: 0, after: 0 }),

      // 7. Risks
      h3("7. Key Risks and Open Questions"),
      numbered("Sample size: 146 OOS days and 7 rebalances are insufficient for statistical inference at conventional confidence levels. A minimum of 3 years of OOS data would be required for publication-standard robustness."),
      numbered("TAO-denomination: all returns are expressed in TAO. USD P&L requires a TAO/USD overlay. Current TAO perpetual funding rates and hedge costs are not yet modelled."),
      numbered("Liquidity: 128 subnet tokens with heterogeneous liquidity profiles. Realistic transaction costs could materially reduce Sharpe ratios, particularly for smaller subnets."),
      numbered("N/T ratio: with 128 assets and a 252-day window, N/T \u2248 0.51. The sample covariance matrix is borderline singular. Ledoit-Wolf shrinkage produces identical results (eigenstructure dominated by a few factors), but a formal regularisation study is warranted."),
      numbered("Regime stability: all OOS observations fall in bull or sideways regimes. Strategy behaviour in a sustained TAO bear market is entirely untested."),
      numbered("Market impact: equal-weight allocation to 128 tokens implies \u223C0.78% per token. At $1M deployment this is \u223C$7,800 per subnet. For illiquid subnets this may represent multiple days of average volume."),

      // disclaimer
      para(" ", { before: 120, after: 0 }),
      divider(),
      para("This memo is for internal use only and does not constitute investment advice. All projections and backtested performance figures involve model assumptions and uncertainty. Past performance is not indicative of future results.", {
        italic: true, size: 18, color: GRAY, align: AlignmentType.CENTER, before: 100, after: 60
      }),
      para("Quantitative Research Desk \u2022 White Star Capital \u2022 March 2026", {
        bold: true, size: 18, color: NAVY, align: AlignmentType.CENTER, before: 0, after: 0
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(
    "/Users/ale/Documents/quantitative-research/risk_premium_pca/TAO_Subnet_Portfolio_Memo.docx",
    buffer
  );
  console.log("Memo written successfully.");
});
