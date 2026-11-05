#!/usr/bin/env python3
"""Build the README charts in docs/assets/ from the benchmark data in this repository. Standard library only."""
from __future__ import annotations

import base64
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/assets"
FONT = "Inter, 'SF Pro Display', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
INK, SOFT, MUTED, FAINT, GRID = "#F4F5FF", "#C9CEEA", "#959CC4", "#5F6790", "#1A2149"
CORAL, VIOLET, BLUE, CYAN = "#F7936F", "#B86CEC", "#5E8FF0", "#6FD3F7"
SLATE, SLATE_HI = "#2B3360", "#3D477D"

DEFS = """<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="0.35" y2="1"><stop offset="0" stop-color="#0A1033"/><stop offset="1" stop-color="#03050F"/></linearGradient>
  <radialGradient id="nebula" cx="0.88" cy="0.0" r="0.75"><stop offset="0" stop-color="#8A55E6" stop-opacity="0.30"/><stop offset="0.55" stop-color="#4B3BB8" stop-opacity="0.07"/><stop offset="1" stop-color="#4B3BB8" stop-opacity="0"/></radialGradient>
  <radialGradient id="ember" cx="0.0" cy="1.0" r="0.65"><stop offset="0" stop-color="#F7936F" stop-opacity="0.15"/><stop offset="1" stop-color="#F7936F" stop-opacity="0"/></radialGradient>
  <linearGradient id="fold" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#F7936F"/><stop offset="0.5" stop-color="#B86CEC"/><stop offset="1" stop-color="#5E8FF0"/></linearGradient>
  <linearGradient id="foldv" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#5E8FF0"/><stop offset="0.55" stop-color="#B86CEC"/><stop offset="1" stop-color="#F7936F"/></linearGradient>
  <linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#B86CEC" stop-opacity="0.30"/><stop offset="1" stop-color="#5E8FF0" stop-opacity="0"/></linearGradient>
  <linearGradient id="meshfade" x1="1" y1="0" x2="0.35" y2="0.8"><stop offset="0" stop-color="#FFFFFF" stop-opacity="1"/><stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient>
  <mask id="meshmask"><rect width="100%" height="100%" fill="url(#meshfade)"/></mask>
  <filter id="glow" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="7"/></filter>
  <filter id="halo" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3.5"/></filter>
</defs>"""


def mesh(w, h, seed):
    """A faint triangulated mesh in the top-right corner: the TensorFold whale's skin, as texture."""
    rng = random.Random(seed)
    cols, rows, cw, rh = 9, 5, 64, 58
    x0 = w - cols * cw + 20
    pts = [[(x0 + c * cw + rng.uniform(-18, 18), -10 + r * rh + rng.uniform(-14, 14)) for c in range(cols + 1)]
           for r in range(rows + 1)]
    lines, nodes = [], []
    for r in range(rows + 1):
        for c in range(cols + 1):
            x, y = pts[r][c]
            for dr, dc in ((0, 1), (1, 0), (1, 1)):
                if r + dr <= rows and c + dc <= cols:
                    a, b = pts[r + dr][c + dc]
                    lines.append(f"M{x:.0f} {y:.0f}L{a:.0f} {b:.0f}")
            nodes.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{rng.choice((1.1, 1.4, 2.0))}"/>')
    return (f'<g mask="url(#meshmask)" opacity="0.5"><path d="{" ".join(lines)}" stroke="#7F8CE8" stroke-opacity="0.22" '
            f'stroke-width="0.8" fill="none"/><g fill="#C9B6F2" fill-opacity="0.55">{"".join(nodes)}</g></g>')


def card(w, h, body, kicker, title, subtitle, stat, stat_label, seed):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
{DEFS}
<rect width="{w}" height="{h}" rx="22" fill="url(#bg)"/>
<rect width="{w}" height="{h}" rx="22" fill="url(#nebula)"/>
<rect width="{w}" height="{h}" rx="22" fill="url(#ember)"/>
{mesh(w, h, seed)}
<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="22" fill="none" stroke="#2A3366"/>
<rect x="44" y="34" width="28" height="3" rx="1.5" fill="url(#fold)"/>
<text x="80" y="38" fill="{VIOLET}" font-size="12" font-weight="700" letter-spacing="2">{kicker}</text>
<text x="44" y="74" fill="{INK}" font-size="28" font-weight="800" letter-spacing="-0.6">{title}</text>
<text x="44" y="100" fill="{MUTED}" font-size="14">{subtitle}</text>
<text x="{w - 44}" y="78" fill="url(#fold)" font-size="50" font-weight="800" letter-spacing="-1.5" text-anchor="end">{stat}</text>
<text x="{w - 44}" y="100" fill="{SOFT}" font-size="11.5" font-weight="700" letter-spacing="1.6" text-anchor="end">{stat_label}</text>
{body}
</svg>
"""


def text(x, y, s, size: float = 13, fill=MUTED, anchor="start", weight=400, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{extra}>{s}</text>'


def bar(x, y, w, h, fill, r=6, extra=""):
    """Bar with rounded data end only; the baseline end stays square."""
    if w <= 0 or h <= 0:
        return ""
    r = min(r, w / 2, h)
    return (f'<path d="M{x:.1f} {y + h:.1f}V{y + r:.1f}Q{x:.1f} {y:.1f} {x + r:.1f} {y:.1f}H{x + w - r:.1f}'
            f'Q{x + w:.1f} {y:.1f} {x + w:.1f} {y + r:.1f}V{y + h:.1f}Z" fill="{fill}"{extra}/>')


def hbar(x, y, w, h, fill, r=6, extra=""):
    if w <= 0 or h <= 0:
        return ""
    r = min(r, h / 2, w)
    return (f'<path d="M{x:.1f} {y:.1f}H{x + w - r:.1f}Q{x + w:.1f} {y:.1f} {x + w:.1f} {y + r:.1f}V{y + h - r:.1f}'
            f'Q{x + w:.1f} {y + h:.1f} {x + w - r:.1f} {y + h:.1f}H{x:.1f}Z" fill="{fill}"{extra}/>')


def chip(x, y, s, fill=CORAL):
    w = 9 + 7.6 * len(s)
    return (f'<rect x="{x - w / 2:.1f}" y="{y - 15:.1f}" width="{w:.1f}" height="21" rx="10.5" fill="{fill}" fill-opacity="0.16" '
            f'stroke="{fill}" stroke-opacity="0.55"/>' + text(x, y, s, 12.5, fill, "middle", 800))


def panel(x0, x1, y0, y1, ticks, top, sx, xticks, title, xlabel, series, end_fmt, tick_fmt=str, bottom=0):
    """One line chart: recessive grid, 2.5px lines with halo, markers, nudged end-value labels."""
    sy = lambda v: y0 - (y0 - y1) * (v - bottom) / (top - bottom)
    p = [text(x0, y1 - 26, title, 14, SOFT, weight=700)]
    for v in ticks:
        p.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
        p.append(text(x0 - 10, sy(v) + 4, tick_fmt(v), 11.5, FAINT, "end"))
    for x, label in xticks:
        p.append(text(sx(x), y0 + 22, label, 12, MUTED, "middle", 600))
    p.append(text((x0 + x1) / 2, y0 + 44, xlabel, 12, FAINT, "middle"))
    p.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="#2A3366"/>')
    ends = []
    for name, color, dash, pts, hero in series:
        line = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in pts)
        dash_attr = ' stroke-dasharray="7 5"' if dash else ""
        if hero:
            p.append(f'<polygon points="{sx(pts[0][0]):.1f},{y0} {line} {sx(pts[-1][0]):.1f},{y0}" fill="url(#fade)"/>')
        p.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="7" opacity="0.3" filter="url(#halo)"{dash_attr}/>')
        p.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')
        for x, v in pts:
            p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="4" fill="#060A1E" stroke="{color}" stroke-width="2"/>')
        ends.append([sy(pts[-1][1]), pts[-1][1], color, sx(pts[-1][0])])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 17)
    shift = max(0, ends[-1][0] - (y0 - 8))
    for y, v, color, x in ends:
        y -= shift
        if not end_fmt(v):
            continue
        p.append(f'<line x1="{x + 9:.1f}" x2="{x + 17:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        p.append(text(x + 22, y + 5, end_fmt(v), 14, INK, weight=800))
    return p


def legend(x, y, items):
    p = []
    for name, color, dash in items:
        dash_attr = ' stroke-dasharray="7 5"' if dash else ""
        p.append(f'<line x1="{x}" x2="{x + 26}" y1="{y - 4}" y2="{y - 4}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        p.append(text(x + 34, y, name, 12.5, SOFT, weight=600))
        x += 52 + 7.2 * len(name)
    return p


def prefill(summary):
    rows = summary["prefill"]
    w, h, y0, y1 = 1000, 510, 432, 176
    xt = [(i, f"{r['targetTokens'] // 1024}K") for i, r in enumerate(rows)]
    p = []
    a0, a1 = 92, 470
    sa = lambda i: a0 + (a1 - a0) * i / (len(rows) - 1)
    p += panel(a0, a1, y0, y1, (3500, 3600, 3700, 3800, 3900), 3950, sa, xt, "Prefill tok/s (axis starts at 3,400)", "Prompt length, tokens",
               [("Prefill", CORAL, False, [(i, r["prefillTps"]) for i, r in enumerate(rows)], False)],
               lambda v: f"{v:,.0f}", lambda v: f"{v:,}", bottom=3400)
    peak = max(rows, key=lambda r: r["prefillTps"])
    sy = lambda v: y0 - (y0 - y1) * (v - 3400) / 550
    for i, r in enumerate(rows[:-1]):
        top = r is peak
        p.append(text(sa(i) + (12 if i == 0 else 0), sy(r["prefillTps"]) + (24 if i == 0 else -13), f"{r['prefillTps']:,.0f}", 14 if top else 12, INK if top else SOFT, "start" if i == 0 else "middle", 800 if top else 700))
    b0, b1 = 590, 900
    sb = lambda i: b0 + (b1 - b0) * i / (len(rows) - 1)
    p += panel(b0, b1, y0, y1, (10, 20, 30, 40), 40, sb, xt, "Time to first token, seconds", "Prompt length, tokens",
               [("TTFT", BLUE, False, [(i, r["ttftMs"] / 1000) for i, r in enumerate(rows)], False)],
               lambda v: f"{v:.1f} s", lambda v: f"{v} s")
    st = lambda v: y0 - (y0 - y1) * v / 40
    for i, r in enumerate(rows[:-1]):
        s = r["ttftMs"] / 1000
        p.append(text(sb(i) - 9, st(s) - 9, f"{s:.1f}", 12, SOFT, "end", 700))
    return card(w, h, "\n".join(p), "PREFILL", f"{rows[0]['prefillTps']:,.0f}–{peak['prefillTps']:,.0f} tok/s from 4K to 128K tokens",
                f"Cold prompts, one request at a time; a 128K prompt reaches its first token in {rows[-1]['ttftMs'] / 1000:.0f} s",
                f"{peak['prefillTps']:,.0f}", f"TOK/S PEAK · {peak['targetTokens'] // 1024}K PROMPT", 11)


CLIMB = [("Port flags", "off", 3011), ("Read-ahead,", "ragged, masked", 3142), ("All 1.2.8 flags,", "4K rows", 3471),
         ("6K logical", "rows", 3525), ("8K logical", "rows", 3581), ("Shared-range kernel", "+ aliasing", 3685),
         ("Exact shared", "split", 3720), ("v1.1", "release", 3833)]


def climb():
    w, h, x0, x1, y0, y1 = 1000, 520, 96, 956, 400, 160
    lo, hi = 2900, 3900
    sy = lambda v: y0 - (y0 - y1) * (v - lo) / (hi - lo)
    p = []
    for v in (3000, 3250, 3500, 3750):
        p.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}"/>')
        p.append(text(x0 - 14, sy(v) + 4, f"{v:,}", 12, FAINT, "end"))
    slot = (x1 - x0) / len(CLIMB)
    bw = slot * 0.56
    prev = None
    for i, (a, b, v) in enumerate(CLIMB):
        cx = x0 + (i + 0.5) * slot
        bx = cx - bw / 2
        first, last = i == 0, i == len(CLIMB) - 1
        if first or last:
            fill = "url(#foldv)" if last else SLATE_HI
            if last:
                p.append(bar(bx, sy(v), bw, y0 - sy(v), fill, extra=' opacity="0.6" filter="url(#glow)"'))
            p.append(bar(bx, sy(v), bw, y0 - sy(v), fill))
            p.append(text(cx, sy(v) - 14, f"{v:,}", 22 if last else 16, INK, "middle", 800))
        else:
            p.append(f'<rect x="{bx:.1f}" y="{sy(v):.1f}" width="{bw:.1f}" height="{sy(prev) - sy(v):.1f}" rx="3" fill="url(#fold)"/>')
            p.append(f'<line x1="{bx - slot + bw:.1f}" x2="{bx:.1f}" y1="{sy(prev):.1f}" y2="{sy(prev):.1f}" stroke="{MUTED}" stroke-dasharray="2 3"/>')
            p.append(text(cx, sy(v) - 26, f"+{v - prev}", 14, CORAL, "middle", 800))
            p.append(text(cx, sy(v) - 10, f"{v:,}", 12, MUTED, "middle", 600))
        p.append(text(cx, y0 + 26, a, 12, INK if last else SOFT, "middle", 700 if last else 600))
        p.append(text(cx, y0 + 42, b, 12, INK if last else FAINT, "middle", 600 if last else 400))
        prev = v
    p.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="#2A3366"/>')
    p.append(text(x0 - 14, y0 + 4, "2,900", 12, FAINT, "end"))
    p.append(text(x0, y0 + 78, "Axis starts at 2,900 tok/s. Every step matched the reference replies (8 long, 12 short greedy checks).", 12, FAINT))
    gain = (CLIMB[-1][2] / CLIMB[0][2] - 1) * 100
    return card(w, h, "\n".join(p), "PREFILL PORT", "Seven steps, +822 tok/s at 32K",
                "How the LakyFx 1.2.8 prefill port reached three ranks, step by step on the same weights",
                f"+{gain:.0f}%", "32K PREFILL", 23)


DECODE = [("prose", "Prose", "#D66F4E"), ("code", "Code", "#358FBF")]
LABELED = (1, 2, 4, 8, 12, 16)


def decode(summary):
    conc = [1, 2, 4, 6, 8, 10, 12, 16]
    w, h, y0, y1, top = 1000, 520, 432, 196, 300
    agg = lambda mode, prompt: {r["concurrency"]: r["aggregateDecodeTps"] for r in summary["decode"]
                                if r["mode"] == mode and r["promptType"] == prompt}
    sy = lambda v: y0 - (y0 - y1) * v / top
    p = legend(44, 140, [(name, color, False) for _, name, color in DECODE])
    for mode, title, x0 in (("greedy", "Greedy · temperature 0, thinking off", 84),
                            ("think", "Thinking · temperature 0.7, thinking on", 560)):
        x1 = x0 + 370
        sx = lambda c, x0=x0, x1=x1: x0 + 22 + (x1 - x0 - 22) * (c - 1) / 15
        series = {prompt: agg(mode, prompt) for prompt, _, _ in DECODE}
        p += panel(x0, x1, y0, y1, (50, 100, 150, 200, 250, 300), top, sx, [(c, f"C{c}") for c in conc], title,
                   "Requests at once", [(n, col, False, [(c, series[pr][c]) for c in conc], False) for pr, n, col in DECODE],
                   lambda v: "", bottom=0)
        for c in LABELED:
            (hi, hv), (lo, lv) = sorted(((pr, series[pr][c]) for pr, _, _ in DECODE), key=lambda t: -t[1])
            color = {pr: col for pr, _, col in DECODE}
            p.append(text(sx(c), sy(hv) - 11, f"{hv:.0f}", 12.5, color[hi], "middle", 800))
            p.append(text(sx(c), sy(lv) + 21, f"{lv:.0f}", 12.5, color[lo], "middle", 800))
    single = agg("greedy", "code")[1]
    sixteen = agg("greedy", "prose")[16]
    return card(w, h, "\n".join(p), "DECODE", f"Up to {single:.0f} tok/s for one request, {sixteen:.0f} across sixteen",
                "All requests combined, 400 output tokens per request", f"{sixteen:.0f}", "TOK/S · 16 STREAMS", 37)


NAMES = {"This release": ("This release", "3.84 bpw mix + mrfakename edit"),
         "LakyFx EXL3, no edit": ("LakyFx 3.5 bpw", "no edit, not abliterated"),
         "LakyFx EXL3 + mrfakename": ("LakyFx 3.5 bpw", "+ mrfakename edit"),
         "LakyFx EXL3 + pqhaz heretic": ("LakyFx 3.5 bpw", "+ heretic edit"),
         "This composition + Keys": ("Our 3.84 bpw mix", "+ Keys overlay"),
         "LakyFx EXL3 + Keys": ("LakyFx 3.5 bpw", "+ Keys overlay"),
         "LibertAIDAI EXL3 + Keys": ("LibertAIDAI 3.5 bpw", "+ Keys overlay"),
         "LakyFx EXL3 + Mia Keys K5": ("LakyFx 3.5 bpw", "+ Mia's Keys K5")}


def kl():
    with open(ROOT / "benchmarks/v1/fidelity.csv", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["mean_kl"]))
    pitch = 50
    w, lx, x0, x1, y = 1000, 300, 322, 900, 172
    h = y + pitch * len(rows) + 70
    top = 0.14
    sx = lambda v: x0 + (x1 - x0) * v / top
    base = next(float(r["mean_kl"]) for r in rows if r["abliterated"] == "no")
    p = []
    for v in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14):
        p.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{y - 16}" y2="{y + pitch * len(rows) - 14}" stroke="{GRID}"/>')
        p.append(text(sx(v), y + pitch * len(rows) + 8, f"{v:.2f}", 11.5, FAINT, "middle"))
    p.append(text(x0, y + pitch * len(rows) + 30, "Mean KL to official logits · lower is closer", 12, FAINT))
    bx = sx(base)
    p.append(f'<line x1="{bx:.1f}" x2="{bx:.1f}" y1="{y - 34}" y2="{y + pitch * len(rows) - 14}" stroke="{SOFT}" stroke-dasharray="4 4"/>')
    p.append(text(bx + 8, y - 24, "unedited quant", 12, SOFT, weight=700))
    for r in rows:
        ours = r["weights"] == "This release"
        plain = r["abliterated"] == "no"
        v = float(r["mean_kl"])
        name, sub = NAMES[r["weights"]]
        cy = y + 10
        if ours:
            p.append(f'<rect x="20" y="{y - 13}" width="{w - 40}" height="{pitch - 8}" rx="12" fill="#B86CEC" fill-opacity="0.08" stroke="#B86CEC" stroke-opacity="0.25"/>')
        p.append(text(lx, cy - 1, name, 15 if ours else 14, INK if ours else SOFT, "end", 800 if ours else 600))
        p.append(text(lx, cy + 15, sub, 11.5, CORAL if ours else FAINT, "end", 600 if ours else 400))
        color = "url(#fold)" if ours else (SLATE_HI if plain else SLATE)
        if ours:
            p.append(hbar(x0, cy - 6, sx(v) - x0, 12, color, extra=' opacity="0.7" filter="url(#glow)"'))
        p.append(hbar(x0, cy - 5 if ours else cy - 3, sx(v) - x0, 10 if ours else 6, color, 5 if ours else 3))
        p.append(f'<circle cx="{sx(v):.1f}" cy="{cy:.1f}" r="{8 if ours else 5}" fill="{"#FFFFFF" if ours else ("#060A1E" if not plain else SOFT)}" '
                 f'stroke="{VIOLET if ours else MUTED}" stroke-width="2.5"/>')
        p.append(text(sx(v) + 16, cy + 5, f"{v:.4f}", 18 if ours else 13.5, INK if ours else MUTED, "start", 800 if ours else 600))
        y += pitch
    return card(w, h, "\n".join(p), "FIDELITY", "Abliterated, and still closest to the official model",
                "The only abliterated pack below an unedited quant · 65 × 2,048-token panel, 133,120 positions",
                "0.0784", "MEAN KL · LOWEST SCORED", 53)


def quality_rows():
    with open(ROOT / "benchmarks/v1.1/quality.csv", newline="") as f:
        return {(r["suite"], r["metric"]): r for r in csv.DictReader(f)}


def quality():
    q = quality_rows()
    w, h = 1000, 520
    p = []
    x0, x1, y = 44, 560, 176
    sx = lambda v: x0 + 150 + (x1 - x0 - 150) * v / 100
    p.append(text(x0, 146, "Task scores", 14, SOFT, weight=700))
    bars = [("GPQA-Diamond", q[("GPQA-Diamond", "mean pass@1")], "198 × 8, effort 50", "DeepSeek, max effort"),
            ("AIME 2025", q[("AIME 2025", "mean pass@1")], "30 × 4, effort 50", ""),
            ("IFEval", q[("IFEval", "prompt-level strict")], "541 prompts, strict", "Official API, same prompts"),
            ("LiveCodeBench v6", q[("LiveCodeBench v6", "pass@1")], "175 problems", ""),
            ("MMLU-Pro", q[("MMLU-Pro", "pass@1")], "500 stratified, thinking on", "")]
    for name, r, sub, ref_label in bars:
        v = float(r["score_percent"])
        p.append(text(x0, y + 4, name, 15, INK, weight=700))
        p.append(text(x0, y + 21, sub, 11.5, FAINT))
        p.append(hbar(sx(0), y - 10, sx(100) - sx(0), 26, "#141B40", 8))
        p.append(hbar(sx(0), y - 10, sx(v) - sx(0), 26, "url(#fold)", 8, ' opacity="0.55" filter="url(#glow)"'))
        p.append(hbar(sx(0), y - 10, sx(v) - sx(0), 26, "url(#fold)", 8))
        p.append(text(sx(v) - 10, y + 9, f"{v:.1f}%", 15, "#0A0F2A", "end", 800))
        if r["reference_percent"]:
            ref = float(r["reference_percent"])
            p.append(f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{y - 18}" y2="{y + 24}" stroke="{INK}" stroke-width="2"/>')
            p.append(text(sx(ref) + 4, y + 36, f"{ref_label}: {ref:.1f}%", 11.5, SOFT, "end", 600))
        y += 68
    p.append(f'<line x1="610" x2="610" y1="132" y2="{h - 40}" stroke="{GRID}"/>')
    p.append(text(646, 146, "Refusals", 14, SOFT, weight=700))
    refused = q[("HarmBench", "complied (official cais/HarmBench-Llama-2-13b-cls classifier)")]
    xs_full = q[("XSTest safe", "full refusal")]
    xs_part = q[("XSTest safe", "partial refusal")]
    with open(ROOT / "benchmarks/v1/refusal.csv", newline="") as f:
        chat = next(r for r in csv.DictReader(f) if r["weights"] == "This release" and r["subset"] == "HarmBench harmful" and r["mode"] == "chat")
    tiles = [(chat["refused"], f"of {chat['prompts']}", "harmful prompts refused"),
             (refused["numerator"], f"of {refused['denominator']}", "complied, per HarmBench's own classifier"),
             (xs_full["numerator"], f"of {xs_full['denominator']}", f"safe prompts refused ({xs_part['numerator']} partial)")]
    ty = 214
    for big, of, label in tiles:
        p.append(text(646, ty, big, 46, "url(#fold)", weight=800, extra=' letter-spacing="-1.5"'))
        p.append(text(646 + 27 * len(big) + 10, ty - 4, of, 15, SOFT, weight=700))
        p.append(text(646, ty + 24, label, 12.5, MUTED))
        ty += 104
    return card(w, h, "\n".join(p), "QUALITY", "Refusals removed, scores stay high",
                "Thinking evals at effort 50, temperature 1; IFEval and refusal screens greedy through the API",
                "90.5%", "GPQA-DIAMOND · 1,584 ANSWERS", 71)


EMBER, ROSE, LILAC, PEACH = "#FFA24C", "#FF5C8A", "#C77DF0", "#F3C9A8"
CODE, PROSE = "#D47834", "#AE70D5"
W_INK, W_SOFT, W_MUTED, W_LINE, W_EDGE = "#FBF3EE", "#D8CCD2", "#A4939F", "#2E1A26", "#3A1E2E"
LOADS = (1, 2, 4, 8, 12, 16)
TABULAR = ' style="font-variant-numeric:tabular-nums"'
SCORE_DEFS = f"""<defs>
  <linearGradient id="sbg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0A0406"/><stop offset="1" stop-color="#0C0610"/></linearGradient>
  <linearGradient id="heat" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{EMBER}"/><stop offset="0.55" stop-color="{ROSE}"/><stop offset="1" stop-color="{LILAC}"/></linearGradient>
  <linearGradient id="spanel" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1A0E15"/><stop offset="1" stop-color="#110913"/></linearGradient>
  <linearGradient id="srow" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{EMBER}" stop-opacity="0.20"/><stop offset="1" stop-color="{LILAC}" stop-opacity="0.14"/></linearGradient>
  <linearGradient id="sink" gradientUnits="userSpaceOnUse" x1="0" y1="270" x2="0" y2="362"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>
  <filter id="feather" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="13"/></filter>
  <filter id="numglow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="16"/></filter>
  <filter id="lineglow" x="-10%" y="-40%" width="120%" height="180%"><feGaussianBlur stdDeviation="5"/></filter>
  <mask id="headfade" maskUnits="userSpaceOnUse" x="0" y="0" width="1400" height="430"><path d="M790 -30 Q700 42 622 86 Q600 118 610 150 Q628 190 700 232 L860 330 L860 460 L1440 460 L1440 -30 Z" fill="#fff" filter="url(#feather)"/></mask>
  <mask id="basefade" maskUnits="userSpaceOnUse" x="0" y="0" width="1400" height="430"><rect width="1400" height="430" fill="url(#sink)"/></mask>
  <clipPath id="notitle"><path d="M0 0H1117V340H1000V600H0Z"/></clipPath>
</defs>"""


def tps(v):
    """One decimal, halves rounded up, so the card agrees with the benchmark tables."""
    return f"{int(v * 10 + 0.5 + 1e-9) / 10:.1f}"


def decode_panel(x, y, title, tag, rates):
    w = 660
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="704" rx="20" fill="url(#spanel)" stroke="{W_EDGE}" stroke-width="1.5"/>',
         text(x + 30, y + 48, title, 29, W_INK, weight=800, extra=' letter-spacing="-0.4"'), text(x + 30, y + 76, tag, 17, PEACH, weight=600)]
    for dx, name, fill in ((470, "Code", CODE), (556, "Prose", PROSE)):
        p.append(f'<rect x="{x + dx}" y="{y + 32}" width="16" height="16" rx="4" fill="{fill}"/>' + text(x + dx + 24, y + 46, name, 17, W_SOFT, weight=700))
    x0, x1, base, top = x + 30, x + w - 30, y + 328, y + 112
    px = lambda i: x0 + 26 + (x1 - x0 - 78) * i / (len(LOADS) - 1)
    py = lambda v: base - (base - top) * v / 300
    for tick in (100, 200, 300):
        p.append(f'<line x1="{x0}" x2="{x1}" y1="{py(tick):.1f}" y2="{py(tick):.1f}" stroke="{W_LINE}"/>' + text(x1, py(tick) - 6, tick, 14, W_MUTED, "end", 600))
    for kind, fill in (("prose", PROSE), ("code", CODE)):
        points = " ".join(f"{px(i):.1f},{py(rates[kind, load]):.1f}" for i, load in enumerate(LOADS))
        p.append(f'<polyline points="{points}" fill="none" stroke="{fill}" stroke-width="9" opacity="0.4" filter="url(#lineglow)"/>')
        p.append(f'<polyline points="{points}" fill="none" stroke="{fill}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        for i, load in enumerate(LOADS):
            p.append(f'<circle cx="{px(i):.1f}" cy="{py(rates[kind, load]):.1f}" r="6.5" fill="#150B13" stroke="{fill}" stroke-width="3.5"/>')
    for i, load in enumerate(LOADS):
        p.append(text(px(i), base + 25, f"C{load}", 16, W_MUTED, "middle", 700))
    p.append(f'<line x1="{x0}" x2="{x1}" y1="{base}" y2="{base}" stroke="#3A2432"/>')
    t0, t1, ty, row = x + 18, x + w - 18, y + 392, 46
    code_x = t0 + (t1 - t0) * 0.62
    for cx, name, anchor in ((t0 + 18, "LOAD", "start"), (code_x, "CODE", "end"), (t1 - 18, "PROSE", "end")):
        p.append(text(cx, ty, name, 15, W_MUTED, anchor, 700, ' letter-spacing="1.6"'))
    ty += 14
    for load in LOADS:
        if load == LOADS[-1]:
            p.append(f'<rect x="{t0}" y="{ty}" width="{t1 - t0}" height="{row}" rx="10" fill="url(#srow)" stroke="{EMBER}" stroke-opacity="0.45"/>')
        else:
            p.append(f'<line x1="{t0 + 18}" x2="{t1 - 18}" y1="{ty + row}" y2="{ty + row}" stroke="{W_LINE}"/>')
        b = ty + row * 0.70
        p.append(text(t0 + 18, b, f"C{load}", 23, W_INK if load == LOADS[-1] else W_SOFT, weight=700))
        p.append(text(code_x, b, tps(rates["code", load]), 29, W_INK, "end", 800, TABULAR))
        p.append(text(t1 - 18, b, tps(rates["prose", load]), 29, W_INK, "end", 800, TABULAR))
        ty += row
    return "".join(p)


def scorecard(summary):
    """Every speed on one card; the README opens with it. Quality has its own section."""
    w, h = 1400, 1158
    art = base64.b64encode((OUT / "hero.jpg").read_bytes()).decode()
    pre = summary["prefill"]
    peak = max(r["prefillTps"] for r in pre)
    rates = {(r["mode"], r["promptType"], r["concurrency"]): r["aggregateDecodeTps"] for r in summary["decode"]}
    # The hero art without its baked-in title and stats, mirrored so the whale swims toward the headline.
    p = ['<g mask="url(#headfade)"><g mask="url(#basefade)"><g transform="translate(1436 8) scale(-0.75 0.75)"><g clip-path="url(#notitle)">'
         f'<image href="data:image/jpeg;base64,{art}" width="1536" height="768"/></g></g></g></g>',
         text(60, 64, "3× DGX SPARK  ·  TP3  ·  TENSORFOLD", 21, PEACH, weight=700, extra=' letter-spacing="2.6"'),
         text(58, 128, "DeepSeek V4.1 Flash", 54, W_INK, weight=800, extra=' letter-spacing="-1.4"'),
         f'<rect x="60" y="150" width="362" height="64" rx="32" fill="{EMBER}" fill-opacity="0.12" stroke="{EMBER}" stroke-opacity="0.7" stroke-width="1.5"/>',
         text(241, 194, "ABLITERATED", 31, W_INK, "middle", 800, ' letter-spacing="4"'),
         f'<rect x="440" y="150" width="316" height="64" rx="32" fill="{LILAC}" fill-opacity="0.12" stroke="{LILAC}" stroke-opacity="0.7" stroke-width="1.5"/>',
         text(598, 194, "3.84 BPW EXL3", 29, W_INK, "middle", 800, ' letter-spacing="2"'),
         f'<text x="58" y="340" fill="url(#heat)" font-size="150" font-weight="800" letter-spacing="-6" opacity="0.5" filter="url(#numglow)">{peak:,.0f}</text>',
         text(58, 340, f"{peak:,.0f}", 150, "url(#heat)", weight=800, extra=' letter-spacing="-6"'),
         text(432, 306, "tok/s prefill", 44, W_INK, weight=800, extra=' letter-spacing="-0.8"')]
    for x, title, tag, mode in ((30, "Thinking off", "Greedy · temperature 0", "greedy"), (710, "Thinking on", "Non-greedy · temperature 0.7", "think")):
        p.append(decode_panel(x, 380, title, tag, {(k, c): v for (m, k, c), v in rates.items() if m == mode}))
    p.append(text(700, 1122, "Combined tok/s across all requests · 400 output tokens each · engine 678ea1b · measured 4–5 November 2026",
                  17, W_MUTED, "middle"))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
{SCORE_DEFS}
<rect width="{w}" height="{h}" rx="26" fill="url(#sbg)"/>
{chr(10).join(p)}
<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="25" fill="none" stroke="{W_EDGE}" stroke-width="2"/>
</svg>
"""


def build():
    summary = json.loads((ROOT / "benchmarks/v1.2/summary.json").read_text())
    return {"chart-prefill.svg": prefill(summary), "chart-prefill-climb.svg": climb(),
            "chart-decode.svg": decode(summary), "chart-kl.svg": kl(), "chart-quality.svg": quality(),
            "scorecard.svg": scorecard(summary)}


def main():
    for name, svg in build().items():
        (OUT / name).write_text(svg)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
