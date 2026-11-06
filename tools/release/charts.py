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
INK, SOFT, MUTED, FAINT, GRID = "#FFF4E6", "#E9D8C6", "#CDB7A2", "#A98F7A", "#4F3627"
GOLD, EMBER, ROSE, CREAM = "#FFC83D", "#FF7A1A", "#FF4F7B", "#FFF1E0"
CORAL, VIOLET, BLUE = EMBER, ROSE, GOLD
SLATE, SLATE_HI = "#4A3325", "#6B4A35"
AXIS, DOT = "#6B4A35", "#120804"

DEFS = f"""<defs>
  <linearGradient id="fold" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{GOLD}"/><stop offset="0.55" stop-color="{EMBER}"/><stop offset="1" stop-color="{ROSE}"/></linearGradient>
  <linearGradient id="foldv" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="{ROSE}"/><stop offset="0.55" stop-color="{EMBER}"/><stop offset="1" stop-color="{GOLD}"/></linearGradient>
  <linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{EMBER}" stop-opacity="0.30"/><stop offset="1" stop-color="{EMBER}" stop-opacity="0"/></linearGradient>
  <filter id="glow" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="7"/></filter>
  <filter id="halo" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="3.5"/></filter>
</defs>"""


def card(w, h, body, kicker, title, subtitle, stat, stat_label):
    art = base64.b64encode((OUT / "chart-texture.jpg").read_bytes()).decode()
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
{DEFS}
<clipPath id="round"><rect width="{w}" height="{h}" rx="22"/></clipPath>
<g clip-path="url(#round)"><image href="data:image/jpeg;base64,{art}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice"/><rect width="{w}" height="{h}" fill="#0A0503" fill-opacity="0.32"/></g>
<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="22" fill="none" stroke="{EMBER}" stroke-opacity="0.55"/>
<rect x="44" y="34" width="28" height="3" rx="1.5" fill="url(#fold)"/>
<text x="80" y="38" fill="{GOLD}" font-size="12" font-weight="700" letter-spacing="2">{kicker}</text>
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
    p.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="{AXIS}"/>')
    ends = []
    for name, color, dash, pts, hero in series:
        line = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in pts)
        dash_attr = ' stroke-dasharray="7 5"' if dash else ""
        if hero:
            p.append(f'<polygon points="{sx(pts[0][0]):.1f},{y0} {line} {sx(pts[-1][0]):.1f},{y0}" fill="url(#fade)"/>')
        p.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="7" opacity="0.3" filter="url(#halo)"{dash_attr}/>')
        p.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')
        for x, v in pts:
            p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="4" fill="{DOT}" stroke="{color}" stroke-width="2"/>')
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
                f"{peak['prefillTps']:,.0f}", f"TOK/S PEAK · {peak['targetTokens'] // 1024}K PROMPT")


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
    p.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="{AXIS}"/>')
    p.append(text(x0 - 14, y0 + 4, "2,900", 12, FAINT, "end"))
    p.append(text(x0, y0 + 78, "Axis starts at 2,900 tok/s. Every step matched the reference replies (8 long, 12 short greedy checks).", 12, FAINT))
    gain = (CLIMB[-1][2] / CLIMB[0][2] - 1) * 100
    return card(w, h, "\n".join(p), "PREFILL PORT", "Seven steps, +822 tok/s at 32K",
                "How the LakyFx 1.2.8 prefill port reached three ranks, step by step on the same weights",
                f"+{gain:.0f}%", "32K PREFILL")


DECODE = [("prose", "Prose", EMBER), ("code", "Code", GOLD)]
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
                "All requests combined, 400 output tokens per request", f"{sixteen:.0f}", "TOK/S · 16 STREAMS")


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
            p.append(f'<rect x="20" y="{y - 13}" width="{w - 40}" height="{pitch - 8}" rx="12" fill="{EMBER}" fill-opacity="0.10" stroke="{EMBER}" stroke-opacity="0.35"/>')
        p.append(text(lx, cy - 1, name, 15 if ours else 14, INK if ours else SOFT, "end", 800 if ours else 600))
        p.append(text(lx, cy + 15, sub, 11.5, CORAL if ours else FAINT, "end", 600 if ours else 400))
        color = "url(#fold)" if ours else (SLATE_HI if plain else SLATE)
        if ours:
            p.append(hbar(x0, cy - 6, sx(v) - x0, 12, color, extra=' opacity="0.7" filter="url(#glow)"'))
        p.append(hbar(x0, cy - 5 if ours else cy - 3, sx(v) - x0, 10 if ours else 6, color, 5 if ours else 3))
        p.append(f'<circle cx="{sx(v):.1f}" cy="{cy:.1f}" r="{8 if ours else 5}" fill="{"#FFFFFF" if ours else (DOT if not plain else SOFT)}" '
                 f'stroke="{EMBER if ours else MUTED}" stroke-width="2.5"/>')
        p.append(text(sx(v) + 16, cy + 5, f"{v:.4f}", 18 if ours else 13.5, INK if ours else MUTED, "start", 800 if ours else 600))
        y += pitch
    return card(w, h, "\n".join(p), "FIDELITY", "Abliterated, and still closest to the official model",
                "The only abliterated pack below an unedited quant · 65 × 2,048-token panel, 133,120 positions",
                "0.0784", "MEAN KL · LOWEST SCORED")


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
        p.append(hbar(sx(0), y - 10, sx(100) - sx(0), 26, "#2A1C15", 8))
        p.append(hbar(sx(0), y - 10, sx(v) - sx(0), 26, "url(#fold)", 8, ' opacity="0.55" filter="url(#glow)"'))
        p.append(hbar(sx(0), y - 10, sx(v) - sx(0), 26, "url(#fold)", 8))
        p.append(text(sx(v) - 10, y + 9, f"{v:.1f}%", 15, "#1A0E08", "end", 800))
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
                "90.5%", "GPQA-DIAMOND · 1,584 ANSWERS")


TITLE_INK, LABEL_INK, RULE = "#F9B54C", "#E9D8C6", "#6B4A35"
LOADS = (1, 2, 4, 8, 12, 16)
GRID_LOADS = (1, 4, 8, 16)
SERIES = (("prose", "greedy", "Prose Greedy", EMBER), ("prose", "think", "Prose Thinking", ROSE),
          ("code", "greedy", "Code Greedy", GOLD), ("code", "think", "Code Thinking", CREAM))
SCORE_DEFS = f"""<defs>
  <linearGradient id="heat" gradientUnits="userSpaceOnUse" x1="70" y1="0" x2="1330" y2="0"><stop offset="0" stop-color="{GOLD}"/><stop offset="0.5" stop-color="{EMBER}"/><stop offset="1" stop-color="{ROSE}"/></linearGradient>
  <filter id="numglow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="9"/></filter>
  <clipPath id="corners"><rect width="1402" height="1066" rx="26"/></clipPath>
</defs>"""


def whole(v):
    """Decode speeds are shown as whole tok/s; halves round up."""
    return str(int(v + 0.5))


def quality_score(suite):
    row = next(r for (name, _), r in quality_rows().items() if name.startswith(suite))
    return f"{float(row['score_percent']):.1f}%"


def clock_rows():
    with open(ROOT / "benchmarks/v1.2/clocks.csv", newline="") as f:
        return list(csv.DictReader(f))


def heading(x, y, title, note=""):
    out = text(x, y, title, 30, TITLE_INK, weight=800, extra=' letter-spacing="1.6"')
    if note:
        out += text(x + 30 + 22 * len(title), y - 1, note, 22, LABEL_INK, weight=500)
    return out


def rule(x0, x1, y, opacity=0.8):
    return f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" stroke="{RULE}" stroke-opacity="{opacity}"/>'


def graph(xs, y_bottom, y_top, lo, hi, ticks, xlabels, lines, point_labels=False):
    """A line graph plotted from data: recessive grid, 3px lines, round markers."""
    sy = lambda v: y_bottom - (y_bottom - y_top) * (v - lo) / (hi - lo)
    p = []
    for t in ticks:
        p.append(f'<line x1="{xs[0] - 38}" x2="{xs[-1] + 38}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{CREAM}" stroke-opacity="{0.55 if t == lo else 0.16}"/>')
        p.append(text(xs[0] - 48, sy(t) + 8, t, 22, LABEL_INK, "end", 600))
    for x, label in zip(xs, xlabels):
        p.append(f'<line x1="{x}" x2="{x}" y1="{sy(hi):.1f}" y2="{y_bottom}" stroke="{CREAM}" stroke-opacity="0.10"/>')
        for i, part in enumerate(label.split("|")):
            p.append(text(x, y_bottom + 27 + 22 * i, part, 20 if "|" in label else 23, CREAM, "middle", 600))
    for values, color, above in lines:
        points = " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, values))
        p.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, v in zip(xs, values):
            p.append(f'<circle cx="{x:.1f}" cy="{sy(v):.1f}" r="6.5" fill="{color}" stroke="#120804" stroke-width="1.5"/>')
            if point_labels:
                p.append(text(x, sy(v) + (-14 if above else 30), f"{v:g}", 22, color, "middle", 800))
    return "".join(p)


def scorecard(summary):
    """The image both cards open with: whale, headline speeds, decode by load, clocks, temperature and quality."""
    w, h = 1402, 1066
    art = base64.b64encode((OUT / "card-art.jpg").read_bytes()).decode()
    rate = {(r["promptType"], r["mode"], r["concurrency"]): r["aggregateDecodeTps"] for r in summary["decode"]}
    clocks = clock_rows()
    peak = max(r["prefillTps"] for r in summary["prefill"])
    p = [f'<image href="data:image/jpeg;base64,{art}" width="{w}" height="{h}"/>']
    # Headline stats.
    stats = ((f"{peak:,.0f}", "TOK/S PREFILL AT 32K"), (whole(rate["prose", "greedy", 1]), "TOK/S PROSE DECODE"),
             (whole(rate["code", "greedy", 1]), "TOK/S CODE DECODE"), (whole(rate["prose", "greedy", 16]), "TOK/S AT 16 REQUESTS"),
             ("1M", "CONTEXT"))
    step = w / len(stats)
    for i, (value, label) in enumerate(stats):
        x = step * (i + 0.5)
        if i:
            p.append(f'<line x1="{x - step / 2:.1f}" x2="{x - step / 2:.1f}" y1="336" y2="416" stroke="{TITLE_INK}" stroke-opacity="0.55"/>')
        p.append(f'<text x="{x:.1f}" y="388" fill="url(#heat)" font-size="76" font-weight="800" letter-spacing="-1.5" text-anchor="middle" opacity="0.4" filter="url(#numglow)">{value}</text>')
        p.append(text(x, 388, value, 76, "url(#heat)", "middle", 800, ' letter-spacing="-1.5"'))
        p.append(text(x, 416, label, 19, CREAM, "middle", 700, ' letter-spacing="0.8"'))
    # Decode table.
    p.append(heading(40, 476, "DECODE", "combined tok/s"))
    col = (362, 446, 530, 610)
    for x, load in zip(col, GRID_LOADS):
        p.append(text(x, 510, f"C{load}", 23, LABEL_INK, "end", 700))
    p.append(rule(40, 610, 522))
    highlight = {("prose", "greedy", 1), ("prose", "greedy", 16), ("code", "greedy", 1)}
    y = 556
    for prompt, name in (("prose", "PROSE"), ("code", "CODE")):
        p.append(text(40, y + 22, name, 25, TITLE_INK, weight=800, extra=' letter-spacing="1"'))
        for mode, label in (("greedy", "Greedy"), ("think", "Thinking")):
            p.append(text(150, y, label, 27, CREAM, weight=600))
            for x, load in zip(col, GRID_LOADS):
                top = (prompt, mode, load) in highlight
                p.append(text(x, y, whole(rate[prompt, mode, load]), 30, GOLD if top else CREAM, "end", 800))
            y += 40
        if prompt == "prose":
            p.append(rule(40, 610, y - 25))
        y += 8
    # Decode graph.
    p.append(heading(684, 476, "DECODE", "tok/s"))
    xs = (780, 890, 1000, 1110)
    p.append(graph(xs, 674, 500, 0, 300, (0, 100, 200, 300), [f"C{n}" for n in GRID_LOADS],
                   [([rate[prompt, mode, n] for n in GRID_LOADS], color, True) for prompt, mode, _, color in SERIES]))
    for i, (_, _, label, color) in enumerate(SERIES):
        y = 534 + 40 * i
        p.append(f'<line x1="1166" x2="1194" y1="{y - 8}" y2="{y - 8}" stroke="{color}" stroke-width="3"/><circle cx="1180" cy="{y - 8}" r="6.5" fill="{color}"/>')
        p.append(text(1204, y, label, 23, CREAM, weight=600))
    # Clocks: stock against the production cap.
    stock = clocks[0]
    capped = next(r for r in clocks if r["gpu_cap"] == "2,300 MHz")
    p.append(heading(38, 778, "CLOCKS"))
    p.append(text(336, 822, "Stock GPU", 23, LABEL_INK, "end", 700))
    p.append(text(488, 822, capped["gpu_cap"], 23, GOLD, "end", 800))
    p.append(rule(38, 488, 836))
    y = 874
    facts = (("32K prefill", lambda r: f"{float(r['prefill_32k_tps']):,.0f}"), ("Peak SoC", lambda r: f"{float(r['peak_soc_c']):.1f} °C"),
             ("Peak GPU", lambda r: f"{float(r['peak_gpu_c']):.0f} °C"), ("Decode power", lambda r: f"{float(r['decode_gpu_w']):.0f} W"))
    for i, (label, fmt) in enumerate(facts):
        p.append(text(38, y, label, 25, CREAM, weight=600))
        p.append(text(336, y, fmt(stock), 28, CREAM, "end", 800))
        p.append(text(488, y, fmt(capped), 28, CREAM, "end", 800))
        if i < len(facts) - 1:
            p.append(rule(38, 488, y + 15))
        y += 49
    # Temperature graph.
    p.append(heading(552, 778, "TEMPERATURE", "°C"))
    xs = (633, 728, 823, 918)
    names = ["Stock|GPU" if r["gpu_cap"].startswith("Stock") else r["gpu_cap"].replace(" MHz", "|MHz") for r in clocks]
    p.append(graph(xs, 952, 808, 70, 100, (70, 80, 90, 100), names,
                   [([float(r["peak_soc_c"]) for r in clocks], EMBER, True), ([float(r["peak_gpu_c"]) for r in clocks], CREAM, False)], True))
    for x, label, color in ((584, "Peak SoC", EMBER), (774, "Peak GPU", CREAM)):
        p.append(f'<line x1="{x}" x2="{x + 28}" y1="1026" y2="1026" stroke="{color}" stroke-width="3"/><circle cx="{x + 14}" cy="1026" r="6.5" fill="{color}"/>')
        p.append(text(x + 38, 1034, label, 22, CREAM, weight=600))
    # Quality.
    p.append(heading(1010, 778, "QUALITY"))
    scores = (("Standard set", f"{summary['quality']['overallPct']:.1f}%"), ("AIME 2025", quality_score("AIME")),
              ("GPQA-Diamond", quality_score("GPQA")), ("HarmBench refusals", "0 of 100"))
    y = 838
    for i, (label, value) in enumerate(scores):
        p.append(text(1010, y, label, 24, CREAM, weight=600))
        p.append(text(1372, y, value, 27, CREAM, "end", 800))
        if i < len(scores) - 1:
            p.append(rule(1010, 1372, y + 21))
        y += 60
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
{SCORE_DEFS}
<g clip-path="url(#corners)">
{chr(10).join(p)}
</g>
</svg>
"""


def social_preview(summary):
    """The 2:1 link preview: the whale plate with the headline speeds in its rock band."""
    w, h = 1280, 640
    art = base64.b64encode((OUT / "social-art.jpg").read_bytes()).decode()
    rate = {(r["promptType"], r["mode"], r["concurrency"]): r["aggregateDecodeTps"] for r in summary["decode"]}
    stats = ((f"{max(r['prefillTps'] for r in summary['prefill']):,.0f}", "TOK/S PREFILL"), (f"{rate['prose', 'greedy', 1]:.0f}", "TOK/S PROSE"),
             (f"{rate['code', 'greedy', 1]:.0f}", "TOK/S CODE"), (f"{rate['prose', 'greedy', 16]:.0f}", "TOK/S AT 16 REQUESTS"), ("1M", "CONTEXT"))
    p = [f'<image href="data:image/jpeg;base64,{art}" width="{w}" height="{h}"/>']
    step = w / len(stats)
    for i, (value, label) in enumerate(stats):
        x = step * (i + 0.5)
        if i:
            p.append(f'<line x1="{x - step / 2:.1f}" x2="{x - step / 2:.1f}" y1="462" y2="596" stroke="{TITLE_INK}" stroke-opacity="0.55"/>')
        p.append(f'<text x="{x:.1f}" y="540" fill="url(#heat)" font-size="84" font-weight="800" letter-spacing="-2" text-anchor="middle" opacity="0.4" filter="url(#numglow)">{value}</text>')
        p.append(text(x, 540, value, 84, "url(#heat)", "middle", 800, ' letter-spacing="-2"'))
        p.append(text(x, 580, label, 18, CREAM, "middle", 700, ' letter-spacing="1.4"'))
    defs = SCORE_DEFS.replace('x2="1330"', 'x2="1210"').replace('<rect width="1402" height="1066" rx="26"/>', f'<rect width="{w}" height="{h}"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
{defs}
{chr(10).join(p)}
</svg>
"""


def build():
    summary = json.loads((ROOT / "benchmarks/v1.2/summary.json").read_text())
    return {"chart-prefill.svg": prefill(summary), "chart-prefill-climb.svg": climb(),
            "chart-decode.svg": decode(summary), "chart-kl.svg": kl(), "chart-quality.svg": quality(),
            "scorecard.svg": scorecard(summary), "social-preview.svg": social_preview(summary)}


def main():
    for name, svg in build().items():
        (OUT / name).write_text(svg)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
