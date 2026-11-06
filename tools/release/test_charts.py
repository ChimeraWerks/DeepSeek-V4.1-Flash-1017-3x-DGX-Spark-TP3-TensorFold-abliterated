"""The README charts carry the benchmark numbers they are built from; no network."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

SPEC = importlib.util.spec_from_file_location("charts", Path(__file__).with_name("charts.py"))
charts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(charts)
SUMMARY = json.loads((charts.ROOT / "benchmarks/v1.2/summary.json").read_text())

def labels(svg):
    ET.fromstring(svg)  # well-formed
    return {e.text for e in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}text")}

def test_prefill_chart_shows_every_measured_point():
    found = labels(charts.prefill(SUMMARY))
    for r in SUMMARY["prefill"]:
        assert f"{r['prefillTps']:,.0f}" in found

def test_decode_chart_ends_at_the_sixteen_stream_values():
    found = labels(charts.decode(SUMMARY))
    for mode, prompt in [("greedy", "prose"), ("greedy", "code"), ("think", "prose"), ("think", "code")]:
        value = next(r["aggregateDecodeTps"] for r in SUMMARY["decode"]
                     if r["mode"] == mode and r["promptType"] == prompt and r["concurrency"] == 16)
        assert f"{value:.0f}" in found

def test_kl_chart_matches_fidelity_csv():
    found = labels(charts.kl())
    assert {"0.0784", "0.0872", "0.0937", "0.1002", "0.1111", "0.1225", "0.1234", "0.1268"} <= found

def test_quality_chart_matches_quality_csv():
    found = labels(charts.quality())
    assert {"90.5%", "97.5%", "88.7%", "88.0%", "97", "of 250"} <= found

def test_scorecard_lists_speed_clocks_and_quality():
    found = labels(charts.scorecard(SUMMARY))
    assert f"{max(r['prefillTps'] for r in SUMMARY['prefill']):,.0f}" in found
    for r in SUMMARY["decode"]:
        if r["promptType"] != "structured" and r["concurrency"] in charts.GRID_LOADS:
            assert charts.whole(r["aggregateDecodeTps"]) in found
    assert not {"82.0", "124.4", "270.3"} & found
    assert {"3,816", "3,711", "94.1 °C", "86.2 °C", "85 °C", "78 °C", "58 W", "41 W"} <= found
    assert not {"3,764", "3,734"} & found
    assert {"97.1%", "97.5%", "90.5%", "0 of 100"} <= found

def test_clock_table_is_on_both_cards():
    for path in ("README.md", "docs/huggingface-card.md"):
        card = (charts.ROOT / path).read_text()
        assert "| Stock GPU | 3,816 |" in card and "| **2,300 MHz (production)** | 3,711 |" in card
        assert "| 2,400 MHz |" not in card and "| 2,350 MHz |" not in card

def test_scorecard_decode_matches_both_cards():
    cards = [(charts.ROOT / path).read_text() for path in ("README.md", "docs/huggingface-card.md")]
    for card in cards:
        for r in SUMMARY["prefill"]:
            assert f"{r['prefillTps']:,.0f}" in card
        rows = [line for line in card.splitlines() if line.startswith(("| Prose,", "| Code,"))]
        assert len(rows) == 8 and not any("." in line for line in rows)
        for r in SUMMARY["decode"]:
            if r["promptType"] != "structured" and r["concurrency"] in charts.LOADS:
                assert charts.whole(r["aggregateDecodeTps"]) in card

def test_social_preview_shows_the_headlines():
    assert {"3,816", "82", "124", "270", "1M"} <= labels(charts.social_preview(SUMMARY))

def test_charts_carry_no_old_palette():
    for name, svg in charts.build().items():
        for color in ("#B86CEC", "#5E8FF0", "#8A55E6", "#C77DF0", "#060A1E", "#2A3366"):
            assert color not in svg, (name, color)

def test_committed_charts_are_current():
    for name, svg in charts.build().items():
        assert (charts.OUT / name).read_text() == svg, name
