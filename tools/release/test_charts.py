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

def test_scorecard_lists_every_speed():
    found = labels(charts.scorecard(SUMMARY))
    assert f"{max(r['prefillTps'] for r in SUMMARY['prefill']):,.0f}" in found
    for r in SUMMARY["decode"]:
        if r["promptType"] != "structured" and r["concurrency"] in charts.LOADS:
            assert charts.tps(r["aggregateDecodeTps"]) in found

def test_scorecard_decode_matches_both_cards():
    cards = [(charts.ROOT / path).read_text() for path in ("README.md", "docs/huggingface-card.md")]
    for card in cards:
        for r in SUMMARY["prefill"]:
            assert f"{r['prefillTps']:,.0f}" in card
        for r in SUMMARY["decode"]:
            if r["promptType"] != "structured" and r["concurrency"] in charts.LOADS:
                assert charts.tps(r["aggregateDecodeTps"]) in card

def test_committed_charts_are_current():
    for name, svg in charts.build().items():
        assert (charts.OUT / name).read_text() == svg, name
