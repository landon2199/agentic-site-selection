"""Copy the derived results the static page needs into docs/data/ (run after run_methods.py, validate.py, agents).

    python docs/build_docs_data.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

val = json.loads((ROOT / "results" / "validation.json").read_text())
summ = json.loads((ROOT / "results" / "summary.json").read_text())
(OUT / "results.json").write_text(json.dumps({"validation": val, "summary": summ}, indent=1))

runs = []
for p in sorted((ROOT / "results" / "agent_runs").glob("run_*.json")):
    r = json.loads(p.read_text())
    steps = []
    for e in r["events"]:
        if e["type"] == "delegation":
            steps.append({"kind": "delegate", "agent": e["agent"], "model": e["model"], "text": e["task"]})
        elif e["type"] == "mcp_tool_call":
            steps.append({"kind": "tool", "agent": e["agent"], "tool": e["tool"], "input": e["input"], "error": e["is_error"]})
        elif e["type"] == "report":
            steps.append({"kind": "report", "agent": e["agent"], "text": e["text"]})
    runs.append({"label": r["label"], "question": r["question"], "answer": r["final_answer"], "cost_usd": r["cost_usd"],
                 "tokens": r["tokens"], "seconds": r["seconds"], "n_api_calls": r["n_api_calls"],
                 "n_mcp_tool_calls": r["n_mcp_tool_calls"], "run_at": r["run_at"], "steps": steps,
                 "transcript": f"https://github.com/landon2199/agentic-site-selection/blob/main/results/agent_runs/{p.name}"})
(OUT / "runs.json").write_text(json.dumps(runs, indent=1))
print(f"wrote results.json and {len(runs)} runs")
