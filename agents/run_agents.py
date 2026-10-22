"""Manager + subagents over the local MCP server (Anthropic Python SDK, manual tool-use loops).

    python -m agents.run_agents                 # run the 5 saved demo questions
    python -m agents.run_agents --only 2        # run one of them
    python -m agents.run_agents --ask "..."     # ask your own question (not saved to docs)

Architecture
  Manager (Sonnet)       plans, delegates through four tools (ask_*_agent); never calls MCP tools itself
    geocode agent  (Haiku)  MCP: geocode, hex_profile
    data agent     (Haiku)  MCP: hex_profile, hot_spots
    analysis agent (Sonnet) MCP: overlay_suitability, find_similar, hot_spots, hex_profile
    validation agent (Haiku) MCP: validate
  Summary agent (Sonnet) no tools; writes the answer from the reports: ranked sites, reasons, method agreement,
                         and a confidence / caveats block.

Spending: every API call's usage is priced and added to results/agent_runs/spend_ledger.json; a run refuses
to start (and stops mid-run) if total spend would pass the hard budget (BUDGET_USD).
The API key is read at runtime from CLAUDE_API_CREDITS_KEY (env or Windows user registry) and passed to the
client in-process only; it is never printed or written.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "agent_runs"
LEDGER = RUNS / "spend_ledger.json"
BUDGET_USD = 8.00
RESERVE_USD = 0.40  # never start an API call with less than this left under the cap

SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"
# USD per million tokens: input, output, cache read, cache write (5-minute)
PRICES = {SONNET: (2.00, 10.00, 0.20, 2.50), HAIKU: (0.10, 0.50, 0.01, 0.125)}

QUESTIONS = [
    "Where in Houston should an independent coffee shop open next? Give me the top 5 areas and why.",
    "I'm looking at a storefront at 2700 White Oak Dr, Houston, TX 77007. Is that area a good bet for a new "
    "independent coffee shop?",
    "If I weight young professionals (ages 25-34) most heavily and care less about competition, which areas rank "
    "highest, and does the embedding method agree?",
    "Which parts of Harris County look demographically like the places where independent coffee shops already "
    "thrive, but don't have one yet?",
    "What monthly revenue will a coffee shop at 1500 McKinney St, Houston, TX 77010 make, and which exact storefront "
    "on that block should I lease?",
]

SUBAGENTS = {
    "geocode": {"model": HAIKU, "effort": "low", "tools": ["geocode", "hex_profile"],
                "system": "You are the geocode agent. Turn addresses or place descriptions into lon/lat and H3 hex ids "
                          "with the tools, then report the matched address, hex id, whether it is in the study area, and "
                          "any failed check. Street addresses only; if a name or landmark does not match, say so. Be brief."},
    "data": {"model": HAIKU, "effort": "low", "tools": ["hex_profile", "hot_spots"],
             "system": "You are the data agent. Fetch facts about hexes (profiles, hot spots) and report them as compact "
                       "bullet lists with the exact numbers, units and any warn/fail checks. Do not interpret beyond the data."},
    "analysis": {"model": SONNET, "effort": "medium",
                 "tools": ["overlay_suitability", "find_similar", "hot_spots", "hex_profile"],
                 "system": "You are the analysis agent. Run the spatial methods the task asks for: overlay_suitability "
                           "(method 1, expert-weighted overlay) and find_similar (method 2, Esri geodemographic embeddings). "
                           "Report ranked hexes with ids, lon/lat, scores, the key reasons (sub-scores, demographics, "
                           "competition), the engine used, and where the two methods agree or disagree. Quote numbers exactly "
                           "as returned and list warn/fail checks. Keep it under 400 words."},
    "validation": {"model": HAIKU, "effort": "low", "tools": ["validate"],
                   "system": "You are the validation agent. Call validate and report, for each method, mean AUC and "
                             "precision@k with their min-max range across blocks, the random baseline, the weakest block, and "
                             "the method agreement (top-10% Jaccard). Numbers only as returned. Be brief."},
}

MANAGER_SYSTEM = """You are the manager agent for a coffee-shop site-selection system for Harris County (Houston), TX.
You never see raw data and never call spatial tools yourself. Plan, then delegate with the ask_*_agent tools:
- geocode agent: addresses -> hex ids (only if the question names an address or place)
- analysis agent: overlay suitability and embedding similarity rankings, agreement between methods
- data agent: profiles of specific hexes, hot spots
- validation agent: how well each method recovered held-out shops (always ask it once)
Give each subagent a precise, self-contained task (it cannot see the question). Use as few delegations as
answer the question well (usually 2-4). The data has no sales, revenue, rent, lease, foot-traffic, parcel or
storefront information and hexes are ~5 km^2: if the question needs that, still gather what the data can say,
then make clear what cannot be answered. When you are done, reply with a short brief for the summary writer:
what was asked, what each subagent found (with numbers), and anything that cannot be answered reliably."""

SUMMARY_SYSTEM = """You are the summary agent. Write the final answer for a small-business owner from the manager's
brief and the subagent reports. Use only numbers that appear in the reports. Structure:
1. Direct answer (1-3 sentences). If the question cannot be answered reliably with this data, say so first and plainly.
2. Ranked sites (if any): hex id, approximate location (lon, lat), and why, in plain words.
3. Method agreement: do the overlay and embedding methods agree here? Cite the numbers.
4. Confidence and caveats block, headed "Confidence: high / medium / low" with one line of why, then bullets:
   validation results vs random baseline, data limits (no sales data, ~5 km^2 hexes, beta embeddings, Overture
   completeness), and anything a field visit should check.
Under 450 words. No marketing language."""


# ------------------------------------------------------------------ key, budget
def _api_key() -> str:
    key = os.environ.get("CLAUDE_API_CREDITS_KEY")
    if not key and sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            key = winreg.QueryValueEx(k, "CLAUDE_API_CREDITS_KEY")[0]
    if not key:
        raise SystemExit("CLAUDE_API_CREDITS_KEY is not set.")
    return key


class Budget:
    def __init__(self) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        self.ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"budget_usd": BUDGET_USD, "runs": []}
        self.run_cost = 0.0
        self.tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    @property
    def spent_before(self) -> float:
        return sum(r["cost_usd"] for r in self.ledger["runs"])

    def check(self) -> None:
        if self.spent_before + self.run_cost + RESERVE_USD > BUDGET_USD:
            raise BudgetExceeded(f"budget stop: spent ${self.spent_before + self.run_cost:.3f} of ${BUDGET_USD:.2f}")

    def add(self, model: str, usage) -> dict:
        pin, pout, pcr, pcw = PRICES.get(model, PRICES[SONNET])
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        u = {"input": usage.input_tokens, "output": usage.output_tokens, "cache_read": cr, "cache_write": cw}
        cost = (u["input"] * pin + u["output"] * pout + cr * pcr + cw * pcw) / 1e6
        self.run_cost += cost
        for k, v in u.items():
            self.tokens[k] += v
        return {**u, "cost_usd": round(cost, 5)}

    def save(self, label: str) -> None:
        self.ledger["runs"].append({"label": label, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "cost_usd": round(self.run_cost, 5), "tokens": self.tokens})
        LEDGER.write_text(json.dumps(self.ledger, indent=2))


class BudgetExceeded(RuntimeError):
    pass


# ------------------------------------------------------------------ agent loop
class Run:
    def __init__(self, client: anthropic.Anthropic, mcp: Client, mcp_tools: dict, budget: Budget) -> None:
        self.client, self.mcp, self.mcp_tools, self.budget = client, mcp, mcp_tools, budget
        self.events: list[dict] = []

    def _create(self, agent: str, model: str, effort: str, system: str, tools: list, messages: list):
        self.budget.check()
        t0 = time.perf_counter()
        resp = self.client.messages.create(model=model, max_tokens=8000, system=system, tools=tools or anthropic.NOT_GIVEN,
                                           messages=messages, output_config={"effort": effort},
                                           cache_control={"type": "ephemeral"})
        usage = self.budget.add(resp.model if resp.model in PRICES else model, resp.usage)
        self.events.append({"type": "api_call", "agent": agent, "model": resp.model, "stop_reason": resp.stop_reason,
                            "seconds": round(time.perf_counter() - t0, 1), "usage": usage})
        return resp

    async def _call_mcp(self, agent: str, name: str, args: dict) -> tuple[str, bool]:
        t0 = time.perf_counter()
        r = await self.mcp.call_tool(name, args)
        text = r.content[0].text if r.content else ""
        if not r.is_error and r.structured_content is not None:
            sc = r.structured_content
            text = json.dumps(sc.get("result", sc) if isinstance(sc, dict) else sc)
        self.events.append({"type": "mcp_tool_call", "agent": agent, "tool": name, "input": args, "is_error": r.is_error,
                            "seconds": round(time.perf_counter() - t0, 2), "result": _maybe_json(text)})
        return text, r.is_error

    async def subagent(self, name: str, task: str) -> str:
        cfg = SUBAGENTS[name]
        tools = [self.mcp_tools[t] for t in cfg["tools"]]
        messages = [{"role": "user", "content": task}]
        self.events.append({"type": "delegation", "agent": name, "model": cfg["model"], "task": task})
        for _ in range(8):
            resp = self._create(name, cfg["model"], cfg["effort"], cfg["system"], tools, messages)
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                break
            results = []
            for b in uses:
                text, err = await self._call_mcp(name, b.name, b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": err})
            messages.append({"role": "user", "content": results})
        report = _text(resp) or f"({name} agent returned no text; stop_reason={resp.stop_reason})"
        self.events.append({"type": "report", "agent": name, "text": report})
        return report

    async def manager(self, question: str) -> tuple[str, list[dict]]:
        tools = [{"name": f"ask_{n}_agent", "description": f"Delegate a self-contained task to the {n} agent "
                  f"(tools: {', '.join(c['tools'])}; model: {c['model']}). Returns its written report.",
                  "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}
                 for n, c in SUBAGENTS.items()]
        messages = [{"role": "user", "content": question}]
        reports = []
        for _ in range(8):
            resp = self._create("manager", SONNET, "medium", MANAGER_SYSTEM, tools, messages)
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                break
            results = await asyncio.gather(*[self.subagent(b.name[4:-6], b.input["task"]) for b in uses])
            for b, rep in zip(uses, results):
                reports.append({"agent": b.name[4:-6], "task": b.input["task"], "report": rep})
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": b.id, "content": rep}
                                                         for b, rep in zip(uses, results)]})
        brief = _text(resp)
        self.events.append({"type": "manager_brief", "text": brief})
        return brief, reports

    def summary(self, question: str, brief: str, reports: list[dict]) -> str:
        body = f"Question: {question}\n\nManager brief:\n{brief}\n\n" + "\n\n".join(
            f"--- {r['agent']} agent report (task: {r['task']})\n{r['report']}" for r in reports)
        resp = self._create("summary", SONNET, "medium", SUMMARY_SYSTEM, [], [{"role": "user", "content": body}])
        ans = _text(resp)
        self.events.append({"type": "final_answer", "text": ans})
        return ans


def _text(resp) -> str:
    return "\n".join(b.text for b in resp.content if b.type == "text").strip()


def _maybe_json(t: str):
    try:
        return json.loads(t)
    except (ValueError, TypeError):
        return t


async def ask(question: str, label: str, save: bool = True) -> dict:
    budget = Budget()
    budget.check()
    client = anthropic.Anthropic(api_key=_api_key(), max_retries=3)
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT))
    t0 = time.perf_counter()
    status, answer = "ok", ""
    async with Client(params, read_timeout_seconds=900) as mcp:
        listed = (await mcp.list_tools()).tools
        mcp_tools = {t.name: {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in listed}
        run = Run(client, mcp, mcp_tools, budget)
        try:
            brief, reports = await run.manager(question)
            answer = run.summary(question, brief, reports)
        except BudgetExceeded as exc:
            status, answer = "budget_stop", str(exc)
    record = {"label": label, "question": question, "status": status,
              "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "models": {"manager": SONNET, "summary": SONNET, **{k: v["model"] for k, v in SUBAGENTS.items()}},
              "seconds": round(time.perf_counter() - t0, 1),
              "cost_usd": round(budget.run_cost, 4), "tokens": budget.tokens,
              "n_api_calls": sum(e["type"] == "api_call" for e in run.events),
              "n_mcp_tool_calls": sum(e["type"] == "mcp_tool_call" for e in run.events),
              "final_answer": answer, "events": run.events}
    budget.save(label)
    if save:
        (RUNS / f"{label}.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"[{label}] {status} ${budget.run_cost:.3f} ({budget.spent_before + 0:.3f} total incl. this) "
          f"{record['n_api_calls']} API calls, {record['n_mcp_tool_calls']} tool calls, {record['seconds']} s")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, help="run only demo question N (1-5)")
    ap.add_argument("--ask", help="ask your own question")
    a = ap.parse_args()
    if a.ask:
        r = asyncio.run(ask(a.ask, f"adhoc_{int(time.time())}", save=False))
        print(r["final_answer"])
        return
    for i, q in enumerate(QUESTIONS, 1):
        if a.only and i != a.only:
            continue
        r = asyncio.run(ask(q, f"run_{i:02d}"))
        if r["status"] != "ok":
            break


if __name__ == "__main__":
    main()
