"""
Definitive benchmark / validation tool for the Connect 4 AI Arena.

Does NOT modify anything under arena/. Read-only imports only.
No prompt changes. No parser changes. No model changes.

Reuses the EXACT prompt-building and parsing code the live game uses:
  - Player.system() / Player.user()   for prompt text
  - Board                              for board state and rendering
  - clean_response / strip_thinking / strip_markdown / extract_json  for cleanup
  - cols                               for column-letter -> index conversion
  - benchmark_engine                   for read-only tactical move scoring

Usage:
    python benchmark.py                          # 20 runs/model, all models
    python benchmark.py --runs 5                 # quick smoke test
    python benchmark.py --model qwen/qwen3-32b    # single model only
    python benchmark.py --replay benchmark_raw.json   # rerun only failed cases

Output:
    benchmark_raw.json      - every individual run, full detail (prompts,
                               every pipeline stage, tactical score, errors)
    benchmark_summary.json  - per-model aggregate statistics + classification
    benchmark_report.html   - sortable tables, charts, failure examples
"""
import os
import sys
import json
import time
import argparse
import statistics
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

from arena.board import Board, cols
from arena.player import Player
from arena.llm import (
    clean_response,
    strip_thinking,
    strip_markdown,
    extract_json,
    MODEL_PROVIDER,
    GROQ_SUPPORTS_REASONING_FORMAT,
    GROQ_SUPPORTS_JSON_MODE,
    GEMINI_SUPPORTS_THINKING_CONFIG,
    _get_max_tokens,
)
from benchmark_engine import evaluate_move

ALL_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "gemini-2.5-flash",
]

TRUNCATION_FINISH_REASONS = {"length", "MAX_TOKENS", "FinishReason.MAX_TOKENS", "2"}

# ---------------------------------------------------------------------------
# Fixed board state — identical across every model and every run, so results
# are comparable. Mover is whichever color the fixed sequence leaves to move,
# at a position with both a live threat (an open 3) and an opportunity, so
# both win-detection and block-detection scoring paths get exercised.
# ---------------------------------------------------------------------------
def build_fixed_board() -> Board:
    b = Board()
    for col_letter in ["D", "D", "C", "E", "C", "E", "B"]:
        b.move(cols.find(col_letter))
    return b


BOARD = build_fixed_board()
MOVER_COLOR = BOARD.player
LEGAL_MOVES = ", ".join(BOARD.legal_moves())
_illegal = BOARD.illegal_moves()
ILLEGAL_MOVES = (
    "\nYou must NOT make any of these moves which are ILLEGAL: " + ", ".join(_illegal)
) if _illegal else ""

_dummy_player = Player.__new__(Player)
_dummy_player.color = MOVER_COLOR
SYSTEM_PROMPT = _dummy_player.system(BOARD, LEGAL_MOVES, ILLEGAL_MOVES)
USER_PROMPT = _dummy_player.user(BOARD, LEGAL_MOVES, ILLEGAL_MOVES)

BOARD_STATE_SNAPSHOT = {
    "cells": [row[:] for row in BOARD.cells],
    "player_to_move": MOVER_COLOR,
    "legal_moves": BOARD.legal_moves(),
    "ascii": repr(BOARD),
}


# ---------------------------------------------------------------------------
# Structured error capture (requirement 4)
# ---------------------------------------------------------------------------

def _capture_exception(exc: Exception) -> dict:
    """
    Extract structured fields from a provider SDK exception, not just str(exc).
    Groq's SDK (openai-style) and google-generativeai exceptions expose
    different attributes; use getattr defensively so this works for both
    and degrades gracefully for unknown exception types.
    """
    info = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "full_traceback": traceback.format_exc(),
        "http_status": getattr(exc, "status_code", None),
        "provider_error_code": None,
        "provider_error_message": getattr(exc, "message", None),
    }
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            info["provider_error_code"] = err.get("code") or err.get("type")
            info["provider_error_message"] = err.get("message", info["provider_error_message"])
    response = getattr(exc, "response", None)
    if response is not None:
        info["http_status"] = info["http_status"] or getattr(response, "status_code", None)
    return info


# ---------------------------------------------------------------------------
# Per-call raw probes
# ---------------------------------------------------------------------------

def call_groq_raw(model_name: str, retry_count: int = 0):
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=60.0)

    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "temperature": 0.0,
        "max_tokens": _get_max_tokens(),
    }
    if model_name in GROQ_SUPPORTS_REASONING_FORMAT:
        kwargs["reasoning_format"] = "hidden"
    if model_name in GROQ_SUPPORTS_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency = time.time() - t0

    choice = resp.choices[0]
    message = choice.message
    usage = resp.usage

    return {
        "latency_s": round(latency, 3),
        "finish_reason": choice.finish_reason,
        "raw_content": message.content,
        "reasoning_field": getattr(message, "reasoning", None),
        "tool_calls": str(getattr(message, "tool_calls", None)) if getattr(message, "tool_calls", None) else None,
        "refusal": getattr(message, "refusal", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "reasoning_tokens": (
            getattr(usage, "completion_tokens_details", None).reasoning_tokens
            if usage and getattr(usage, "completion_tokens_details", None) else None
        ),
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "retry_count": retry_count,
        "request_kwargs_used": {k: v for k, v in kwargs.items() if k != "messages"},
    }


def call_gemini_raw(model_name: str, retry_count: int = 0):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": _get_max_tokens(),
        "response_mime_type": "application/json",
    }
    if model_name in GEMINI_SUPPORTS_THINKING_CONFIG:
        generation_config["thinking_config"] = {"thinking_budget": 0}

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
        generation_config=generation_config,
    )

    t0 = time.time()
    resp = model.generate_content(USER_PROMPT, request_options={"timeout": 60.0})
    latency = time.time() - t0

    candidate = resp.candidates[0] if resp.candidates else None
    finish_reason = str(candidate.finish_reason) if candidate else None
    usage = getattr(resp, "usage_metadata", None)

    try:
        text_value = resp.text
    except Exception:
        text_value = None

    return {
        "latency_s": round(latency, 3),
        "finish_reason": finish_reason,
        "raw_content": text_value,
        "reasoning_field": None,
        "tool_calls": None,
        "refusal": None,
        "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "completion_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
        "reasoning_tokens": getattr(usage, "thoughts_token_count", None) if usage else None,
        "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        "prompt_feedback": str(getattr(resp, "prompt_feedback", None)),
        "retry_count": retry_count,
        "request_kwargs_used": {"generation_config": {k: v for k, v in generation_config.items()}},
    }


# ---------------------------------------------------------------------------
# Pipeline replay: every intermediate stage saved (requirement 3)
# ---------------------------------------------------------------------------

def run_pipeline(raw_content: str) -> dict:
    stages = {
        "raw_content": raw_content,
        "after_strip_thinking": None,
        "after_strip_markdown": None,
        "after_extract_json": None,
        "after_clean_response": None,
        "parsed_json": None,
    }

    result = {
        "cleaned_response": None,
        "json_parse_success": False,
        "schema_valid": False,
        "move_column_raw": None,
        "move_column_normalized": None,
        "board_col_index": None,
        "move_legal": False,
        "error": None,
        # Requirement 6: explicit prompt-compliance booleans
        "contains_json_only": False,
        "contains_markdown": False,
        "contains_prose": False,
        "contains_think_tags": False,
        "contains_multiple_json_objects": False,
        "contains_invalid_schema": False,
        "contains_missing_move_column": False,
        "contains_invalid_move_column": False,
        # Requirement 5: truncation detail
        "truncated": False,
        "last_200_chars": (raw_content or "")[-200:],
    }

    if not raw_content:
        result["error"] = "empty_or_none_content"
        result["contains_missing_move_column"] = True
        return {**result, "pipeline_stages": stages}

    result["contains_think_tags"] = "<think>" in raw_content

    after_think = strip_thinking(raw_content)
    stages["after_strip_thinking"] = after_think

    after_md = strip_markdown(after_think)
    stages["after_strip_markdown"] = after_md
    result["contains_markdown"] = "```" in raw_content

    after_extract = extract_json(after_md)
    stages["after_extract_json"] = after_extract

    cleaned = clean_response(raw_content)
    stages["after_clean_response"] = cleaned
    result["cleaned_response"] = cleaned

    depth = 0
    object_starts = 0
    for ch in after_md:
        if ch == "{":
            if depth == 0:
                object_starts += 1
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
    result["contains_multiple_json_objects"] = object_starts > 1

    left = after_md.find("{")
    right = after_md.rfind("}")
    has_json_object = left != -1 and right != -1 and right > left
    text_stripped = after_md.strip()
    if has_json_object:
        has_prose = (left > 0) or (right < len(text_stripped) - 1) or object_starts > 1
    else:
        has_prose = bool(text_stripped)  # non-empty text with no JSON object at all is pure prose
    result["contains_prose"] = has_prose
    result["contains_json_only"] = has_json_object and (not has_prose) and (not result["contains_markdown"])

    try:
        parsed = json.loads(cleaned)
        stages["parsed_json"] = parsed
        result["json_parse_success"] = True
    except Exception as e:
        result["error"] = f"json_decode_error: {e}"
        result["contains_invalid_schema"] = True
        return {**result, "pipeline_stages": stages}

    required_keys = {"evaluation", "threats", "opportunities", "strategy", "move_column"}
    present_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
    result["schema_valid"] = required_keys.issubset(present_keys)
    result["contains_invalid_schema"] = not result["schema_valid"]

    move_raw = str(parsed.get("move_column") or "") if isinstance(parsed, dict) else ""
    move_norm = move_raw.strip().upper()
    result["move_column_raw"] = move_raw
    result["move_column_normalized"] = move_norm

    if not move_raw:
        result["contains_missing_move_column"] = True

    if len(move_norm) == 1 and move_norm in cols:
        col_index = cols.find(move_norm)
        result["board_col_index"] = col_index
        if 0 <= col_index <= 6 and BOARD.height(col_index) < 6:
            result["move_legal"] = True
        else:
            result["error"] = "illegal_move_full_or_out_of_range"
            result["contains_invalid_move_column"] = True
    else:
        result["board_col_index"] = -1
        if move_raw:
            result["contains_invalid_move_column"] = True
        result["error"] = result["error"] or "move_column_not_single_valid_letter"

    return {**result, "pipeline_stages": stages}


# ---------------------------------------------------------------------------
# Tactical move scoring (requirement 1) — read-only, uses benchmark_engine
# ---------------------------------------------------------------------------

def score_move_quality(move_col: str) -> dict:
    if not move_col or move_col not in cols:
        return {
            "wins_immediately": False,
            "blocks_opponent_win": False,
            "is_legal_tactical": False,
            "is_engine_top_choice": False,
            "engine_rank": None,
            "engine_top_moves": None,
            "opponent_threats": [],
        }
    tactical = evaluate_move(BOARD, MOVER_COLOR, move_col, top_n=2, depth=6)
    return {
        "wins_immediately": tactical["wins_immediately"],
        "blocks_opponent_win": tactical["blocks_opponent_win"],
        "is_legal_tactical": tactical["is_legal"],
        "is_engine_top_choice": tactical["is_top_choice"],
        "engine_rank": tactical["engine_rank"],
        "engine_top_moves": tactical["engine_top_moves"],
        "opponent_threats": tactical["opponent_threats"],
    }


# ---------------------------------------------------------------------------
# Main per-run execution
# ---------------------------------------------------------------------------

def run_once(model_name: str, run_index: int, retry_count: int = 0) -> dict:
    provider = MODEL_PROVIDER.get(model_name)
    record = {
        "model": model_name,
        "provider": provider,
        "run": run_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Requirement 2: full raw prompt saved per run
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "board_state": BOARD_STATE_SNAPSHOT,
    }
    try:
        if provider == "groq":
            raw = call_groq_raw(model_name, retry_count=retry_count)
        elif provider == "gemini":
            raw = call_gemini_raw(model_name, retry_count=retry_count)
        else:
            record["exception"] = _capture_exception(ValueError(f"unknown provider {provider}"))
            return record

        record.update(raw)
        record["raw_content_length"] = len(raw["raw_content"]) if raw["raw_content"] else 0
        finish_reason_truncated = str(raw.get("finish_reason")) in TRUNCATION_FINISH_REASONS

        pipeline_result = run_pipeline(raw["raw_content"])
        record.update(pipeline_result)
        record["truncated"] = finish_reason_truncated or pipeline_result.get("truncated", False)

        move_col = record.get("move_column_normalized")
        record["tactical"] = score_move_quality(move_col)

    except Exception as e:
        record["exception"] = _capture_exception(e)

    return record


def benchmark_model(model_name: str, n_runs: int) -> list:
    provider = MODEL_PROVIDER.get(model_name)
    if not provider:
        print(f"  SKIP: {model_name} not in MODEL_PROVIDER")
        return []

    runs = []
    for i in range(n_runs):
        record = run_once(model_name, i + 1)
        runs.append(record)
        if "exception" in record:
            status = f"EXCEPTION ({record['exception']['exception_type']})"
        elif record.get("move_legal"):
            tac = record.get("tactical", {})
            tag = []
            if tac.get("wins_immediately"):
                tag.append("WIN")
            if tac.get("blocks_opponent_win"):
                tag.append("BLOCK")
            if tac.get("is_engine_top_choice"):
                tag.append("TOP-CHOICE")
            status = "LEGAL " + (",".join(tag) if tag else "ordinary")
        else:
            status = "ILLEGAL/INVALID"
        print(f"  [{model_name}] run {i+1}/{n_runs}: {status}"
              f"  finish_reason={record.get('finish_reason')}"
              f"  latency={record.get('latency_s')}")

    return runs


# ---------------------------------------------------------------------------
# Replay mode (requirement 7)
# ---------------------------------------------------------------------------

def is_failed_run(record: dict) -> bool:
    if "exception" in record:
        return True
    if not record.get("move_legal"):
        return True
    return False


def replay_failures(input_path: str):
    with open(input_path) as f:
        previous_runs = json.load(f)

    failed = [r for r in previous_runs if is_failed_run(r)]
    print(f"Loaded {len(previous_runs)} runs from {input_path}; {len(failed)} were failures.")

    replayed = []
    for i, old_record in enumerate(failed):
        model_name = old_record["model"]
        prior_retries = old_record.get("retry_count", 0)
        print(f"Replaying failure {i+1}/{len(failed)}: {model_name} "
              f"(original run #{old_record.get('run')})")
        new_record = run_once(model_name, old_record.get("run", i + 1), retry_count=prior_retries + 1)
        new_record["replay_of_original_error"] = old_record.get("error") or (
            old_record.get("exception", {}).get("exception_type") if "exception" in old_record else None
        )
        replayed.append(new_record)

    out_path = "benchmark_replay.json"
    with open(out_path, "w") as f:
        json.dump(replayed, f, indent=2, default=str)

    still_failing = sum(1 for r in replayed if is_failed_run(r))
    print(f"\nReplay complete. {len(replayed) - still_failing}/{len(replayed)} "
          f"previously-failed cases now succeed.")
    print(f"Wrote {out_path}")
    return replayed


# ---------------------------------------------------------------------------
# Summarization + classification (requirement 9)
# ---------------------------------------------------------------------------

def summarize(runs: list) -> dict:
    if not runs:
        return {}

    n = len(runs)
    exceptions = [r for r in runs if "exception" in r]
    completed = [r for r in runs if "exception" not in r]

    json_ok = sum(1 for r in completed if r.get("json_parse_success"))
    schema_ok = sum(1 for r in completed if r.get("schema_valid"))
    legal_ok = sum(1 for r in completed if r.get("move_legal"))
    empty_content = sum(1 for r in completed if not r.get("raw_content"))
    think_tags = sum(1 for r in completed if r.get("contains_think_tags"))
    markdown_fences = sum(1 for r in completed if r.get("contains_markdown"))
    extra_prose = sum(1 for r in completed if r.get("contains_prose"))
    multi_json = sum(1 for r in completed if r.get("contains_multiple_json_objects"))
    truncated = sum(1 for r in completed if r.get("truncated"))
    missing_move = sum(1 for r in completed if r.get("contains_missing_move_column"))
    invalid_move = sum(1 for r in completed if r.get("contains_invalid_move_column"))

    wins_found = sum(1 for r in completed if r.get("move_legal") and r.get("tactical", {}).get("wins_immediately"))
    blocks_found = sum(1 for r in completed if r.get("move_legal") and r.get("tactical", {}).get("blocks_opponent_win"))
    top_choice = sum(1 for r in completed if r.get("move_legal") and r.get("tactical", {}).get("is_engine_top_choice"))

    latencies = [r["latency_s"] for r in completed if r.get("latency_s") is not None]
    completion_tokens = [r["completion_tokens"] for r in completed if r.get("completion_tokens") is not None]
    reasoning_tokens = [r["reasoning_tokens"] for r in completed if r.get("reasoning_tokens") is not None]
    content_lengths = [r["raw_content_length"] for r in completed if r.get("raw_content_length") is not None]

    def avg(lst):
        return round(statistics.mean(lst), 2) if lst else None

    move_legal_pct = round(100 * legal_ok / n, 1)
    json_valid_pct = round(100 * json_ok / n, 1)
    schema_valid_pct = round(100 * schema_ok / n, 1)
    api_success_pct = round(100 * (n - len(exceptions)) / n, 1)

    # Requirement 9: classification purely from statistics, thresholds stated explicitly
    if api_success_pct >= 95 and move_legal_pct >= 95 and json_valid_pct >= 98:
        classification = "Production Ready"
    elif api_success_pct >= 85 and move_legal_pct >= 80:
        classification = "Mostly Reliable"
    elif api_success_pct >= 50 and move_legal_pct >= 40:
        classification = "Experimental"
    else:
        classification = "Not Recommended"

    error_types = {}
    for r in exceptions:
        et = r["exception"]["exception_type"]
        error_types[et] = error_types.get(et, 0) + 1
    for r in completed:
        if r.get("error"):
            error_types[r["error"]] = error_types.get(r["error"], 0) + 1

    return {
        "model": runs[0]["model"],
        "provider": runs[0]["provider"],
        "total_runs": n,
        "api_exceptions": len(exceptions),
        "api_success_pct": api_success_pct,
        "empty_content_count": empty_content,
        "json_valid_pct": json_valid_pct,
        "schema_valid_pct": schema_valid_pct,
        "move_legal_pct": move_legal_pct,
        "illegal_move_pct": round(100 * (n - legal_ok) / n, 1),
        "truncated_pct": round(100 * truncated / n, 1),
        "think_tag_pct": round(100 * think_tags / n, 1),
        "markdown_fence_pct": round(100 * markdown_fences / n, 1),
        "extra_prose_pct": round(100 * extra_prose / n, 1),
        "multi_json_object_pct": round(100 * multi_json / n, 1),
        "missing_move_column_pct": round(100 * missing_move / n, 1),
        "invalid_move_column_pct": round(100 * invalid_move / n, 1),
        "wins_found_count": wins_found,
        "blocks_found_count": blocks_found,
        "engine_top_choice_pct": round(100 * top_choice / max(legal_ok, 1), 1),
        "avg_latency_s": avg(latencies),
        "avg_completion_tokens": avg(completion_tokens),
        "avg_reasoning_tokens": avg(reasoning_tokens) if reasoning_tokens else 0,
        "avg_content_length_chars": avg(content_lengths),
        "error_type_breakdown": error_types,
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# HTML report (requirement 8)
# ---------------------------------------------------------------------------

def generate_html_report(all_runs: list, all_summaries: list, out_path: str = "benchmark_report.html"):
    def esc(s):
        if s is None:
            return ""
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    class_colors = {
        "Production Ready": "#1a7f37",
        "Mostly Reliable": "#9a6700",
        "Experimental": "#bf3989",
        "Not Recommended": "#cf222e",
    }

    summaries_sorted = sorted(
        all_summaries,
        key=lambda s: (s["json_valid_pct"], s["schema_valid_pct"], s["move_legal_pct"]),
        reverse=True,
    )

    rows_html = []
    for s in summaries_sorted:
        color = class_colors.get(s["classification"], "#666")
        rows_html.append(f"""
        <tr>
          <td>{esc(s['model'])}</td>
          <td>{esc(s['provider'])}</td>
          <td><span class="badge" style="background:{color}">{esc(s['classification'])}</span></td>
          <td>{s['api_success_pct']}%</td>
          <td>{s['json_valid_pct']}%</td>
          <td>{s['schema_valid_pct']}%</td>
          <td>{s['move_legal_pct']}%</td>
          <td>{s['engine_top_choice_pct']}%</td>
          <td>{s['wins_found_count']}</td>
          <td>{s['blocks_found_count']}</td>
          <td>{s['truncated_pct']}%</td>
          <td>{s['think_tag_pct']}%</td>
          <td>{s['avg_latency_s']}</td>
          <td>{s['avg_completion_tokens']}</td>
          <td>{s['avg_reasoning_tokens']}</td>
        </tr>""")

    failure_examples = []
    seen_models = set()
    for r in all_runs:
        if r["model"] in seen_models:
            continue
        if is_failed_run(r):
            seen_models.add(r["model"])
            if "exception" in r:
                reason = f"{r['exception']['exception_type']}: {esc(r['exception']['exception_message'][:200])}"
                raw = ""
            else:
                reason = esc(r.get("error") or "illegal move")
                raw = esc((r.get("raw_content") or "")[:500])
            failure_examples.append(f"""
        <div class="failure-card">
          <h4>{esc(r['model'])} — run #{r.get('run')}</h4>
          <p><b>Reason:</b> {reason}</p>
          <p><b>finish_reason:</b> {esc(r.get('finish_reason'))} &nbsp; <b>truncated:</b> {r.get('truncated')}</p>
          <pre>{raw}</pre>
        </div>""")

    provider_groups = {}
    for s in all_summaries:
        provider_groups.setdefault(s["provider"], []).append(s)
    provider_compare_rows = []
    for prov, items in provider_groups.items():
        avg_json = round(statistics.mean([i["json_valid_pct"] for i in items]), 1)
        avg_legal = round(statistics.mean([i["move_legal_pct"] for i in items]), 1)
        avg_lat = round(statistics.mean([i["avg_latency_s"] or 0 for i in items]), 2)
        provider_compare_rows.append(
            f"<tr><td>{esc(prov)}</td><td>{len(items)}</td><td>{avg_json}%</td>"
            f"<td>{avg_legal}%</td><td>{avg_lat}s</td></tr>"
        )

    chart_labels = json.dumps([s["model"] for s in summaries_sorted])
    chart_json = json.dumps([s["json_valid_pct"] for s in summaries_sorted])
    chart_legal = json.dumps([s["move_legal_pct"] for s in summaries_sorted])
    chart_latency = json.dumps([s["avg_latency_s"] or 0 for s in summaries_sorted])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Connect 4 Arena — Benchmark Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 24px; background: #fafafa; color: #1f2328; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 17px; margin-top: 36px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; margin-top: 10px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; font-size: 13px; text-align: left; }}
  th {{ background: #f0f2f4; cursor: pointer; position: sticky; top: 0; }}
  th:hover {{ background: #e2e5e8; }}
  tr:nth-child(even) {{ background: #f9fafb; }}
  .badge {{ color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }}
  .failure-card {{ background: white; border: 1px solid #d0d7de; border-radius: 6px; padding: 12px; margin-bottom: 12px; }}
  .failure-card pre {{ background: #f6f8fa; padding: 8px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }}
  .chart-box {{ background: white; border: 1px solid #d0d7de; border-radius: 6px; padding: 16px; margin-top: 10px; max-width: 900px; }}
  .note {{ font-size: 12px; color: #57606a; margin-top: 4px; }}
</style>
</head>
<body>
<h1>Connect 4 AI Arena — Benchmark Report</h1>
<p class="note">Generated {esc(datetime.now(timezone.utc).isoformat())} · {len(all_runs)} total runs across {len(all_summaries)} models</p>

<h2>Model Ranking (click column headers to sort)</h2>
<table id="rankTable">
  <thead><tr>
    <th onclick="sortTable(0)">Model</th>
    <th onclick="sortTable(1)">Provider</th>
    <th onclick="sortTable(2)">Classification</th>
    <th onclick="sortTable(3)">API Success</th>
    <th onclick="sortTable(4)">JSON Valid</th>
    <th onclick="sortTable(5)">Schema Valid</th>
    <th onclick="sortTable(6)">Move Legal</th>
    <th onclick="sortTable(7)">Engine Top-Choice</th>
    <th onclick="sortTable(8)">Wins Found</th>
    <th onclick="sortTable(9)">Blocks Found</th>
    <th onclick="sortTable(10)">Truncated</th>
    <th onclick="sortTable(11)">Think-tags</th>
    <th onclick="sortTable(12)">Avg Latency (s)</th>
    <th onclick="sortTable(13)">Avg Tokens</th>
    <th onclick="sortTable(14)">Avg Reasoning Tokens</th>
  </tr></thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>
<p class="note">Engine Top-Choice / Wins Found / Blocks Found come from a shallow heuristic
minimax (depth 6), not a perfect Connect 4 solver — treat as tactical sanity-checking,
not ground-truth optimal play.</p>

<h2>Provider Comparison</h2>
<table>
  <thead><tr><th>Provider</th><th># Models</th><th>Avg JSON Valid</th><th>Avg Move Legal</th><th>Avg Latency</th></tr></thead>
  <tbody>{"".join(provider_compare_rows)}</tbody>
</table>

<h2>Charts</h2>
<div class="chart-box"><canvas id="jsonChart"></canvas></div>
<div class="chart-box"><canvas id="legalChart"></canvas></div>
<div class="chart-box"><canvas id="latencyChart"></canvas></div>

<h2>Failure Examples (first failure per model)</h2>
{"".join(failure_examples) if failure_examples else "<p>No failures recorded.</p>"}

<script>
function sortTable(colIdx) {{
  const table = document.getElementById("rankTable");
  const rows = Array.from(table.tBodies[0].rows);
  const asc = table.dataset.sortCol == colIdx && table.dataset.sortDir !== "asc";
  rows.sort((a, b) => {{
    let x = a.cells[colIdx].innerText.replace('%','');
    let y = b.cells[colIdx].innerText.replace('%','');
    const xn = parseFloat(x), yn = parseFloat(y);
    if (!isNaN(xn) && !isNaN(yn)) return asc ? xn - yn : yn - xn;
    return asc ? x.localeCompare(y) : y.localeCompare(x);
  }});
  rows.forEach(r => table.tBodies[0].appendChild(r));
  table.dataset.sortCol = colIdx;
  table.dataset.sortDir = asc ? "asc" : "desc";
}}

const labels = {chart_labels};
new Chart(document.getElementById('jsonChart'), {{
  type: 'bar',
  data: {{ labels: labels, datasets: [{{ label: 'JSON Valid %', data: {chart_json}, backgroundColor: '#0969da' }}] }},
  options: {{ plugins: {{ title: {{ display: true, text: 'JSON Validity by Model' }} }} }}
}});
new Chart(document.getElementById('legalChart'), {{
  type: 'bar',
  data: {{ labels: labels, datasets: [{{ label: 'Move Legal %', data: {chart_legal}, backgroundColor: '#1a7f37' }}] }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Move Legality by Model' }} }} }}
}});
new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: {{ labels: labels, datasets: [{{ label: 'Avg Latency (s)', data: {chart_latency}, backgroundColor: '#9a6700' }}] }},
  options: {{ plugins: {{ title: {{ display: true, text: 'Average Latency by Model' }} }} }}
}});
</script>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--model", type=str, default=None, help="Benchmark a single model only")
    parser.add_argument("--replay", type=str, default=None, help="Replay only failed cases from a prior benchmark_raw.json")
    args = parser.parse_args()

    if args.replay:
        replay_failures(args.replay)
        return

    models = [args.model] if args.model else ALL_MODELS

    print(f"Board under test:\n{BOARD}")
    print(f"Legal moves: {LEGAL_MOVES}")
    print(f"Runs per model: {args.runs}\n")

    all_runs = []
    all_summaries = []

    for model_name in models:
        print(f"\n=== Benchmarking {model_name} ===")
        runs = benchmark_model(model_name, args.runs)
        all_runs.extend(runs)
        summary = summarize(runs)
        if summary:
            all_summaries.append(summary)

    with open("benchmark_raw.json", "w") as f:
        json.dump(all_runs, f, indent=2, default=str)
    with open("benchmark_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    generate_html_report(all_runs, all_summaries)

    print("\n\n" + "=" * 110)
    print("SUMMARY TABLE")
    print("=" * 110)
    header = (f"{'Model':42s} {'Class':17s} {'API%':>6s} {'JSON%':>6s} {'Schema%':>8s} "
              f"{'Legal%':>7s} {'Top%':>6s} {'AvgLat':>7s}")
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(f"{s['model']:42s} {s['classification']:17s} {s['api_success_pct']:6.1f} "
              f"{s['json_valid_pct']:6.1f} {s['schema_valid_pct']:8.1f} {s['move_legal_pct']:7.1f} "
              f"{s['engine_top_choice_pct']:6.1f} {s['avg_latency_s'] or 0:7.2f}")

    print("\nWrote benchmark_raw.json, benchmark_summary.json, benchmark_report.html")


if __name__ == "__main__":
    main()
