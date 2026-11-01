import argparse
import json
import math
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from xml.sax.saxutils import escape


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from artifacts.phase1.common import PHASE1_MANIFEST_PATH, PHASE1_RESULTS_ROOT, PHASE1_SUMMARY_DIR, read_jsonl, write_csv


METHOD_ORDER = ["best_of_n", "bbon", "mean_search", "slg"]
METHOD_LABELS = {
    "best_of_n": "bon",
    "bbon": "bbon",
    "mean_search": "mean",
    "slg": "slg",
}
COMPARISON_FIELDS = [
    "phase",
    "subphase",
    "dataset",
    "lm",
    "rm",
    "budget",
    "priority",
    "config_label",
    "action_strategy",
    "output_mode",
    "structured_answer_test_mode",
    "local_first_block_split_test_mode",
    "prefix_normalization_test_mode",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PHASE1_MANIFEST_PATH)
    parser.add_argument("--results-root", type=Path, default=PHASE1_RESULTS_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=PHASE1_SUMMARY_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    status_rows, experiments_cache = _collect_status_rows(manifest_rows)
    pairwise_rows = _collect_pairwise_rows(status_rows, experiments_cache)
    dashboard_rows = _build_dashboard_rows(status_rows, pairwise_rows)
    summary_dir = args.summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    _write_status_outputs(summary_dir, status_rows)
    _write_pairwise_outputs(summary_dir, pairwise_rows)
    _write_dashboard_outputs(summary_dir, dashboard_rows)
    _write_workbook(summary_dir / "phase1_dashboard.xlsx", status_rows, pairwise_rows, dashboard_rows)

    print(f"Wrote dashboard workbook to {summary_dir / 'phase1_dashboard.xlsx'}")
    print(f"Wrote live status CSV to {summary_dir / 'phase1_live_status.csv'}")
    print(f"Wrote pairwise CSV to {summary_dir / 'phase1_pairwise_common_prompt.csv'}")
    print(f"Wrote dashboard summary CSV to {summary_dir / 'phase1_dashboard_summary.csv'}")


def _collect_status_rows(manifest_rows: list[dict]) -> tuple[list[dict], dict[Path, dict[int | str, dict]]]:
    rows: list[dict] = []
    experiments_cache: dict[Path, dict[int | str, dict]] = {}

    for manifest_row in manifest_rows:
        output_dir = Path(manifest_row["expected_output_dir"])
        results_path = output_dir / "results.json"
        final_path = output_dir / "final_summary.json"
        results_data = _load_json(results_path)
        final_data = _load_json(final_path)

        experiments = {}
        if results_data is not None:
            for experiment in results_data.get("experiments", []):
                problem_id = experiment.get("problem_id")
                if problem_id is None:
                    problem_id = experiment.get("experiment_index")
                experiments[problem_id] = experiment
            experiments_cache[output_dir] = experiments

        completed_prompts = len(experiments)
        expected_prompts = None
        if results_data is not None:
            expected_prompts = results_data.get("total_experiments")
        if expected_prompts is None and final_data is not None:
            expected_prompts = final_data.get("total_prompt_count")

        average_score = _dig(results_data, "summary", "average_score")
        average_time = _dig(results_data, "summary", "average_time")
        total_generated_responses = _dig(results_data, "summary", "total_generated_responses")
        total_tokens = _dig(final_data, "total_tokens")

        if final_data is not None:
            run_status = "final"
        elif results_data is not None:
            run_status = "partial"
        else:
            run_status = "missing"

        row = {
            "phase": manifest_row.get("phase"),
            "subphase": manifest_row.get("subphase"),
            "dataset": manifest_row.get("dataset"),
            "lm": manifest_row.get("lm"),
            "rm": manifest_row.get("rm"),
            "method": manifest_row.get("method"),
            "budget": int(manifest_row.get("budget")) if manifest_row.get("budget") is not None else None,
            "priority": manifest_row.get("priority"),
            "config_label": manifest_row.get("config_label"),
            "action_strategy": manifest_row.get("action_strategy_override"),
            "output_mode": manifest_row.get("output_mode_override"),
            "structured_answer_test_mode": manifest_row.get("structured_answer_test_mode"),
            "local_first_block_split_test_mode": manifest_row.get("local_first_block_split_test_mode"),
            "prefix_normalization_test_mode": manifest_row.get("prefix_normalization_test_mode"),
            "manifest_status": manifest_row.get("status"),
            "run_status": run_status,
            "results_exists": results_data is not None,
            "final_summary_exists": final_data is not None,
            "completed_prompts": completed_prompts,
            "expected_prompts": expected_prompts,
            "completion_rate": _safe_ratio(completed_prompts, expected_prompts),
            "avg_best_score": average_score,
            "avg_time_per_prompt_sec": average_time,
            "est_total_time_sec": _product_or_none(average_time, completed_prompts),
            "total_generated_responses": total_generated_responses,
            "avg_generated_responses_per_prompt": _safe_ratio(total_generated_responses, completed_prompts),
            "total_tokens": total_tokens,
            "final_headline_metric_name": _dig(final_data, "final_headline_metric_name"),
            "final_headline_score": _dig(final_data, "final_headline_score"),
            "final_headline_source": _dig(final_data, "final_headline_source"),
            "interim_metric_name": _dig(final_data, "interim_metric_name"),
            "interim_metric_value": _dig(final_data, "interim_metric_value"),
            "official_metric_name": _dig(final_data, "official_metric_name"),
            "official_score": _dig(final_data, "official_score"),
            "official_eval_status": _dig(final_data, "official_eval_status"),
            "official_eval_note": _dig(final_data, "official_eval_note"),
            "git_commit_hash": _dig(final_data, "git_commit_hash"),
            "code_version_tag": _dig(final_data, "code_version_tag"),
            "results_json_path": str(results_path) if results_path.exists() else "",
            "final_summary_path": str(final_path) if final_path.exists() else "",
            "expected_output_dir": str(output_dir),
        }
        rows.append(row)

    rows.sort(key=lambda r: (_sort_key(r.get("subphase")), r["dataset"], r["lm"], r["rm"], r["budget"], _method_rank(r["method"])))
    return rows, experiments_cache


def _collect_pairwise_rows(status_rows: list[dict], experiments_cache: dict[Path, dict[int | str, dict]]) -> list[dict]:
    grouped_rows: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in status_rows:
        output_dir = Path(row["expected_output_dir"])
        if output_dir not in experiments_cache:
            continue
        key = tuple(row.get(field) for field in COMPARISON_FIELDS)
        grouped_rows[key][row["method"]] = row

    pairwise_rows: list[dict] = []
    for key, methods in grouped_rows.items():
        method_names = sorted(methods.keys(), key=_method_rank)
        for method_a, method_b in combinations(method_names, 2):
            row_a = methods[method_a]
            row_b = methods[method_b]
            exp_a = experiments_cache[Path(row_a["expected_output_dir"])]
            exp_b = experiments_cache[Path(row_b["expected_output_dir"])]
            common_ids = sorted(set(exp_a).intersection(exp_b), key=str)
            if not common_ids:
                continue
            diffs = []
            wins = ties = losses = 0
            for problem_id in common_ids:
                score_a = _numeric(exp_a[problem_id].get("best_score"))
                score_b = _numeric(exp_b[problem_id].get("best_score"))
                if score_a is None or score_b is None:
                    continue
                diff = score_a - score_b
                diffs.append(diff)
                if diff > 0:
                    wins += 1
                elif diff < 0:
                    losses += 1
                else:
                    ties += 1
            if not diffs:
                continue
            key_dict = dict(zip(COMPARISON_FIELDS, key))
            pairwise_rows.append(
                {
                    **key_dict,
                    "method_a": method_a,
                    "method_b": method_b,
                    "common_prompts": len(diffs),
                    "mean_score_diff_a_minus_b": sum(diffs) / len(diffs),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate_a": wins / len(diffs),
                }
            )

    pairwise_rows.sort(
        key=lambda r: (
            _sort_key(r.get("subphase")),
            r["dataset"],
            r["lm"],
            r["rm"],
            r["budget"],
            _method_rank(r["method_a"]),
            _method_rank(r["method_b"]),
        )
    )
    return pairwise_rows


def _build_dashboard_rows(status_rows: list[dict], pairwise_rows: list[dict]) -> list[dict]:
    status_lookup: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in status_rows:
        key = tuple(row.get(field) for field in COMPARISON_FIELDS)
        status_lookup[key][row["method"]] = row

    pairwise_lookup: dict[tuple, dict[tuple[str, str], dict]] = defaultdict(dict)
    for row in pairwise_rows:
        key = tuple(row.get(field) for field in COMPARISON_FIELDS)
        pairwise_lookup[key][(row["method_a"], row["method_b"])] = row

    dashboard_rows: list[dict] = []
    for key in sorted(status_lookup.keys(), key=_dashboard_key):
        key_dict = dict(zip(COMPARISON_FIELDS, key))
        row = dict(key_dict)
        methods = status_lookup[key]

        for method in METHOD_ORDER:
            label = METHOD_LABELS[method]
            status = methods.get(method)
            row[f"{label}_run_status"] = status.get("run_status") if status else None
            row[f"{label}_completed_prompts"] = status.get("completed_prompts") if status else None
            row[f"{label}_expected_prompts"] = status.get("expected_prompts") if status else None
            row[f"{label}_completion_rate"] = status.get("completion_rate") if status else None
            row[f"{label}_avg_best_score"] = status.get("avg_best_score") if status else None
            row[f"{label}_avg_time_per_prompt_sec"] = status.get("avg_time_per_prompt_sec") if status else None
            row[f"{label}_total_generated_responses"] = status.get("total_generated_responses") if status else None
            row[f"{label}_official_score"] = status.get("official_score") if status else None
            row[f"{label}_official_eval_status"] = status.get("official_eval_status") if status else None

        for method in ["bbon", "mean_search", "slg"]:
            label = METHOD_LABELS[method]
            pair = _lookup_pairwise(pairwise_lookup.get(key, {}), method, "best_of_n")
            row[f"{label}_vs_bon_common_prompts"] = pair.get("common_prompts") if pair else None
            row[f"{label}_vs_bon_mean_diff"] = pair.get("mean_score_diff_a_minus_b") if pair else None
            row[f"{label}_vs_bon_wins"] = pair.get("wins") if pair else None
            row[f"{label}_vs_bon_ties"] = pair.get("ties") if pair else None
            row[f"{label}_vs_bon_losses"] = pair.get("losses") if pair else None
            row[f"{label}_vs_bon_win_rate"] = pair.get("win_rate_a") if pair else None

        dashboard_rows.append(row)

    return dashboard_rows


def _lookup_pairwise(rows_for_key: dict[tuple[str, str], dict], method_a: str, method_b: str) -> dict | None:
    if (method_a, method_b) in rows_for_key:
        return rows_for_key[(method_a, method_b)]
    if (method_b, method_a) in rows_for_key:
        reverse = rows_for_key[(method_b, method_a)]
        common = reverse["common_prompts"]
        return {
            **reverse,
            "method_a": method_a,
            "method_b": method_b,
            "mean_score_diff_a_minus_b": -_numeric(reverse.get("mean_score_diff_a_minus_b"), 0.0),
            "wins": reverse["losses"],
            "ties": reverse["ties"],
            "losses": reverse["wins"],
            "win_rate_a": _safe_ratio(reverse["losses"], common),
        }
    return None


def _write_status_outputs(summary_dir: Path, status_rows: list[dict]) -> None:
    if status_rows:
        write_csv(summary_dir / "phase1_live_status.csv", status_rows, list(status_rows[0].keys()))

    main_rows = [row for row in status_rows if row.get("subphase") == "main_scaling"]
    if main_rows:
        write_csv(summary_dir / "main_scaling_status.csv", main_rows, list(main_rows[0].keys()))


def _write_pairwise_outputs(summary_dir: Path, pairwise_rows: list[dict]) -> None:
    if pairwise_rows:
        write_csv(summary_dir / "phase1_pairwise_common_prompt.csv", pairwise_rows, list(pairwise_rows[0].keys()))

    main_rows = [
        {
            "dataset": row["dataset"],
            "lm": row["lm"],
            "rm": row["rm"],
            "budget": row["budget"],
            "common_prompts": row["common_prompts"],
            "slg_minus_bon_mean_diff": row["mean_score_diff_a_minus_b"],
            "wins": row["wins"],
            "ties": row["ties"],
            "losses": row["losses"],
        }
        for row in pairwise_rows
        if row.get("subphase") == "main_scaling" and row["method_a"] == "slg" and row["method_b"] == "best_of_n"
    ]
    if main_rows:
        write_csv(summary_dir / "main_scaling_pairwise_slg_vs_bon_common_prompt.csv", main_rows, list(main_rows[0].keys()))


def _write_dashboard_outputs(summary_dir: Path, dashboard_rows: list[dict]) -> None:
    if dashboard_rows:
        write_csv(summary_dir / "phase1_dashboard_summary.csv", dashboard_rows, list(dashboard_rows[0].keys()))


def _write_workbook(path: Path, status_rows: list[dict], pairwise_rows: list[dict], dashboard_rows: list[dict]) -> None:
    refresh_time = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    readme_rows = [
        {"Key": "Refresh Time", "Value": refresh_time},
        {"Key": "Workbook", "Value": str(path)},
        {"Key": "Refresh Command", "Value": "python artifacts/phase1/build_phase1_dashboard.py"},
        {"Key": "Live Status CSV", "Value": "data/results/phase1/summaries/phase1_live_status.csv"},
        {"Key": "Pairwise CSV", "Value": "data/results/phase1/summaries/phase1_pairwise_common_prompt.csv"},
        {"Key": "Dashboard CSV", "Value": "data/results/phase1/summaries/phase1_dashboard_summary.csv"},
        {"Key": "Meaning", "Value": "Re-run the refresh command any time to rebuild this workbook from the latest results.json/final_summary.json files."},
    ]
    sheets = [
        ("README", readme_rows),
        ("Dashboard", dashboard_rows),
        ("Runs", status_rows),
        ("Pairwise", pairwise_rows),
    ]
    _write_xlsx(path, sheets)


def _write_xlsx(path: Path, sheets: list[tuple[str, list[dict]]]) -> None:
    sheet_defs = []
    for name, rows in sheets:
        headers = _headers_for_rows(rows)
        data_rows = [[row.get(header) for header in headers] for row in rows]
        sheet_defs.append({"name": name[:31], "headers": headers, "rows": data_rows})

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(len(sheet_defs)))
        zf.writestr("_rels/.rels", _root_rels_xml())
        zf.writestr("docProps/app.xml", _app_xml([sheet["name"] for sheet in sheet_defs]))
        zf.writestr("docProps/core.xml", _core_xml())
        zf.writestr("xl/workbook.xml", _workbook_xml(sheet_defs))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(sheet_defs)))
        zf.writestr("xl/styles.xml", _styles_xml())
        for idx, sheet in enumerate(sheet_defs, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(sheet["headers"], sheet["rows"]))


def _headers_for_rows(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    seen = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    return seen


def _worksheet_xml(headers: list[str], rows: list[list]) -> str:
    data = [headers] + rows
    last_col = _excel_col(max(1, len(headers)))
    last_row = max(1, len(data))

    width_xml = []
    for idx, header in enumerate(headers, start=1):
        max_len = len(str(header))
        for row in rows[:5000]:
            if idx - 1 < len(row):
                max_len = max(max_len, _display_len(row[idx - 1]))
        width = min(max(max_len + 2, 10), 40)
        width_xml.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')

    row_xml = []
    for r_idx, row in enumerate(data, start=1):
        cell_xml = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_excel_col(c_idx)}{r_idx}"
            style = "1" if r_idx == 1 else "0"
            cell = _cell_xml(ref, value, style=style)
            if cell:
                cell_xml.append(cell)
        row_xml.append(f'<row r="{r_idx}">{"".join(cell_xml)}</row>')

    sheet_views = (
        "<sheetViews><sheetView workbookViewId=\"0\">"
        "<pane ySplit=\"1\" topLeftCell=\"A2\" activePane=\"bottomLeft\" state=\"frozen\"/>"
        "</sheetView></sheetViews>"
    )
    auto_filter = f'<autoFilter ref="A1:{last_col}{last_row}"/>' if headers else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"{sheet_views}"
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"<cols>{''.join(width_xml)}</cols>"
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        f"{auto_filter}"
        "</worksheet>"
    )


def _cell_xml(ref: str, value, style: str = "0") -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return _inline_str_cell(ref, "TRUE" if value else "FALSE", style)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    return _inline_str_cell(ref, str(value), style)


def _inline_str_cell(ref: str, value: str, style: str) -> str:
    safe = escape(value)
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{safe}</t></is></c>'


def _content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}"
        "</Types>"
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def _app_xml(sheet_names: list[str]) -> str:
    parts = "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name in sheet_names)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Codex</Application>'
        f"<TitlesOfParts><vt:vector size=\"{len(sheet_names)}\" baseType=\"lpstr\">{parts}</vt:vector></TitlesOfParts>"
        f"<HeadingPairs><vt:vector size=\"2\" baseType=\"variant\"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>{len(sheet_names)}</vt:i4></vt:variant></vt:vector></HeadingPairs>"
        "</Properties>"
    )


def _core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>Phase 1 Dashboard</dc:title>'
        '<dc:creator>Codex</dc:creator>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def _workbook_xml(sheet_defs: list[dict]) -> str:
    sheets_xml = []
    for idx, sheet in enumerate(sheet_defs, start=1):
        sheets_xml.append(
            f'<sheet name="{escape(sheet["name"])}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<bookViews><workbookView xWindow="0" yWindow="0" windowWidth="24000" windowHeight="12000"/></bookViews>'
        f"<sheets>{''.join(sheets_xml)}</sheets>"
        "</workbook>"
    )


def _workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}"
        "</Relationships>"
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )


def _excel_col(index: int) -> str:
    result = []
    while index > 0:
        index, rem = divmod(index - 1, 26)
        result.append(chr(65 + rem))
    return "".join(reversed(result)) or "A"


def _display_len(value) -> int:
    if value is None:
        return 0
    return min(len(str(value).replace("\n", " ")), 80)


def _safe_ratio(num, den):
    num = _numeric(num)
    den = _numeric(den)
    if num is None or den in (None, 0):
        return None
    return num / den


def _product_or_none(a, b):
    a = _numeric(a)
    b = _numeric(b)
    if a is None or b is None:
        return None
    return a * b


def _numeric(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _dig(obj, *keys):
    cur = obj
    for key in keys:
        if cur is None:
            return None
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _method_rank(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def _sort_key(value):
    if value is None:
        return ""
    return str(value)


def _dashboard_key(key: tuple) -> tuple:
    mapping = dict(zip(COMPARISON_FIELDS, key))
    return (
        _sort_key(mapping.get("subphase")),
        _sort_key(mapping.get("dataset")),
        _sort_key(mapping.get("lm")),
        _sort_key(mapping.get("rm")),
        int(mapping.get("budget") or 0),
        _sort_key(mapping.get("config_label")),
    )


if __name__ == "__main__":
    main()
