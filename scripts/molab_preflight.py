"""Read existing caches and audit the real single-rank planner; never train or cache."""

import argparse
import copy
import json
import math
import re
from collections import Counter
from pathlib import Path


def _summary():
    return {"seen_samples": 0, "microbatches": 0, "batch_size_distribution": Counter(),
            "caption_variants": Counter(), "planned_joint_tokens": 0,
            "actual_joint_tokens": 0, "text_dropped_samples": 0}


def _add(stats, batch, planned, actual, dropped):
    stats["seen_samples"] += len(batch)
    stats["microbatches"] += 1
    stats["batch_size_distribution"][str(len(batch))] += 1
    stats["caption_variants"].update(str(item.caption_variant) for item in batch)
    stats["planned_joint_tokens"] += planned
    stats["actual_joint_tokens"] += actual
    stats["text_dropped_samples"] += dropped


def _finish(stats):
    stats["mean_batch_size"] = stats["seen_samples"] / max(1, stats["microbatches"])
    for field in ("batch_size_distribution", "caption_variants"):
        stats[field] = dict(sorted(stats[field].items(), key=lambda pair: int(pair[0])))
    return stats


def audit_planner(planner, config, *, epochs=20, actual_tokens=None):
    """CPU-only core. Walk each real epoch, including partial configured-budget epochs.

    actual_tokens(batch, global_microbatch_index) returns (joint tokens, dropped count).
    Without it the exact planner token counts are the actual counts (no text dropout).
    Counts are exposures, not unique rows; epoch ordering and tail batches are retained.
    """
    grad_acc = config.get("GRAD_ACC", 1)
    steps = config.get("STEPS", 1000)
    if any(type(n) is not int for n in (epochs, grad_acc, steps)) or epochs < 1 or grad_acc < 1 or steps < 0:
        raise ValueError("epochs/GRAD_ACC must be positive integers and STEPS nonnegative")
    if config.get("PARALLEL", "none") != "none":
        raise ValueError("Preflight currently supports PARALLEL='none' only")
    target = steps * grad_acc
    total, budget = _summary(), _summary()
    epoch_stats, index, epoch = [], 0, 0
    completed, partial = 0, 0
    while epoch < epochs or index < target:
        batches = planner.epoch(epoch)
        if not batches:
            raise ValueError(f"planner produced an empty epoch: {epoch}")
        current = _summary()
        taken = 0
        for batch in batches:
            if epoch >= epochs and index >= target:
                break
            planned = sum(int(planner.tokens(item)) for item in batch)
            actual, dropped = actual_tokens(batch, index) if actual_tokens else (planned, 0)
            if epoch < epochs:
                _add(current, batch, planned, actual, dropped)
                _add(total, batch, planned, actual, dropped)
            if index < target:
                _add(budget, batch, planned, actual, dropped)
                taken += 1
            index += 1
        if taken == len(batches):
            completed += 1
        elif taken:
            partial = taken
        if epoch < epochs:
            current.update(epoch=epoch, overlong_dropped=int(getattr(planner, "drops", {}).get(epoch, 0)),
                           unbatched_samples=int(getattr(planner, "unbatched", {}).get(epoch, 0)))
            epoch_stats.append(_finish(current))
        epoch += 1
    _finish(total)
    _finish(budget)
    recommended_steps = math.ceil(total["microbatches"] / grad_acc)
    effective_batch = total["mean_batch_size"] * grad_acc
    save_every = config.get("SAVE_EVERY", 100)
    if type(save_every) is not int or save_every < 0:
        raise ValueError("SAVE_EVERY must be a nonnegative integer")
    budget.update(steps=steps, grad_acc=grad_acc, complete_epochs=completed,
                  microbatches_in_partial_epoch=partial)
    return {"schema_version": 1, "world_size": 1, "target_epochs": epochs,
            "configured": {key: copy.deepcopy(config.get(key)) for key in ("STEPS", "SAVE_EVERY", "LR", "GRAD_ACC")},
            "epochs": epoch_stats, "total": total, "configured_budget": budget,
            "checkpoints": {"save_every": save_every,
                "configured_periodic_count": steps // save_every if save_every else 0,
                "recommended_steps_periodic_count": recommended_steps // save_every if save_every else 0,
                "note": "Periodic step checkpoints only; final/time-based saves excluded. KEEP_LAST is local retention, not an upload limit."},
            "recommendations": {"steps": recommended_steps,
                "mean_effective_batch_size": effective_batch,
                "lr": 3e-4 * 9289728 / 74333952 * math.sqrt(effective_batch / 4),
                "lr_formula": "3e-4 * 9289728 / 74333952 * sqrt(mean_batch_size * GRAD_ACC / 4)",
                "applied": False}, "warnings": []}


def _spec_matches(spec, knobs, kind):
    """Check explicit compute choices; auto must use the recorded build provenance."""
    field = "TEXT_ENCODER_SPEC" if kind == "text" else "LATENT_ENCODER_SPEC"
    if field in knobs and knobs[field] != spec:
        return False
    parts = spec.split("/")
    plan = knobs.get("TEXT_ENCODE") or {}
    dtype = plan.get("dtype", "auto") if kind == "text" else knobs.get("VAE_DTYPE", "auto")
    if dtype != "auto" and parts[0] != dtype:
        return False
    if kind == "text":
        precision = plan.get("precision", "auto")
        if precision != "auto" and (len(parts) < 2 or parts[1] != precision):
            return False
        if precision in ("int8", "fp8") and f"g{plan.get('quant_group', 0)}" not in parts:
            return False
        if precision == "fp8_compute" and plan.get("fp8_recipe", "row") not in parts:
            return False
    return True


def build_dataset(knobs):
    """Use cached catalog + exact text lengths, with cache-recorded encoder identity.

    No trainer/model imports or hardware-dependent dtype inference. Auto encoder
    choices are explicitly conditional on the recorded cache build, not CPU defaults.
    """
    from kohakumoss_data.cache.settings import recorded, settings_of
    from kohakumoss_data.catalog.vault import is_catalog_vault
    from kohakumoss_data.config import catalog_cache_dir, resolved_cache_dir
    from kohakumoss_data.dataset import MossDataset, source_kwargs
    from kohakumoss_data.scan.paths import scan_path
    from kohakumoss_data.scripting import data_config

    cfg = data_config(copy.deepcopy(knobs))
    if cfg.text_cache != "offline" or cfg.token_budget != "exact":
        raise ValueError("Preflight requires TEXT_CACHE='offline' and TOKEN_BUDGET='exact'")
    if cfg.latent_cache != "offline":
        raise ValueError("Preflight requires LATENT_CACHE='offline' (no cache writes)")
    cache_dir = Path(resolved_cache_dir(cfg))
    if not cache_dir.is_dir() or not (cache_dir / "text.db").is_file():
        raise FileNotFoundError(f"Existing offline text cache required: {cache_dir / 'text.db'}")
    provenance = {}
    for kind, field in (("text", "text_encoder_spec"), ("latents", "latent_encoder_spec")):
        expected = settings_of(cfg, kind)
        candidates = [row for row in recorded(str(cache_dir), kind)
                      if all(row.get(key) == value for key, value in expected.items() if key != field)
                      and _spec_matches(row.get(field, ""), knobs, kind)]
        specs = {row.get(field, "") for row in candidates}
        if len(specs) != 1:
            raise ValueError(f"Need one matching {kind} cache compute spec, found {sorted(specs)}; check cache settings/config")
        setattr(cfg, field, specs.pop())
        provenance[kind] = candidates[0]
    # Bind existing catalogs explicitly: no source walk, image probe, rescan, or wait.
    for entry in cfg.datasets:
        path = Path(entry.get("catalog") or scan_path(catalog_cache_dir(cfg), source_kwargs(entry)))
        if not path.is_file():
            raise FileNotFoundError(f"Existing catalog required (build caches first): {path}")
        if not is_catalog_vault(str(path)):
            raise ValueError(f"Catalog must be an existing catalog vault (refusing scan fallback): {path}")
        entry["catalog"] = str(path)
    cfg.data_rescan = "never"
    cfg.data_wait = 0
    return MossDataset(cfg, writer=False), provenance


def artist_prefix_stats(dataset):
    """Inspect the actual rendered variants, not optional catalog artist metadata."""
    prefix = re.compile(r"\A\s*<\|artist\|>([^<>]*)<\|/artist\|>")
    missing, partial, absent = [], [], Counter()
    live_rows = [i for i, live in enumerate(dataset.catalog.live) if live]
    for row_id in live_rows:
        present = []
        for variant in range(dataset.cfg.caption_variants):
            match = prefix.match(dataset.caption(row_id, variant))
            valid = bool(match and match.group(1).strip())
            present.append(valid)
            if not valid:
                absent[str(variant)] += 1
        if not any(present):
            missing.append(row_id)
        elif not all(present):
            partial.append(row_id)
    return {"live_rows": len(live_rows), "artist_detection": "rendered_caption_prefix",
            "missing_artist_rows": len(missing), "missing_artist_row_examples": missing[:32],
            "partial_artist_rows": len(partial), "partial_artist_row_examples": partial[:32],
            "missing_artist_by_variant": {str(i): absent[str(i)] for i in range(dataset.cfg.caption_variants)}}


def audit_config(knobs, *, epochs=20):
    dataset, provenance = build_dataset(knobs)
    empty_length = None
    if dataset.cfg.text_drop:
        empty_length = dataset._text_lengths.get(dataset.texts.key(""))
        if empty_length is None:
            raise ValueError("TEXT_DROP requires the existing empty-prompt text cache entry")

    def actual_tokens(batch, index):
        total, dropped = 0, 0
        for slot, item in enumerate(batch):
            image, text = dataset.planner.sizes(item)
            if dataset._drop(index, 0, slot):
                text = min(empty_length, item.text_limit or empty_length)
                dropped += 1
            total += int(image + text)
        return total, dropped

    report = audit_planner(dataset.planner, knobs, epochs=epochs, actual_tokens=actual_tokens)
    report["catalog"] = artist_prefix_stats(dataset)
    report["cache_provenance"] = provenance
    report["caption_variant_count"] = dataset.cfg.caption_variants
    report["caption_variant_labels"] = {"0": "short", "1": "medium", "2": "long"} if dataset.cfg.caption.get("template") == "options" else {}
    report["warnings"].append("Encoder auto choices use matching recorded cache compute specs; this does not independently resolve GPU/model auto precision. No model was loaded.")
    missing = report["catalog"]["missing_artist_rows"]
    partial = report["catalog"]["partial_artist_rows"]
    if missing:
        report["warnings"].append(f"{missing} live catalog rows lack a valid artist prefix in every rendered caption variant")
    if partial:
        report["warnings"].append(f"{partial} live catalog rows have an artist prefix in only some rendered caption variants")
    if dataset.cfg.caption_variants != 3:
        report["warnings"].append(f"Expected three caption variants; configured {dataset.cfg.caption_variants}")
    report["warnings"].append("Exposure budget starts at microbatch 0 (fresh run), not a resumed checkpoint; recommendations do not change the config.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.config.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --config")
    knobs = json.loads(args.config.read_text(encoding="utf-8"))
    report = audit_config(knobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "recommendations": report["recommendations"]}))
    return report


if __name__ == "__main__":
    main()
