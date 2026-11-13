#!/usr/bin/env python
"""Generate the markdown API reference for the just-the-docs documentation site.

Statically analyses ``nrcats`` with griffe (no runtime imports, so heavy
dependencies like lalsuite/pycbc/sxs are NOT required) and renders one markdown
page per module into ``docs/api/``, each with just-the-docs front matter.

Usage (from the repository root)::

    python bin/generate_api_docs.py            # write docs/api/*.md
    python bin/generate_api_docs.py --check    # exit 1 if output is stale

Requires only ``griffe`` (and the repository checkout itself).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import griffe

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "api"

# (module path relative to the package, page title, one-line summary)
# Order determines nav_order under "API Reference".
MODULES = [
    (
        "catalog",
        "nrcats.catalog",
        "Abstract base classes `CatalogABC` and `CatalogBase`",
    ),
    (
        "rit",
        "nrcats.rit",
        "`RITCatalog` and `RITCatalogHelper` for the RIT catalog",
    ),
    ("sxs", "nrcats.sxs", "`SXSCatalog` wrapping the `sxs` package"),
    ("maya", "nrcats.maya", "`MayaCatalog` for the Georgia Tech MAYA catalog"),
    (
        "waveform.modes",
        "nrcats.waveform.modes",
        "The central `WaveformModes` object",
    ),
    (
        "waveform.matching",
        "nrcats.waveform.matching",
        "Mode matching, rotation, and PSD helpers",
    ),
    (
        "waveform.loaders",
        "nrcats.waveform.loaders",
        "HDF5 / tar.gz loaders for `WaveformModes`",
    ),
    (
        "waveform.units",
        "nrcats.waveform.units",
        "Waveform-level constants and time-step helper",
    ),
    (
        "surrogate",
        "nrcats.surrogate",
        "NRSur7dq4, NRSur7dq4v2 and NRHJSur3dq8 loading, evaluation, and prior check",
    ),
    (
        "comparisons",
        "nrcats.comparisons",
        "End-to-end NR vs surrogate comparison pipeline",
    ),
    (
        "classification",
        "nrcats.classification",
        "Spin/eccentricity classification of catalogs",
    ),
    (
        "metadata",
        "nrcats.metadata",
        "Cross-catalog key mappings and parameter extraction",
    ),
    ("registry", "nrcats.registry", "Catalog plugin registry"),
    ("lvc", "nrcats.lvc", "Frame-rotation helpers and LVCNR format utilities"),
    (
        "utils",
        "nrcats.utils",
        "Cache paths, download helpers, unit conversions",
    ),
    (
        "waveform.frame",
        "nrcats.waveform.frame",
        "BMS frame operations a catalog records, and the coprecessing frame",
    ),
    (
        "waveform.detectors",
        "nrcats.waveform.detectors",
        "Observatory registry: noise curves, bands, PSD resolution",
    ),
]

# ── docstring section rendering ──────────────────────────────────────────────


def _kind(section) -> str:
    return getattr(section.kind, "value", str(section.kind))


def _clean(text: str | None) -> str:
    """Collapse a docstring description for use inside a markdown table cell."""
    if not text:
        return ""
    return " ".join(text.split()).replace("|", "\\|")


def _field_table(rows, header=("Name", "Type", "Description")) -> list[str]:
    lines = [f"| {' | '.join(header)} |", f"|{'---|' * len(header)}"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    lines.append("")
    return lines


def render_sections(docstring, level: int = 4) -> list[str]:
    """Render a parsed griffe docstring into markdown lines."""
    if docstring is None:
        return ["*No docstring.*", ""]
    out: list[str] = []
    h = "#" * level
    for section in docstring.parse("auto"):
        kind = _kind(section)
        if kind == "text":
            out += [section.value, ""]
        elif kind in ("parameters", "other parameters"):
            title = "Parameters" if kind == "parameters" else "Other parameters"
            rows = [
                (
                    f"`{p.name}`",
                    f"`{p.annotation}`" if p.annotation else "",
                    _clean(p.description),
                )
                for p in section.value
            ]
            out += [f"{h} {title}", ""] + _field_table(rows)
        elif kind in ("returns", "yields"):
            rows = [
                (
                    f"`{r.name}`" if r.name else "",
                    f"`{r.annotation}`" if r.annotation else "",
                    _clean(r.description),
                )
                for r in section.value
            ]
            out += [f"{h} {kind.capitalize()}", ""]
            out += _field_table(rows, header=("Name", "Type", "Description"))
        elif kind == "raises":
            rows = [
                (f"`{r.annotation}`" if r.annotation else "", _clean(r.description))
                for r in section.value
            ]
            out += [f"{h} Raises", ""] + _field_table(
                rows, header=("Exception", "Condition")
            )
        elif kind == "attributes":
            rows = [
                (
                    f"`{a.name}`",
                    f"`{a.annotation}`" if a.annotation else "",
                    _clean(a.description),
                )
                for a in section.value
            ]
            out += [f"{h} Attributes", ""] + _field_table(rows)
        elif kind == "examples":
            out += [f"{h} Examples", ""]
            for item_kind, text in section.value:
                if getattr(item_kind, "value", str(item_kind)) == "examples":
                    out += ["```python", text, "```", ""]
                else:
                    out += [text, ""]
        elif kind in ("notes", "warnings", "deprecated"):
            css = {"notes": "note", "warnings": "warning", "deprecated": "warning"}[
                kind
            ]
            body = (
                section.value if isinstance(section.value, str) else str(section.value)
            )
            out += [f"{{: .{css} }}", "> " + body.replace("\n", "\n> "), ""]
        elif kind == "admonition":
            out += [
                f"> **{section.title}**",
                "> " + str(section.value.description or "").replace("\n", "\n> "),
                "",
            ]
        elif kind in ("classes", "functions", "modules"):
            # numpy-style listing sections (e.g. a module docstring's
            # "Classes" section): items have .name and .description
            rows = [
                (f"`{item.name}`", _clean(item.description)) for item in section.value
            ]
            out += [f"{h} {kind.capitalize()}", ""]
            out += _field_table(rows, header=("Name", "Description"))
        else:  # fall back to plain text for anything unexpected
            out += [str(section.value), ""]
    return out


# ── signatures ───────────────────────────────────────────────────────────────


def format_signature(func, is_method: bool = False) -> str:
    parts = []
    for i, p in enumerate(func.parameters):
        if i == 0 and is_method and p.name in ("self", "cls"):
            continue
        # The enum member's name, not str(): griffe 2.3 stringifies
        # ParameterKind to its value ("variadic keyword"), which a substring
        # test on "var_keyword" misses, rendering **kwargs as kwargs={}.
        kind = getattr(getattr(p, "kind", None), "name", "")
        if kind == "var_positional":
            parts.append(f"*{p.name}")
            continue
        if kind == "var_keyword":
            parts.append(f"**{p.name}")
            continue
        name = p.name
        if p.annotation and str(p.annotation) != "None":
            name += f": {p.annotation}"
        if p.default is not None:
            name += f" = {p.default}" if ":" in name else f"={p.default}"
        parts.append(name)
    sig = f"{func.name}({', '.join(parts)})"
    if func.returns and str(func.returns) != "None":
        sig += f" -> {func.returns}"
    return sig


# ── member selection ─────────────────────────────────────────────────────────


def public_members(obj):
    """Yield public, locally-defined members in source order."""
    members = [
        m for m in obj.members.values() if not m.name.startswith("_") and not m.is_alias
    ]
    members.sort(key=lambda m: (m.lineno or 0))
    return members


# ── page rendering ───────────────────────────────────────────────────────────


def render_function(func, level: int = 3, is_method: bool = False) -> list[str]:
    h = "#" * level
    labels = getattr(func, "labels", set()) or set()
    qualifier = ""
    if "classmethod" in labels:
        qualifier = "*classmethod* "
    elif "staticmethod" in labels:
        qualifier = "*staticmethod* "
    lines = [f"{h} {qualifier}`{func.name}`", ""]
    lines += ["```python", format_signature(func, is_method=is_method), "```", ""]
    lines += render_sections(func.docstring, level=level + 1)
    lines.append("---")
    lines.append("")
    return lines


def render_attribute(attr, level: int = 3) -> list[str]:
    h = "#" * level
    labels = getattr(attr, "labels", set()) or set()
    qualifier = "*property* " if "property" in labels else ""
    lines = [f"{h} {qualifier}`{attr.name}`", ""]
    if attr.annotation and "property" not in labels:
        lines += [f"Type: `{attr.annotation}`", ""]
    if attr.docstring:
        lines += render_sections(attr.docstring, level=level + 1)
    lines += ["---", ""]
    return lines


def render_class(cls) -> list[str]:
    bases = ", ".join(f"`{b}`" for b in cls.bases) if cls.bases else ""
    lines = [f"## *class* `{cls.name}`", ""]
    if bases:
        lines += [f"Bases: {bases}", ""]
    lines += render_sections(cls.docstring, level=4)
    lines += [""]
    for member in public_members(cls):
        if member.is_function:
            lines += render_function(member, level=3, is_method=True)
        elif member.is_attribute and member.docstring:
            lines += render_attribute(member, level=3)
        elif member.is_class:
            # nested classes: rare; render heading + docstring only
            lines += [f"### `{member.name}`", ""]
            lines += render_sections(member.docstring, level=4)
    return lines


def render_module_page(pkg, module_path: str, title: str, nav_order: int) -> str:
    mod = pkg[module_path]
    lines = [
        "---",
        f"title: {title}",
        "parent: API Reference",
        f"nav_order: {nav_order}",
        "---",
        "",
        "<!-- GENERATED FILE — DO NOT EDIT."
        " Regenerate with `python bin/generate_api_docs.py`. -->",
        "{% raw %}",
        "",
        f"# `{title}`",
        "",
    ]
    lines += render_sections(mod.docstring, level=3)
    lines += [
        "",
        "## Contents",
        "{: .no_toc .text-delta }",
        "",
        "1. TOC",
        "{:toc}",
        "",
        "---",
        "",
    ]
    classes = [m for m in public_members(mod) if m.is_class]
    functions = [m for m in public_members(mod) if m.is_function]
    attributes = [m for m in public_members(mod) if m.is_attribute and m.docstring]
    constants = [m for m in public_members(mod) if m.is_attribute and not m.docstring]

    if constants:
        lines += ["## Constants", ""]
        lines += _field_table(
            [(f"`{c.name}`", f"`{_clean(str(c.value))}`") for c in constants],
            header=("Name", "Value"),
        )
        lines += ["---", ""]
    for attr in attributes:
        lines += render_attribute(attr, level=3)
    for cls in classes:
        lines += render_class(cls)
    if functions:
        if classes:
            lines += ["## Functions", ""]
        for func in functions:
            lines += render_function(func, level=3)

    lines += ["{% endraw %}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if any output file is missing or stale.",
    )
    args = parser.parse_args()

    pkg = griffe.load("nrcats", search_paths=[str(REPO_ROOT)])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stale = []
    for nav_order, (module_path, title, _summary) in enumerate(MODULES, start=2):
        slug = module_path.replace(".", "_")
        out_file = OUTPUT_DIR / f"{slug}.md"
        content = render_module_page(pkg, module_path, title, nav_order)
        if args.check:
            if not out_file.exists() or out_file.read_text() != content:
                stale.append(out_file)
        else:
            out_file.write_text(content)
            print(f"wrote {out_file.relative_to(REPO_ROOT)}")

    if args.check and stale:
        print("STALE API docs (run `python bin/generate_api_docs.py`):")
        for f in stale:
            print(f"  {f.relative_to(REPO_ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
