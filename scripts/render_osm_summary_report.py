#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def fmt_int(value):
    return f"{value:,}"


def render_layer_section(layer):
    lines = []
    lines.append(f"## Layer: `{layer['name']}`")
    lines.append("")
    lines.append(f"- Geometry type: `{layer['geometry_type']}`")
    lines.append(f"- Feature count: `{fmt_int(layer['feature_count'])}`")

    derived = layer.get("derived_counts", {})
    if derived:
      lines.append(
          "- Derived counts: "
          + ", ".join(
              f"`{key}`={fmt_int(value)}" for key, value in sorted(derived.items())
          )
      )

    fields = layer.get("field_names", [])
    if fields:
        lines.append(f"- Fields: {', '.join(f'`{name}`' for name in fields)}")

    non_null = layer.get("non_null_field_counts", {})
    if non_null:
        top_fields = sorted(non_null.items(), key=lambda item: (-item[1], item[0]))[:15]
        lines.append("- Top non-null fields:")
        for key, count in top_fields:
            lines.append(f"  - `{key}`: {fmt_int(count)}")

    top_tags = layer.get("top_other_tag_keys", [])
    if top_tags:
        lines.append("- Top `other_tags` keys:")
        for entry in top_tags[:15]:
            lines.append(f"  - `{entry['key']}`: {fmt_int(entry['count'])}")

    samples = layer.get("samples", [])
    if samples:
        lines.append("- Sample features:")
        for index, sample in enumerate(samples[:3], start=1):
            prop_keys = ", ".join(sorted(sample.get("properties", {}).keys())[:20])
            lines.append(
                f"  - Sample {index}: `fid`={sample.get('fid')}, geometry=`{sample.get('geometry_summary', {}).get('geometry_type')}`, property keys={prop_keys}"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render a markdown report from an OSM JSON summary.")
    parser.add_argument("--input", required=True, help="Path to JSON summary")
    parser.add_argument("--output", required=True, help="Path to markdown output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    lines = []
    lines.append("# OSM Summary Report")
    lines.append("")
    lines.append(f"- Source: `{summary['input_path']}`")
    lines.append(f"- Driver: `{summary['driver']}`")
    lines.append(f"- Generated: `{summary['generated_at_utc']}`")
    lines.append(f"- Layer count: `{summary['layer_count']}`")
    lines.append(f"- Layer names: {', '.join(f'`{name}`' for name in summary.get('layer_names', []))}")
    lines.append(f"- Total features: `{fmt_int(summary['total_features'])}`")
    lines.append("")

    feature_counts = summary.get("feature_class_counts", {})
    if feature_counts:
        lines.append("## Feature Class Counts")
        lines.append("")
        for key, value in sorted(feature_counts.items()):
            lines.append(f"- `{key}`: {fmt_int(value)}")
        lines.append("")

    theme_counts = summary.get("derived_theme_counts", {})
    if theme_counts:
        lines.append("## Derived Theme Counts")
        lines.append("")
        for key, value in sorted(theme_counts.items()):
            lines.append(f"- `{key}`: {fmt_int(value)}")
        lines.append("")

    extra_layers = summary.get("extra_layers", [])
    if extra_layers:
        lines.append("## Extra GDAL Layers")
        lines.append("")
        for layer in extra_layers:
            lines.append(
                f"- `{layer['name']}`: geometry `{layer['geometry_type']}`, features `{fmt_int(layer['feature_count'])}`"
            )
        lines.append("")

    for layer in summary.get("layers", []):
        lines.append(render_layer_section(layer))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
