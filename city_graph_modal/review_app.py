"""Modal entrypoint for qualitative reconstruction review."""

from __future__ import annotations

import json
from pathlib import Path

import modal

try:
    from .modal_app import DATASET_REL, RUN_REL, _configure_logging, _data_path, _runs_path, data_volume, image, runs_volume
except ImportError:  # pragma: no cover
    from city_graph_modal.modal_app import (
        DATASET_REL,
        RUN_REL,
        _configure_logging,
        _data_path,
        _runs_path,
        data_volume,
        image,
        runs_volume,
    )


REVIEW_APP_NAME = "bonzai-city-graph-review"

app = modal.App(REVIEW_APP_NAME)


def _html_shell(payload_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>City Graph Review</title>
  <style>
    :root {{
      --bg: #f3f1ea;
      --panel: #fffdf8;
      --ink: #1d1d1b;
      --muted: #6d675f;
      --line: #d8d1c7;
      --accent: #0b5fff;
      --good: #18864b;
      --bad: #c53a2f;
      --warn: #d98f00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #f7f4ec 0%, #efe9dc 100%);
      color: var(--ink);
    }}
    header {{
      padding: 24px 28px 12px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
      background: rgba(255,255,255,0.58);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0.02em;
    }}
    .sub {{
      color: var(--muted);
      max-width: 900px;
      line-height: 1.45;
      font-size: 15px;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    select, button {{
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 999px;
      font: inherit;
      cursor: pointer;
    }}
    main {{
      padding: 24px 28px 36px;
      display: grid;
      gap: 18px;
    }}
    .summary, .details, .legend {{
      background: rgba(255,253,248,0.9);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 20px;
      padding: 18px 20px;
      box-shadow: 0 10px 30px rgba(20, 12, 4, 0.06);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .metric {{
      padding: 12px 14px;
      border-radius: 14px;
      background: #f7f2e8;
    }}
    .metric .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric .v {{
      font-size: 24px;
      margin-top: 6px;
    }}
    .canvas-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
    }}
    .canvas-card {{
      background: rgba(255,253,248,0.92);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 24px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(20, 12, 4, 0.06);
    }}
    .canvas-title {{
      margin: 4px 4px 12px;
      font-size: 14px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    canvas {{
      width: 100%;
      aspect-ratio: 1 / 1;
      background: linear-gradient(180deg, #fffefb 0%, #f8f3e8 100%);
      border-radius: 18px;
      border: 1px solid rgba(0,0,0,0.08);
      display: block;
    }}
    .legend-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: middle;
    }}
    .details pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
      line-height: 1.45;
      color: #2c2926;
    }}
  </style>
</head>
<body>
  <header>
    <h1>City Graph Reconstruction Review</h1>
    <div class="sub">
      This viewer shows what the current model can actually do: it sees a partially masked city chunk and tries to reconstruct missing node labels and missing typed edges.
      It is not a free-running city generator yet. Click a node in any panel to inspect its attributes and the model's reconstruction.
    </div>
    <div class="toolbar">
      <button id="prev-btn">Previous</button>
      <select id="sample-select"></select>
      <button id="next-btn">Next</button>
    </div>
  </header>
  <main>
    <section class="summary">
      <div id="sample-headline"></div>
      <div class="summary-grid" id="summary-grid"></div>
    </section>
    <section class="canvas-grid">
      <div class="canvas-card">
        <div class="canvas-title">Original Chunk</div>
        <canvas id="original-canvas" width="720" height="720"></canvas>
      </div>
      <div class="canvas-card">
        <div class="canvas-title">Masked Input</div>
        <canvas id="masked-canvas" width="720" height="720"></canvas>
      </div>
      <div class="canvas-card">
        <div class="canvas-title">Model Reconstruction</div>
        <canvas id="predicted-canvas" width="720" height="720"></canvas>
      </div>
    </section>
    <section class="legend">
      <strong>Legend</strong>
      <div class="legend-grid">
        <div><span class="swatch" style="background:#274c77"></span>Road junction</div>
        <div><span class="swatch" style="background:#0c7c59"></span>Road segment</div>
        <div><span class="swatch" style="background:#d64933"></span>Building</div>
        <div><span class="swatch" style="background:#f4a259"></span>POI</div>
        <div><span class="swatch" style="background:#8d6a9f"></span>Landuse</div>
        <div><span class="swatch" style="background:#d98f00"></span>Masked node/edge</div>
        <div><span class="swatch" style="background:#18864b"></span>Recovered correctly</div>
        <div><span class="swatch" style="background:#c53a2f"></span>Recovered incorrectly</div>
      </div>
    </section>
    <section class="details">
      <strong>Node Details</strong>
      <pre id="details-box">Click a node to inspect the original attributes and the model's prediction.</pre>
    </section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const samples = payload.samples || [];
    const sampleSelect = document.getElementById("sample-select");
    const summaryGrid = document.getElementById("summary-grid");
    const sampleHeadline = document.getElementById("sample-headline");
    const detailsBox = document.getElementById("details-box");

    const canvases = {{
      original: document.getElementById("original-canvas"),
      masked: document.getElementById("masked-canvas"),
      predicted: document.getElementById("predicted-canvas"),
    }};

    const state = {{ sampleIndex: 0 }};

    const nodeColors = {{
      ROAD_JUNCTION: "#274c77",
      ROAD_SEGMENT: "#0c7c59",
      BUILDING: "#d64933",
      POI: "#f4a259",
      LANDUSE: "#8d6a9f",
    }};

    const relationColors = {{
      SEGMENT_CONNECTS_JUNCTION: "rgba(12,124,89,0.55)",
      JUNCTION_ADJACENT_JUNCTION: "rgba(39,76,119,0.35)",
      BUILDING_NEAR_SEGMENT: "rgba(214,73,51,0.30)",
      POI_NEAR_SEGMENT: "rgba(244,162,89,0.38)",
      BUILDING_INSIDE_LANDUSE: "rgba(141,106,159,0.45)",
      POI_INSIDE_LANDUSE: "rgba(141,106,159,0.28)",
    }};

    function shortLabel(text) {{
      if (!text) return "";
      return text.length > 12 ? text.slice(0, 12) + "…" : text;
    }}

    function nodeRadius(node) {{
      if (node.type === "ROAD_JUNCTION") return 3.5;
      if (node.type === "POI") return 4.5;
      return Math.max(4, Math.min(8, 3 + node.size_log1p * 0.35));
    }}

    function panelTransform(canvas, node) {{
      const pad = 40;
      const width = canvas.width - pad * 2;
      const height = canvas.height - pad * 2;
      return {{
        x: pad + node.x * width,
        y: pad + (1 - node.y) * height,
      }};
    }}

    function drawEdge(ctx, canvas, sample, edge, style) {{
      const src = sample.nodes[edge.src];
      const dst = sample.nodes[edge.dst];
      if (!src || !dst) return;
      const a = panelTransform(canvas, src);
      const b = panelTransform(canvas, dst);
      ctx.save();
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.width || 1.6;
      if (style.dash) ctx.setLineDash(style.dash);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    }}

    function drawNodes(ctx, canvas, sample, mode) {{
      const hitNodes = [];
      for (const node of sample.nodes) {{
        const p = panelTransform(canvas, node);
        const r = nodeRadius(node);
        ctx.save();
        ctx.fillStyle = nodeColors[node.type] || "#333";
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();

        if (mode === "masked" && node.display_field && node.display_observed === null && node.display_true !== null) {{
          ctx.strokeStyle = "#d98f00";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(p.x, p.y, r + 3, 0, Math.PI * 2);
          ctx.stroke();
        }}
        if (mode === "predicted" && node.display_field && node.display_observed === null && node.display_true !== null) {{
          const ok = node.display_predicted === node.display_true;
          ctx.strokeStyle = ok ? "#18864b" : "#c53a2f";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(p.x, p.y, r + 3, 0, Math.PI * 2);
          ctx.stroke();
        }}
        ctx.restore();

        let label = "";
        if (mode === "masked" && node.display_field && node.display_observed === null) {{
          label = "?";
        }} else if (mode === "predicted" && node.display_field && node.display_observed === null) {{
          label = shortLabel(node.display_predicted || "none");
        }} else if (mode === "original" && node.display_field && node.display_true && node.masked_fields.length > 0) {{
          label = shortLabel(node.display_true);
        }}
        if (label) {{
          ctx.save();
          ctx.font = "12px Georgia";
          ctx.fillStyle = "#201d1a";
          ctx.fillText(label, p.x + 7, p.y - 7);
          ctx.restore();
        }}

        hitNodes.push({{ x: p.x, y: p.y, r: r + 6, node }});
      }}
      canvas.__hitNodes = hitNodes;
    }}

    function drawPanel(canvas, sample, mode) {{
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fffefb";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (mode === "original") {{
        for (const edge of sample.original_edges) {{
          drawEdge(ctx, canvas, sample, edge, {{ color: relationColors[edge.relation] || "rgba(0,0,0,0.22)", width: 1.5 }});
        }}
      }} else if (mode === "masked") {{
        for (const edge of sample.observed_edges) {{
          drawEdge(ctx, canvas, sample, edge, {{ color: "rgba(68,82,96,0.22)", width: 1.4 }});
        }}
        for (const edge of sample.masked_edges) {{
          drawEdge(ctx, canvas, sample, {{
            src: edge.src,
            dst: edge.dst,
          }}, {{ color: "rgba(217,143,0,0.85)", width: 2.0, dash: [7, 6] }});
        }}
      }} else if (mode === "predicted") {{
        for (const edge of sample.observed_edges) {{
          drawEdge(ctx, canvas, sample, edge, {{ color: "rgba(68,82,96,0.18)", width: 1.3 }});
        }}
        for (const edge of sample.masked_edges) {{
          const color = edge.recovered ? "rgba(24,134,75,0.90)" : "rgba(197,58,47,0.88)";
          const dash = edge.predicted_relation ? [] : [9, 6];
          drawEdge(ctx, canvas, sample, {{
            src: edge.src,
            dst: edge.dst,
          }}, {{ color, width: 2.4, dash }});
        }}
      }}

      drawNodes(ctx, canvas, sample, mode);
    }}

    function renderSummary(sample) {{
      sampleHeadline.innerHTML = `
        <strong>${{sample.chunk_id}}</strong><br />
        <span style="color:#6d675f">Checkpoint review for split "${{payload.split}}" on ${{payload.device}}. This chunk is the original city fragment, the masked input, and the reconstruction attempt.</span>
      `;
      const metrics = [
        ["Nodes", sample.node_count],
        ["Original edges", sample.original_edge_count],
        ["Observed edges", sample.observed_edge_count],
        ["Masked edges", sample.masked_edge_count],
        ["Recovered edges", sample.recovered_edge_count],
        ["Masked nodes", sample.masked_node_count],
      ];
      summaryGrid.innerHTML = metrics.map(([k, v]) => `
        <div class="metric">
          <div class="k">${{k}}</div>
          <div class="v">${{v}}</div>
        </div>
      `).join("");
    }}

    function renderSample() {{
      const sample = samples[state.sampleIndex];
      if (!sample) return;
      sampleSelect.value = String(state.sampleIndex);
      renderSummary(sample);
      drawPanel(canvases.original, sample, "original");
      drawPanel(canvases.masked, sample, "masked");
      drawPanel(canvases.predicted, sample, "predicted");
      detailsBox.textContent = "Click a node to inspect the original attributes and the model's prediction.";
    }}

    function setDetails(node, panelName) {{
      const summary = {{
        panel: panelName,
        node_type: node.type,
        display_field: node.display_field,
        display_true: node.display_true,
        display_observed: node.display_observed,
        display_predicted: node.display_predicted,
        masked_fields: node.masked_fields,
        true_attrs: node.true_attrs,
        predicted_attrs: node.predicted_attrs,
      }};
      detailsBox.textContent = JSON.stringify(summary, null, 2);
    }}

    function attachCanvasHandlers() {{
      for (const [panelName, canvas] of Object.entries(canvases)) {{
        canvas.addEventListener("click", (event) => {{
          const rect = canvas.getBoundingClientRect();
          const scaleX = canvas.width / rect.width;
          const scaleY = canvas.height / rect.height;
          const x = (event.clientX - rect.left) * scaleX;
          const y = (event.clientY - rect.top) * scaleY;
          const hits = canvas.__hitNodes || [];
          let found = null;
          let best = Infinity;
          for (const hit of hits) {{
            const dx = hit.x - x;
            const dy = hit.y - y;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d <= hit.r && d < best) {{
              found = hit.node;
              best = d;
            }}
          }}
          if (found) setDetails(found, panelName);
        }});
      }}
    }}

    function init() {{
      samples.forEach((sample, idx) => {{
        const option = document.createElement("option");
        option.value = String(idx);
        option.textContent = `${{idx + 1}}. ${{sample.chunk_id}}`;
        sampleSelect.appendChild(option);
      }});
      sampleSelect.addEventListener("change", () => {{
        state.sampleIndex = Number(sampleSelect.value);
        renderSample();
      }});
      document.getElementById("prev-btn").addEventListener("click", () => {{
        state.sampleIndex = (state.sampleIndex - 1 + samples.length) % samples.length;
        renderSample();
      }});
      document.getElementById("next-btn").addEventListener("click", () => {{
        state.sampleIndex = (state.sampleIndex + 1) % samples.length;
        renderSample();
      }});
      attachCanvasHandlers();
      renderSample();
    }}

    init();
  </script>
</body>
</html>"""


def _write_review_files(payload: dict, output_dir: str) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "city_graph_review.json"
    html_path = out_dir / "city_graph_review.html"

    payload_json = json.dumps(payload, indent=2).replace("</", "<\\/")
    json_path.write_text(payload_json, encoding="utf-8")
    html_path.write_text(_html_shell(payload_json), encoding="utf-8")

    return {
        "json_path": str(json_path),
        "html_path": str(html_path),
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={"/vol/data": data_volume.read_only(), "/vol/runs": runs_volume.read_only()},
)
def build_reconstruction_review_remote(
    dataset_rel_dir: str = DATASET_REL,
    run_rel_dir: str = RUN_REL,
    split: str = "test",
    checkpoint_name: str = "best.pt",
    sample_count: int = 4,
) -> dict:
    logger = _configure_logging()
    try:
        from .review import build_review_payload
    except ImportError:  # pragma: no cover
        from city_graph_modal.review import build_review_payload

    checkpoint_path = Path(_runs_path(run_rel_dir)) / "checkpoints" / checkpoint_name
    logger.info(
        "building reconstruction review dataset=%s checkpoint=%s split=%s sample_count=%d",
        _data_path(dataset_rel_dir),
        checkpoint_path,
        split,
        sample_count,
    )
    payload = build_review_payload(
        dataset_root=_data_path(dataset_rel_dir),
        checkpoint_path=str(checkpoint_path),
        split=split,
        sample_count=sample_count,
    )
    logger.info("built reconstruction review with %d samples", payload["sample_count"])
    return payload


@app.local_entrypoint()
def review(
    dataset_rel_dir: str = DATASET_REL,
    run_rel_dir: str = RUN_REL,
    split: str = "test",
    checkpoint_name: str = "best.pt",
    sample_count: int = 4,
    output_dir: str = "artifacts",
) -> None:
    payload = build_reconstruction_review_remote.remote(
        dataset_rel_dir=dataset_rel_dir,
        run_rel_dir=run_rel_dir,
        split=split,
        checkpoint_name=checkpoint_name,
        sample_count=sample_count,
    )
    paths = _write_review_files(payload, output_dir=output_dir)
    print("review_json:", paths["json_path"])
    print("review_html:", paths["html_path"])
