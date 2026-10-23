#!/usr/bin/env python3

import json
import math
import random
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

try:
    import numpy as np
except Exception:
    np = None
NUMPY_AVAILABLE = np is not None

try:
    from sklearn.manifold import TSNE as SklearnTSNE
except Exception:
    SklearnTSNE = None

HOST = "127.0.0.1"
PORT = 8776
ANNOTATION_STATE_JSON = Path(__file__).resolve().parent / "state" / "alignment_annotations.json"


def _hbeta(dist_row, beta):
    p = np.exp(-dist_row * beta)
    s = np.sum(p)
    if s <= 1e-12:
        p = np.ones_like(dist_row) / max(1, len(dist_row))
        h = 0.0
    else:
        p = p / s
        h = np.log(s) + beta * np.sum(dist_row * p)
    return h, p


def _x2p(x, perplexity=30.0, tol=1e-5):
    n = x.shape[0]
    sum_x = np.sum(np.square(x), axis=1)
    d = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    p = np.zeros((n, n), dtype=np.float64)
    beta = np.ones((n, 1), dtype=np.float64)
    log_u = np.log(perplexity)
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        mask = np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))
        di = d[i, mask]
        h, this_p = _hbeta(di, beta[i])
        hdiff = h - log_u
        tries = 0
        while np.abs(hdiff) > tol and tries < 50:
            if hdiff > 0:
                betamin = beta[i].copy()
                beta[i] = beta[i] * 2.0 if np.isinf(betamax) else (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                beta[i] = beta[i] / 2.0 if np.isinf(betamin) else (beta[i] + betamin) / 2.0
            h, this_p = _hbeta(di, beta[i])
            hdiff = h - log_u
            tries += 1
        p[i, mask] = this_p
    return p


def _pure_python_pca_embedding(matrix, random_state=42):
    rows = [[float(value) for value in row] for row in matrix if isinstance(row, list)]
    if len(rows) < 3:
        raise ValueError("Need at least 3 rows for dependency-free embedding.")
    dims = len(rows[0]) if rows else 0
    rows = [row for row in rows if len(row) == dims and dims > 0]
    if len(rows) < 3:
        raise ValueError("Need at least 3 valid rows for dependency-free embedding.")
    n = len(rows)
    means = [sum(row[j] for row in rows) / n for j in range(dims)]
    centered = [[row[j] - means[j] for j in range(dims)] for row in rows]

    cov = [[0.0 for _ in range(dims)] for _ in range(dims)]
    denom = max(1, n - 1)
    for row in centered:
        for i in range(dims):
            for j in range(i, dims):
                cov[i][j] += row[i] * row[j] / denom
    for i in range(dims):
        for j in range(i):
            cov[i][j] = cov[j][i]

    def mat_vec(mat, vec):
        return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

    def norm(vec):
        return math.sqrt(sum(value * value for value in vec))

    def power_component(mat, seed_offset=0):
        rng = random.Random(int(random_state) + seed_offset)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
        length = norm(vec) or 1.0
        vec = [value / length for value in vec]
        for _ in range(80):
            nxt = mat_vec(mat, vec)
            length = norm(nxt)
            if length <= 1e-12:
                break
            vec = [value / length for value in nxt]
        mv = mat_vec(mat, vec)
        eigenvalue = sum(vec[i] * mv[i] for i in range(dims))
        return vec, max(0.0, eigenvalue)

    v1, e1 = power_component(cov, 1)
    cov2 = [[cov[i][j] - e1 * v1[i] * v1[j] for j in range(dims)] for i in range(dims)]
    v2, _ = power_component(cov2, 2) if dims > 1 else ([0.0 for _ in range(dims)], 0.0)
    y = [[sum(row[j] * v1[j] for j in range(dims)), sum(row[j] * v2[j] for j in range(dims))] for row in centered]
    flat = [value for row in y for value in row]
    mean_y = sum(flat) / max(1, len(flat))
    variance = sum((value - mean_y) ** 2 for value in flat) / max(1, len(flat))
    scale = math.sqrt(variance) or 1.0
    return [[row[0] / scale, row[1] / scale] for row in y]


def fallback_tsne_embedding(matrix, perplexity=30.0, max_iter=500, random_state=42):
    if np is None:
        return _pure_python_pca_embedding(matrix, random_state=random_state)
    x = np.asarray(matrix, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 rows for t-SNE.")
    x = x - np.mean(x, axis=0, keepdims=True)
    # PCA init to 2D.
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:2].T if vt.shape[0] >= 2 else np.c_[x[:, 0], np.zeros(n)]
    y = y / (np.std(y) + 1e-8)
    rng = np.random.default_rng(random_state)
    y = y + rng.normal(0, 1e-4, size=y.shape)

    p = _x2p(x, perplexity=perplexity)
    p = (p + p.T) / (2.0 * n)
    p = np.maximum(p * 4.0, 1e-12)  # early exaggeration

    gains = np.ones_like(y)
    y_inc = np.zeros_like(y)
    momentum = 0.5
    final_momentum = 0.8
    eta = 200.0

    for it in range(max_iter):
        sum_y = np.sum(np.square(y), axis=1)
        num = 1.0 / (1.0 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        np.fill_diagonal(num, 0.0)
        q = num / np.sum(num)
        q = np.maximum(q, 1e-12)
        pq = p - q

        dy = np.zeros_like(y)
        for i in range(n):
            dy[i, :] = np.sum((pq[:, i][:, None] * num[:, i][:, None]) * (y[i, :] - y), axis=0)
        dy *= 4.0

        gains = (gains + 0.2) * ((np.sign(dy) != np.sign(y_inc)).astype(np.float64)) + (gains * 0.8) * (
            (np.sign(dy) == np.sign(y_inc)).astype(np.float64)
        )
        gains = np.maximum(gains, 0.01)
        y_inc = momentum * y_inc - eta * (gains * dy)
        y += y_inc
        y -= np.mean(y, axis=0, keepdims=True)

        if it == 100:
            p = np.maximum(p / 4.0, 1e-12)
            momentum = final_momentum
    return y


def _inline_html_page():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analysis Results Dashboard</title>
  <script defer src="https://cdn.plot.ly/plotly-2.35.2.min.js" onload="window.__analysisPlotlyReady && window.__analysisPlotlyReady()" onerror="window.__analysisPlotlyFailed && window.__analysisPlotlyFailed()"></script>
  <style>
    :root { --bg:#f5f6f8; --card:#ffffff; --text:#1f2937; --muted:#6b7280; --line:#e5e7eb; }
    body { margin: 0; font-family: "Helvetica Neue", Arial, sans-serif; background: var(--bg); color: var(--text); }
    .wrap { width: 100%; max-width: none; box-sizing: border-box; margin: 12px 0; padding: 0 10px; }
    .layout { display: grid; grid-template-columns: minmax(300px, 17vw) minmax(0, 1fr) minmax(300px, 17vw); gap: 12px; align-items: start; }
    .left-col { position: sticky; top: 12px; align-self: start; max-height: calc(100vh - 24px); overflow: auto; padding-right: 4px; }
    .right-col { min-width: 0; }
    .export-col { position: sticky; top: 12px; align-self: start; max-height: calc(100vh - 24px); overflow: auto; padding-right: 4px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; margin-bottom: 12px; }
    .controls { display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; align-items: end; }
    label { font-size: 13px; color: var(--muted); display:block; margin-bottom:6px; }
    select, input[type="file"] { width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    #plot { width: 100%; height: 580px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; border-top: 1px solid var(--line); padding: 8px; }
    tr.selected-row { background: #eef6ff; }
    .msg { color: var(--muted); font-size: 14px; margin-top: 8px; }
    .metric { margin-top: 10px; font-size: 14px; color: #111827; }
    .small { font-size: 12px; color: var(--muted); }
    .plot-meta { display:grid; grid-template-columns: minmax(220px, 1.1fr) minmax(120px, 0.4fr) minmax(260px, 1fr); gap:8px; align-items:stretch; margin:8px 0 10px 0; }
    .plot-meta .metric { margin:0; border:1px solid var(--line); border-radius:8px; padding:9px 10px; background:#f8fafc; }
    .plot-meta #status { margin:0; border:1px solid var(--line); border-radius:8px; padding:9px 10px; background:#f8fafc; color:#111827; }
    .feature-grid { display:grid; grid-template-columns: 1fr; gap:8px; max-height:220px; overflow:auto; border:1px solid var(--line); padding:10px; border-radius:6px; }
    .predictor-columns { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:10px; border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
    .predictor-column { display:grid; gap:8px; align-content:start; max-height:320px; overflow:auto; padding-right:4px; }
    .feature-category { border:1px solid var(--line); border-radius:8px; padding:8px; background:#fbfdff; }
    .feature-category-header { display:flex; justify-content:space-between; gap:8px; align-items:center; margin-bottom:6px; }
    .feature-category-title { font-weight:700; font-size:12px; color:#111827; }
    .mini-btn { padding:4px 7px; font-size:11px; border:1px solid var(--line); border-radius:5px; background:#fff; cursor:pointer; }
    .feature-hit { background:#fff7ed; outline:2px solid #f97316; border-radius:6px; }
    .feature-link { color:#0f766e; text-decoration: underline; cursor: pointer; }
    .export-feature-list { display:grid; gap:8px; max-height:58vh; overflow:auto; border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
    .export-actions { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0; }
    .quality-cards { display:grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap:10px; margin:10px 0; }
    .quality-card { border:1px solid var(--line); border-radius:10px; padding:12px; background:linear-gradient(180deg,#f8fafc,#ffffff); }
    .quality-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
    .quality-value { font-size:24px; font-weight:800; color:#0f172a; margin-top:4px; }
    .quality-note { font-size:11px; color:var(--muted); margin-top:4px; }
    .classifier-diagnostics { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin-top:8px; }
    .classifier-diagnostic-plot { width:100%; height:360px; min-width:0; }
    .row-filter-panel { border:1px solid var(--line); border-radius:8px; padding:10px; margin:8px 0 10px 0; background:#fbfdff; }
    .row-filter-grid { display:grid; grid-template-columns: 1fr; gap:8px; }
    .row-filter-actions { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:8px; }
    .row-filter-rules { display:grid; gap:6px; margin-top:8px; }
    .row-filter-rule { display:flex; justify-content:space-between; gap:8px; align-items:center; border:1px solid var(--line); border-radius:7px; padding:7px 8px; background:#fff; }
    .row-filter-rule span { min-width:0; overflow-wrap:anywhere; }
    @media (max-width: 1300px) {
      .layout { grid-template-columns: 1fr; }
      .left-col, .export-col { position: static; max-height: none; overflow: visible; }
      .plot-meta { grid-template-columns: 1fr; }
      .quality-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .predictor-columns { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .classifier-diagnostics { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .predictor-columns { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2 style="margin:0 0 4px 0;">Analysis Results Dashboard</h2>
      <div class="msg">Load the current alignment annotations or upload a JSON file, select feature columns, and inspect scatterplot correlation.</div>
      <div class="controls" style="margin-top:10px;">
        <div>
          <label for="jsonFile">Upload JSON</label>
          <input id="jsonFile" type="file" accept=".json,application/json" />
        </div>
        <div>
          <label>Current server state</label>
          <button id="loadCurrentState" type="button">Load Current Annotations</button>
        </div>
      </div>
    </div>

    <div id="summary" class="small" style="margin:-4px 0 10px 2px;"></div>

    <div class="layout">
      <div class="left-col">
        <div class="card">
          <h3 style="margin:0 0 8px 0;">Plotted Rows</h3>
          <div id="rowStats" class="small" style="margin-bottom:8px;"></div>
          <div class="row-filter-panel">
            <label for="rowSelPresenceFilter">SEL presence</label>
            <select id="rowSelPresenceFilter" style="margin-bottom:8px;">
              <option value="all">All rows</option>
              <option value="present">SEL present</option>
              <option value="absent">SEL absent</option>
            </select>
            <label for="rowFilterFeature">Create row filter</label>
            <div class="row-filter-grid">
              <select id="rowFilterFeature"></select>
              <select id="rowFilterOperator"></select>
              <input id="rowFilterValue" type="number" step="any" placeholder="Threshold value" />
              <select id="rowFilterCategoryValue"></select>
            </div>
            <div class="row-filter-actions">
              <button id="addRowFilterRule" type="button">Add Filter Rule</button>
              <button id="clearRowFilterRules" type="button">Clear Rules</button>
            </div>
            <div id="rowFilterStatus" class="small" style="margin-top:6px;">No row filters active.</div>
            <div id="rowFilterRules" class="row-filter-rules"></div>
          </div>
          <div style="overflow:auto; max-height: 50vh;">
            <table>
              <thead><tr id="rowsHead"></tr></thead>
              <tbody id="rowsBody"></tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="right-col">
        <div class="card">
          <div class="controls" style="margin-bottom:8px;">
            <div>
              <label for="xFeatureGroup">X-axis feature category</label>
              <select id="xFeatureGroup"></select>
              <label for="xFeature" style="margin-top:8px;">Specific X-axis feature</label>
              <select id="xFeature"></select>
            </div>
            <div>
              <label for="yFeatureGroup">Y-axis feature category</label>
              <select id="yFeatureGroup"></select>
              <label for="yFeature" style="margin-top:8px;">Specific Y-axis feature</label>
              <select id="yFeature"></select>
            </div>
          </div>
          <div class="plot-meta">
            <div id="status" class="msg">Upload a JSON file to begin.</div>
            <div id="r2Value" class="metric">R²: -</div>
            <div id="regressionExpr" class="metric">Regression: -</div>
          </div>
          <button id="swapScatterAxes" type="button">Swap X/Y Axes</button>
          <button id="exportScatterCsv" type="button" style="margin-left:6px;">Export Scatter Data CSV</button>
          <div id="plot"></div>
        </div>
        <div class="card">
          <h3 style="margin-top:0;">Feature-to-Feature Correlation Filter</h3>
          <div class="small" style="margin-bottom:8px;">
            Screen all numeric feature pairs using the currently visible plotted rows. Use the row filters on the left first if you want the correlation screen to focus on a subset.
          </div>
          <div class="controls" style="margin-bottom:8px;">
            <div>
              <label for="corrThreshold">Correlation threshold</label>
              <input id="corrThreshold" type="number" min="0" max="1" step="0.05" value="0.5" />
              <label for="corrR2Threshold" style="margin-top:8px;">R² threshold</label>
              <input id="corrR2Threshold" type="number" min="0" max="1" step="0.05" value="0" />
              <label style="margin-top:8px;">
                <input id="corrUseAbs" type="checkbox" checked />
                Use absolute correlation, so strong negative correlations are included
              </label>
            </div>
            <div>
              <label>Correlation screen</label>
              <button id="runCorrelationFilter" type="button">Run Correlation Filter</button>
              <button id="exportCorrelationCsv" type="button" style="margin-left:6px;">Export Correlations CSV</button>
              <div id="corrStatus" class="small" style="margin-top:8px;">Load annotations to screen feature correlations.</div>
            </div>
          </div>
          <div style="margin-bottom:10px;">
            <label>Correlation feature selection</label>
            <div class="small" style="margin-bottom:6px;">Only checked numeric features are screened for feature-to-feature correlations.</div>
            <button id="corrSelectAll" type="button">Select All Correlation Features</button>
            <button id="corrClearAll" type="button">Clear Correlation Features</button>
            <button id="downloadCorrFeatureSelection" type="button">Download Feature Selection</button>
            <button id="reloadCorrFeatureSelection" type="button">Reload Feature Selection</button>
            <input id="corrFeatureSelectionFile" type="file" accept=".json,application/json" style="display:none;" />
            <div id="corrFeatureList" class="predictor-columns" style="margin-top:8px;"></div>
          </div>
          <div style="overflow:auto; max-height: 360px; margin-top:8px;">
            <table>
              <thead><tr id="corrHead"></tr></thead>
              <tbody id="corrBody"></tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <h3 style="margin-top:0;">Multilinear Regression + ANOVA</h3>
          <div class="small" style="margin-bottom:8px;">
            Select the intended response variable. The dashboard uses the remaining numeric features as independent variables when the row count allows; otherwise it keeps the strongest rank-feasible predictors by absolute correlation with the response.
          </div>
          <div class="controls" style="margin-bottom:8px;">
            <div>
              <label for="mlrResponseGroup">Response feature category</label>
              <select id="mlrResponseGroup"></select>
              <label for="mlrResponseFeature" style="margin-top:8px;">Response variable</label>
              <select id="mlrResponseFeature"></select>
            </div>
            <div>
              <label>Regression model</label>
              <button id="runMlr" type="button">Run Multilinear Regression</button>
              <button id="exportMlrCsv" type="button" style="margin-left:6px;">Export MLR/ANOVA CSV</button>
              <div id="mlrStatus" class="small" style="margin-top:8px;"></div>
            </div>
          </div>
          <div style="margin-bottom:10px;">
            <label>Independent feature selection</label>
            <div class="small" style="margin-bottom:6px;">Checked features are included as candidate predictors. Uncheck features to test how excluding them changes fitting quality.</div>
            <button id="mlrSelectAll" type="button">Select All Predictors</button>
            <button id="mlrClearAll" type="button">Clear All Predictors</button>
            <button id="downloadMlrFeatureSelection" type="button">Download Feature Selection</button>
            <button id="reloadMlrFeatureSelection" type="button">Reload Feature Selection</button>
            <input id="mlrFeatureSelectionFile" type="file" accept=".json,application/json" style="display:none;" />
            <div id="mlrFeatureList" class="predictor-columns" style="margin-top:8px;"></div>
          </div>
          <div id="mlrSummary" class="metric">Model: -</div>
          <div id="mlrPlot" style="width:100%;height:360px;margin-top:8px;"></div>
          <div style="overflow:auto; max-height: 360px; margin-top:8px;">
            <table>
              <thead><tr id="mlrHead"></tr></thead>
              <tbody id="mlrBody"></tbody>
            </table>
          </div>
          <h4 style="margin:16px 0 6px 0;">Component Aggregates</h4>
          <div id="componentSummary" class="small">Run multilinear regression to compute predictor components.</div>
          <div id="componentPlot" style="width:100%;height:360px;margin-top:8px;"></div>
          <div style="overflow:auto; max-height: 360px; margin-top:8px;">
            <table>
              <thead><tr id="componentHead"></tr></thead>
              <tbody id="componentBody"></tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <h3 style="margin-top:0;">Subject Position In Cohort (Lab Feature)</h3>
          <div class="controls" style="margin-bottom:8px;">
            <div>
              <label for="cohortFeatureGroup">Lab feature category</label>
              <select id="cohortFeatureGroup"></select>
              <label for="cohortFeature" style="margin-top:8px;">Lab feature (all subjects)</label>
              <select id="cohortFeature"></select>
            </div>
          </div>
          <div id="cohortStats" class="small"></div>
          <div id="cohortPlot" style="width:100%;height:320px;margin-top:8px;"></div>
        </div>
        <div class="card">
          <h3 style="margin-top:0;">Feature Distribution + Inter/Intra-Subject Variation</h3>
          <div class="controls" style="margin-bottom:8px;">
            <div>
              <label for="distFeatureGroup">Feature category</label>
              <select id="distFeatureGroup"></select>
              <label for="distFeature" style="margin-top:8px;">Feature for subject distribution analysis</label>
              <select id="distFeature"></select>
            </div>
            <div>
              <label for="distLevel">Analysis level</label>
              <select id="distLevel">
                <option value="both">Both cohort + individual</option>
                <option value="cohort">Cohort only</option>
                <option value="individual">Individual only</option>
              </select>
            </div>
          </div>
          <div id="distStats" class="small"></div>
          <div id="distPlot" style="width:100%;height:320px;margin-top:8px;"></div>
          <div id="distHistPlot" style="width:100%;height:300px;margin-top:8px;"></div>
        </div>
        <div class="card">
          <h3 style="margin-top:0;">t-SNE Plot</h3>
          <div class="small" style="margin-bottom:8px;">Select numeric features to include, then run t-SNE when needed. It is intentionally manual to keep loading responsive.</div>
          <button id="runTsne" type="button">Run t-SNE</button>
          <button id="exportTsneCsv" type="button" style="margin-left:6px;">Export t-SNE Data CSV</button>
          <div class="controls" style="margin-top:8px;">
            <div>
              <label for="tsneColorFeatureGroup">Color/class feature category</label>
              <select id="tsneColorFeatureGroup"></select>
              <label for="tsneColorFeature" style="margin-top:8px;">Define classes from feature</label>
              <select id="tsneColorFeature"></select>
            </div>
            <div>
              <label>Color/class rules</label>
              <button id="addTsneColorRule" type="button">Add Class Rule</button>
              <div id="tsneColorLabelStatus" class="small" style="margin-top:8px;">Add one-sided or bounded color rules such as > lower and < upper. These rules are only used for t-SNE coloring and cluster separation.</div>
            </div>
          </div>
          <div style="overflow:auto; max-height: 180px; margin-top:8px;">
            <table>
              <thead><tr id="tsneColorHead"></tr></thead>
              <tbody id="tsneColorBody"></tbody>
            </table>
          </div>
          <div style="margin-top:8px;">
            <button id="tsneSelectAll" type="button">Select All t-SNE Features</button>
            <button id="tsneClearAll" type="button">Clear All t-SNE Features</button>
            <button id="downloadTsneFeatureSelection" type="button">Download Feature Selection</button>
            <button id="reloadTsneFeatureSelection" type="button">Reload Feature Selection</button>
            <input id="tsneFeatureSelectionFile" type="file" accept=".json,application/json" style="display:none;" />
          </div>
          <div id="tsneFeatureList" class="predictor-columns" style="margin-top:8px;"></div>
          <div id="tsneStatus" class="small" style="margin-top:8px;"></div>
          <div id="tsneDropSummary" class="small" style="margin-top:8px;"></div>
          <div style="overflow:auto; max-height: 220px; margin-top:8px;">
            <table>
              <thead><tr id="dropRowsHead"></tr></thead>
              <tbody id="dropRowsBody"></tbody>
            </table>
          </div>
          <div id="tsnePlot" style="width:100%;height:480px;margin-top:8px;"></div>
          <h4 style="margin:16px 0 6px 0;">Cluster Separation Feature Analysis</h4>
          <div class="small" style="margin-bottom:8px;">Uses the user-defined t-SNE color rules as cluster labels, then trains a linear SVM-style one-vs-rest separator to identify original features that best distinguish the labeled groups.</div>
          <button id="runTsneSvm" type="button">Find Separating Features</button>
          <div id="tsneSvmStatus" class="small" style="margin-top:8px;"></div>
          <div id="tsneSvmPlot" style="width:100%;height:340px;margin-top:8px;"></div>
          <div style="overflow:auto; max-height: 320px; margin-top:8px;">
            <table>
              <thead><tr id="tsneSvmHead"></tr></thead>
              <tbody id="tsneSvmBody"></tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <h3 style="margin-top:0;">Principal Component Analysis</h3>
          <div class="small" style="margin-bottom:8px;">Project selected numeric features into PC1/PC2 to inspect whether visible rows separate into clusters. PCA runs locally in the browser on complete observed rows.</div>
          <button id="runPca" type="button">Run PCA</button>
          <button id="refreshAllCasesPca" type="button" style="margin-left:6px;">Refresh to All Cases</button>
          <div class="controls" style="margin-top:8px;">
            <div>
              <label for="pcaColorFeatureGroup">Color feature category</label>
              <select id="pcaColorFeatureGroup"></select>
              <label for="pcaColorFeature" style="margin-top:8px;">Define PCA color classes from feature</label>
              <select id="pcaColorFeature"></select>
            </div>
            <div>
              <label>PCA color/class rules</label>
              <button id="addPcaColorRule" type="button">Add Class Rule</button>
              <label style="display:block; margin-top:8px;">Number of clusters (K) <input id="pcaClusterCount" type="number" min="2" max="12" step="1" value="3" style="width:70px;" /></label>
              <button id="runPcaAutoCluster" type="button" style="margin-top:6px;">Auto Cluster PCA</button>
              <button id="clearPcaAutoCluster" type="button" style="margin-top:6px;">Use Manual Rules</button>
              <div id="pcaColorRuleStatus" class="small" style="margin-top:8px;">Add bounded rules to assign colors to PCA points. Unmatched points are shown as gray.</div>
            </div>
            <div>
              <label>PCA feature selection</label>
              <button id="pcaSelectAll" type="button">Select All PCA Features</button>
              <button id="pcaClearAll" type="button">Clear PCA Features</button>
              <button id="downloadPcaFeatureSelection" type="button">Download Feature Selection</button>
              <button id="reloadPcaFeatureSelection" type="button">Reload Feature Selection</button>
              <input id="pcaFeatureSelectionFile" type="file" accept=".json,application/json" style="display:none;" />
            </div>
          </div>
          <div style="overflow:auto; max-height: 180px; margin-top:8px;">
            <table>
              <thead><tr id="pcaColorRuleHead"></tr></thead>
              <tbody id="pcaColorRuleBody"></tbody>
            </table>
          </div>
          <div id="pcaFeatureList" class="predictor-columns" style="margin-top:8px;"></div>
          <div id="pcaStatus" class="small" style="margin-top:8px;">Load annotations, choose PCA features, then run PCA.</div>
          <div id="pcaPlot" style="width:100%;height:480px;margin-top:8px;"></div>
          <h4 style="margin:16px 0 6px 0;">Supervised Class-Separating Features</h4>
          <div class="small" style="margin-bottom:8px;">Uses either PCA manual color rules or the specified K auto-clusters as labels, then ranks original features by between-class versus within-class separation. Apply the top-ranked features and rerun PCA to inspect separation.</div>
          <button id="runPcaSupervised" type="button">Find Class-Separating Features</button>
          <label style="margin-left:8px;">Top N <input id="pcaTopNFeatures" type="number" min="2" max="100" step="1" value="10" style="width:70px;" /></label>
          <label style="margin-left:8px;">Min complete cases <input id="pcaMinCompleteCases" type="number" min="3" step="1" value="10" style="width:80px;" /></label>
          <button id="applyPcaTopFeatures" type="button" style="margin-left:6px;">Apply Top Features to PCA</button>
          <div id="pcaSupervisedStatus" class="small" style="margin-top:8px;">Define PCA color rules, then rank features.</div>
          <div id="pcaSupervisedPlot" style="width:100%;height:340px;margin-top:8px;"></div>
          <div style="overflow:auto; max-height: 320px; margin-top:8px;">
            <table>
              <thead><tr id="pcaSupervisedHead"></tr></thead>
              <tbody id="pcaSupervisedBody"></tbody>
            </table>
          </div>
          <div style="overflow:auto; max-height: 320px; margin-top:8px;">
            <table>
              <thead><tr id="pcaHead"></tr></thead>
              <tbody id="pcaBody"></tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <h4 style="margin:16px 0 6px 0;">Classification Model + SHAP-style Explainers</h4>
          <div class="small" style="margin-bottom:8px;">Runs directly on selected features and the user-defined class rules. No t-SNE embedding or t-SNE run is required. Choose a deterministic local classifier; SHAP-style importance is approximated by feature occlusion against class probability.</div>
          <div class="controls" style="margin-bottom:8px;">
            <div>
              <label for="classifierModelType">Classifier model</label>
              <select id="classifierModelType">
                <option value="random_forest">Random Forest-style</option>
                <option value="knn">k-nearest neighbors</option>
                <option value="gaussian_nb">Gaussian Naive Bayes</option>
                <option value="linear_svm">Linear SVM-style</option>
              </select>
              <label for="classifierFeatureSpace">Classifier input space</label>
              <select id="classifierFeatureSpace">
                <option value="selected_features">Selected original features</option>
                <option value="principal_components">Principal components of selected features</option>
              </select>
              <label for="classifierEvaluationUnit" style="margin-top:8px;">Evaluation unit</label>
              <select id="classifierEvaluationUnit">
                <option value="row">Rows / cases</option>
                <option value="subject">Unique subjects (majority vote)</option>
              </select>
              <label for="classifierClassFeatureGroup" style="margin-top:8px;">Class feature category</label>
              <select id="classifierClassFeatureGroup"></select>
              <label for="classifierClassFeature" style="margin-top:8px;">Define classes from feature</label>
              <select id="classifierClassFeature"></select>
              <label for="classifierSeed" style="margin-top:8px;">Classifier random seed</label>
              <input id="classifierSeed" type="number" value="20260606" step="1" />
            </div>
            <div>
              <label>Classification</label>
              <button id="runClassifier" type="button">Run Classification</button>
              <button id="exportClassifierCsv" type="button" style="margin-left:6px;">Export Classification CSV</button>
              <button id="addClassifierClassRule" type="button" style="margin-left:6px;">Add Class Rule</button>
              <div id="classifierStatus" class="small" style="margin-top:8px;"></div>
            </div>
          </div>
          <div class="small" id="classifierClassRuleStatus" style="margin-bottom:8px;">Define classification-specific classes here. These labels are independent from t-SNE color rules.</div>
          <div style="overflow:auto; max-height: 180px; margin-top:8px; margin-bottom:10px;">
            <table>
              <thead><tr id="classifierClassRuleHead"></tr></thead>
              <tbody id="classifierClassRuleBody"></tbody>
            </table>
          </div>
          <div style="margin-bottom:10px;">
            <label>Classification feature selection</label>
            <div class="small" style="margin-bottom:6px;">Checked features are used as classifier predictors. Classes come from the classification-specific class rules above.</div>
            <button id="classifierSelectAll" type="button">Select All Classifier Features</button>
            <button id="classifierClearAll" type="button">Clear Classifier Features</button>
            <button id="downloadClassifierFeatureSelection" type="button">Download Feature Selection</button>
            <button id="reloadClassifierFeatureSelection" type="button">Reload Feature Selection</button>
            <input id="classifierFeatureSelectionFile" type="file" accept=".json,application/json" style="display:none;" />
            <div id="classifierFeatureList" class="predictor-columns" style="margin-top:8px;"></div>
          </div>
          <div id="classificationQualityCards" class="quality-cards">
            <div class="quality-card"><div class="quality-label">Accuracy</div><div class="quality-value">-</div><div class="quality-note">training / LOO</div></div>
            <div class="quality-card"><div class="quality-label">Macro AUROC</div><div class="quality-value">-</div><div class="quality-note">one-vs-rest</div></div>
            <div class="quality-card"><div class="quality-label">Sensitivity</div><div class="quality-value">-</div><div class="quality-note">macro recall</div></div>
            <div class="quality-card"><div class="quality-label">Specificity</div><div class="quality-value">-</div><div class="quality-note">macro true negative rate</div></div>
            <div class="quality-card"><div class="quality-label">F1 Score</div><div class="quality-value">-</div><div class="quality-note">macro harmonic mean</div></div>
          </div>
          <div id="classifierMetricPlot" style="width:100%;height:280px;margin-top:8px;"></div>
          <div class="classifier-diagnostics">
            <div id="classifierRocPlot" class="classifier-diagnostic-plot"></div>
            <div id="classifierPerfPlot" class="classifier-diagnostic-plot"></div>
          </div>
          <div id="classifierShapPlot" style="width:100%;height:360px;margin-top:8px;"></div>
          <div style="overflow:auto; max-height: 320px; margin-top:8px;">
            <table>
              <thead><tr id="classifierHead"></tr></thead>
              <tbody id="classifierBody"></tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="export-col">
        <div class="card">
          <h3 style="margin-top:0;">Feature Export</h3>
          <div class="small">
            Select feature columns to export for every extracted observation row. Identifier columns are always included; missing values are written as NA.
          </div>
          <div class="export-actions">
            <button id="exportSelectAll" type="button">Select All</button>
            <button id="exportClearAll" type="button">Clear All</button>
            <button id="exportCsv" type="button">Export CSV</button>
          </div>
          <div id="exportStatus" class="small" style="margin-bottom:8px;">Load annotations to choose export features.</div>
          <div id="exportFeatureList" class="export-feature-list"></div>
        </div>
      </div>
    </div>    
  </div>

  <script>
    const fileInput = document.getElementById("jsonFile");
    const loadCurrentStateBtn = document.getElementById("loadCurrentState");
    const xFeatureGroup = document.getElementById("xFeatureGroup");
    const xFeature = document.getElementById("xFeature");
    const yFeatureGroup = document.getElementById("yFeatureGroup");
    const yFeature = document.getElementById("yFeature");
    const swapScatterAxesBtn = document.getElementById("swapScatterAxes");
    const exportScatterCsvBtn = document.getElementById("exportScatterCsv");
    const corrThresholdInput = document.getElementById("corrThreshold");
    const corrR2ThresholdInput = document.getElementById("corrR2Threshold");
    const corrUseAbsInput = document.getElementById("corrUseAbs");
    const runCorrelationFilterBtn = document.getElementById("runCorrelationFilter");
    const exportCorrelationCsvBtn = document.getElementById("exportCorrelationCsv");
    const corrSelectAllBtn = document.getElementById("corrSelectAll");
    const corrClearAllBtn = document.getElementById("corrClearAll");
    const downloadCorrFeatureSelectionBtn = document.getElementById("downloadCorrFeatureSelection");
    const reloadCorrFeatureSelectionBtn = document.getElementById("reloadCorrFeatureSelection");
    const corrFeatureSelectionFile = document.getElementById("corrFeatureSelectionFile");
    const corrFeatureListEl = document.getElementById("corrFeatureList");
    const corrStatusEl = document.getElementById("corrStatus");
    const corrHead = document.getElementById("corrHead");
    const corrBody = document.getElementById("corrBody");
    const distFeatureGroup = document.getElementById("distFeatureGroup");
    const distFeature = document.getElementById("distFeature");
    const distLevel = document.getElementById("distLevel");
    const cohortFeatureGroup = document.getElementById("cohortFeatureGroup");
    const cohortFeature = document.getElementById("cohortFeature");
    const mlrResponseGroup = document.getElementById("mlrResponseGroup");
    const mlrResponseFeature = document.getElementById("mlrResponseFeature");
    const runMlrBtn = document.getElementById("runMlr");
    const exportMlrCsvBtn = document.getElementById("exportMlrCsv");
    const mlrSelectAllBtn = document.getElementById("mlrSelectAll");
    const mlrClearAllBtn = document.getElementById("mlrClearAll");
    const downloadMlrFeatureSelectionBtn = document.getElementById("downloadMlrFeatureSelection");
    const reloadMlrFeatureSelectionBtn = document.getElementById("reloadMlrFeatureSelection");
    const mlrFeatureSelectionFile = document.getElementById("mlrFeatureSelectionFile");
    const mlrFeatureListEl = document.getElementById("mlrFeatureList");
    const mlrStatusEl = document.getElementById("mlrStatus");
    const mlrSummaryEl = document.getElementById("mlrSummary");
    const mlrHead = document.getElementById("mlrHead");
    const mlrBody = document.getElementById("mlrBody");
    const componentSummaryEl = document.getElementById("componentSummary");
    const componentHead = document.getElementById("componentHead");
    const componentBody = document.getElementById("componentBody");
    const statusEl = document.getElementById("status");
    const summaryEl = document.getElementById("summary");
    const r2El = document.getElementById("r2Value");
    const regressionEl = document.getElementById("regressionExpr");
    const cohortStatsEl = document.getElementById("cohortStats");
    const distStatsEl = document.getElementById("distStats");
    const tsneFeatureListEl = document.getElementById("tsneFeatureList");
    const runTsneBtn = document.getElementById("runTsne");
    const exportTsneCsvBtn = document.getElementById("exportTsneCsv");
    const tsneSelectAllBtn = document.getElementById("tsneSelectAll");
    const tsneClearAllBtn = document.getElementById("tsneClearAll");
    const downloadTsneFeatureSelectionBtn = document.getElementById("downloadTsneFeatureSelection");
    const reloadTsneFeatureSelectionBtn = document.getElementById("reloadTsneFeatureSelection");
    const tsneFeatureSelectionFile = document.getElementById("tsneFeatureSelectionFile");
    const tsneColorFeatureGroup = document.getElementById("tsneColorFeatureGroup");
    const tsneColorFeature = document.getElementById("tsneColorFeature");
    const addTsneColorRuleBtn = document.getElementById("addTsneColorRule");
    const tsneColorLabelStatusEl = document.getElementById("tsneColorLabelStatus");
    const tsneColorHead = document.getElementById("tsneColorHead");
    const tsneColorBody = document.getElementById("tsneColorBody");
    const tsneStatusEl = document.getElementById("tsneStatus");
    const tsneDropSummaryEl = document.getElementById("tsneDropSummary");
    const dropRowsHead = document.getElementById("dropRowsHead");
    const dropRowsBody = document.getElementById("dropRowsBody");
    const runTsneSvmBtn = document.getElementById("runTsneSvm");
    const tsneSvmStatusEl = document.getElementById("tsneSvmStatus");
    const tsneSvmHead = document.getElementById("tsneSvmHead");
    const tsneSvmBody = document.getElementById("tsneSvmBody");
    const runPcaBtn = document.getElementById("runPca");
    const refreshAllCasesPcaBtn = document.getElementById("refreshAllCasesPca");
    const pcaColorFeatureGroup = document.getElementById("pcaColorFeatureGroup");
    const pcaColorFeature = document.getElementById("pcaColorFeature");
    const addPcaColorRuleBtn = document.getElementById("addPcaColorRule");
    const pcaClusterCountInput = document.getElementById("pcaClusterCount");
    const runPcaAutoClusterBtn = document.getElementById("runPcaAutoCluster");
    const clearPcaAutoClusterBtn = document.getElementById("clearPcaAutoCluster");
    const pcaColorRuleStatusEl = document.getElementById("pcaColorRuleStatus");
    const pcaColorRuleHead = document.getElementById("pcaColorRuleHead");
    const pcaColorRuleBody = document.getElementById("pcaColorRuleBody");
    const pcaSelectAllBtn = document.getElementById("pcaSelectAll");
    const pcaClearAllBtn = document.getElementById("pcaClearAll");
    const downloadPcaFeatureSelectionBtn = document.getElementById("downloadPcaFeatureSelection");
    const reloadPcaFeatureSelectionBtn = document.getElementById("reloadPcaFeatureSelection");
    const pcaFeatureSelectionFile = document.getElementById("pcaFeatureSelectionFile");
    const pcaFeatureListEl = document.getElementById("pcaFeatureList");
    const pcaStatusEl = document.getElementById("pcaStatus");
    const runPcaSupervisedBtn = document.getElementById("runPcaSupervised");
    const pcaTopNFeaturesInput = document.getElementById("pcaTopNFeatures");
    const pcaMinCompleteCasesInput = document.getElementById("pcaMinCompleteCases");
    const applyPcaTopFeaturesBtn = document.getElementById("applyPcaTopFeatures");
    const pcaSupervisedStatusEl = document.getElementById("pcaSupervisedStatus");
    const pcaSupervisedHead = document.getElementById("pcaSupervisedHead");
    const pcaSupervisedBody = document.getElementById("pcaSupervisedBody");
    const pcaHead = document.getElementById("pcaHead");
    const pcaBody = document.getElementById("pcaBody");
    const classifierModelType = document.getElementById("classifierModelType");
    const classifierFeatureSpace = document.getElementById("classifierFeatureSpace");
    const classifierEvaluationUnit = document.getElementById("classifierEvaluationUnit");
    const classifierClassFeatureGroup = document.getElementById("classifierClassFeatureGroup");
    const classifierClassFeature = document.getElementById("classifierClassFeature");
    const classifierSeedInput = document.getElementById("classifierSeed");
    const runClassifierBtn = document.getElementById("runClassifier");
    const exportClassifierCsvBtn = document.getElementById("exportClassifierCsv");
    const addClassifierClassRuleBtn = document.getElementById("addClassifierClassRule");
    const classifierSelectAllBtn = document.getElementById("classifierSelectAll");
    const classifierClearAllBtn = document.getElementById("classifierClearAll");
    const downloadClassifierFeatureSelectionBtn = document.getElementById("downloadClassifierFeatureSelection");
    const reloadClassifierFeatureSelectionBtn = document.getElementById("reloadClassifierFeatureSelection");
    const classifierFeatureSelectionFile = document.getElementById("classifierFeatureSelectionFile");
    const classifierFeatureListEl = document.getElementById("classifierFeatureList");
    const classifierStatusEl = document.getElementById("classifierStatus");
    const classifierClassRuleStatusEl = document.getElementById("classifierClassRuleStatus");
    const classifierClassRuleHead = document.getElementById("classifierClassRuleHead");
    const classifierClassRuleBody = document.getElementById("classifierClassRuleBody");
    const classificationQualityCards = document.getElementById("classificationQualityCards");
    const classifierHead = document.getElementById("classifierHead");
    const classifierBody = document.getElementById("classifierBody");
    const rowStatsEl = document.getElementById("rowStats");
    const rowFilterFeature = document.getElementById("rowFilterFeature");
    const rowFilterOperator = document.getElementById("rowFilterOperator");
    const rowFilterValue = document.getElementById("rowFilterValue");
    const rowFilterCategoryValue = document.getElementById("rowFilterCategoryValue");
    const rowSelPresenceFilter = document.getElementById("rowSelPresenceFilter");
    const addRowFilterRuleBtn = document.getElementById("addRowFilterRule");
    const clearRowFilterRulesBtn = document.getElementById("clearRowFilterRules");
    const rowFilterStatusEl = document.getElementById("rowFilterStatus");
    const rowFilterRulesEl = document.getElementById("rowFilterRules");
    const rowsHead = document.getElementById("rowsHead");
    const rowsBody = document.getElementById("rowsBody");
    const exportFeatureListEl = document.getElementById("exportFeatureList");
    const exportSelectAllBtn = document.getElementById("exportSelectAll");
    const exportClearAllBtn = document.getElementById("exportClearAll");
    const exportCsvBtn = document.getElementById("exportCsv");
    const exportStatusEl = document.getElementById("exportStatus");

    let allRows = [];
    let numericFeatures = [];
    let exportFeatures = [];
    let rowFilterFeatures = [];
    let plottedRows = [];
    let rowVisibility = [];
    let rowManualVisibility = [];
    let rowFilterVisibility = [];
    let rowFilterRules = [];
    let rowSelPresenceMode = "all";
    let corrSelectedFeatures = new Set();
    let currentXKey = "";
    let currentYKey = "";
    let tsneSelectedFeatures = new Set();
    let tsneColorRules = [];
    let lastTsneRender = null;
    let pcaSelectedFeatures = new Set();
    let pcaColorRules = [];
    let pcaAutoClusterMode = false;
    let pcaAutoClusterInfo = null;
    let lastPcaRender = null;
    let lastPcaSupervisedRows = [];
    let classifierSelectedFeatures = new Set();
    let classifierClassRules = [];
    let subjectLabMap = {};
    let cohortLabFeatures = [];
    let distFeatureOptions = [];
    let mlrIncludedFeatures = new Set();
    let exportSelectedFeatures = new Set();
    const MIN_MLR_ERROR_DF = 5;
    const mlrFeatureElements = new Map();
    let mlrPlotClickHandler = null;
    let deferredContextPlotTimer = null;
    const pendingPlotlyPlots = new Map();
    let lastScatterExportRows = [];
    let lastCorrelationExportRows = [];
    let lastMlrExportRows = [];
    let lastTsneExportRows = [];
    let lastClassifierExportRows = [];

    function setStatus(msg) {
      statusEl.textContent = msg || "";
    }

    window.__analysisPlotlyReady = function() {
      flushPendingPlotlyPlots();
    };

    window.__analysisPlotlyFailed = function() {
      setStatus("Analysis data can still load, but Plotly could not be loaded for chart rendering.");
    };

    function flushPendingPlotlyPlots() {
      if (!window.Plotly || typeof Plotly.newPlot !== "function") return;
      const queued = Array.from(pendingPlotlyPlots.entries());
      pendingPlotlyPlots.clear();
      for (const [target, args] of queued) {
        safeNewPlot(target, args.traces, args.layout, args.config);
      }
    }

    function safeNewPlot(target, traces, layout, config) {
      if (!window.Plotly || typeof Plotly.newPlot !== "function") {
        pendingPlotlyPlots.set(target, { traces, layout, config });
        const el = typeof target === "string" ? document.getElementById(target) : target;
        if (el && !el.dataset.plotlyQueued) {
          el.dataset.plotlyQueued = "1";
          el.innerHTML = '<div class="small" style="padding:16px;">Chart rendering is waiting for Plotly. Tables and export controls remain available.</div>';
        }
        return Promise.resolve();
      }
      try {
        const el = typeof target === "string" ? document.getElementById(target) : target;
        if (el) {
          delete el.dataset.plotlyQueued;
        }
        return Plotly.newPlot(target, traces, layout, config);
      } catch (err) {
        setStatus("Plot rendering failed: " + (err.message || err));
        return Promise.resolve();
      }
    }

    function parseMaybeNumber(v) {
      if (v === null || v === undefined) return null;
      if (typeof v === "number") return Number.isFinite(v) ? v : null;
      if (typeof v === "string") {
        const s = v.trim();
        if (!s) return null;
        const n = Number(s);
        return Number.isFinite(n) ? n : null;
      }
      return null;
    }

    function extractRowsFromPayload(payload, caseLabFeaturesMap) {
      const out = [];
      const peakPairFeatures =
        payload?.peak_pair_features ||
        payload?.paired_features ||
        payload?.peakPairFeatures ||
        null;
      let cases = (peakPairFeatures && typeof peakPairFeatures === "object") ? peakPairFeatures.cases : null;
      if (!cases && peakPairFeatures && typeof peakPairFeatures === "object") {
        const looksLikeCaseMap = Object.values(peakPairFeatures).some((v) => v && typeof v === "object" && Array.isArray(v.paired_features));
        if (looksLikeCaseMap) cases = peakPairFeatures;
      }

      if (!cases || typeof cases !== "object") return out;

      for (const [casePath, casePayload] of Object.entries(cases)) {
        const pairedFeatures = Array.isArray(casePayload?.paired_features) ? casePayload.paired_features : [];
        for (const pair of pairedFeatures) {
          const row = {
            case_path: casePath,
            instance_id: pair?.instance_id || `${casePath}::${pair?.name || pairedFeatures.indexOf(pair)}`,
            pair_name: pair?.name || "",
            category: pair?.category || "",
            panel: pair?.panel || "",
          };
          const subjectId = extractSubjectIdFromCasePath(casePath);
          if (subjectId) {
            row.subject_id = subjectId;
          }
          const caseLab = caseLabFeaturesMap[casePath];
          if (caseLab && typeof caseLab === "object") {
            flattenIntoRow(caseLab, "lab_features", row);
          }
          const exportFields = {};
          if (caseLab && typeof caseLab === "object") {
            flattenIntoExportFields(caseLab, "lab_features", exportFields);
          }
          flattenIntoExportFields(pair, "pair", exportFields);
          addPairTimestampExportFields(pair, exportFields);
          flattenIntoRow(pair, "pair", row);
          row.__export_fields = exportFields;
          for (const [key, value] of Object.entries(row)) {
            if (["case_path", "instance_id", "pair_name", "category", "panel", "subject_id", "__export_fields"].includes(key)) continue;
            if (value !== undefined) {
              row.__export_fields[key] = value;
            }
          }
          out.push(row);
        }
      }
      return out;
    }

    function extractSubjectIdFromCasePath(casePath) {
      if (!casePath) return "";
      const m = String(casePath).match(/\\b([PH])(\\d{1,3})\\b/i);
      if (!m) return "";
      return `${m[1].toUpperCase()}${String(Number(m[2])).padStart(2, "0")}`;
    }

    function normalizeSubjectId(value) {
      if (value === null || value === undefined) return "";
      const s = String(value).trim();
      if (!s) return "";
      const m = s.match(/\\b([PH])?(\\d{1,3})\\b/i);
      if (!m) return "";
      const prefix = (m[1] || "P").toUpperCase();
      return `${prefix}${String(Number(m[2])).padStart(2, "0")}`;
    }

    function buildCaseLabFeaturesMap(payload) {
      const out = {};
      const savedSegments = payload?.saved_segments;
      if (!savedSegments || typeof savedSegments !== "object") return out;
      for (const [casePath, segObj] of Object.entries(savedSegments)) {
        if (!segObj || typeof segObj !== "object") continue;
        if (casePath === "lab_features") continue;
        const lf = segObj.lab_features;
        if (lf && typeof lf === "object" && Object.keys(lf).length > 0) {
          out[casePath] = lf;
        }
      }
      return out;
    }

    function buildSubjectLabMap(payload) {
      const agg = {};

      function addSubjectFeature(subjectId, key, value) {
        const sid = normalizeSubjectId(subjectId);
        if (!sid) return;
        const num = parseMaybeNumber(value);
        if (num === null) return;
        if (!agg[sid]) agg[sid] = {};
        if (!agg[sid][key]) agg[sid][key] = [];
        agg[sid][key].push(num);
      }

      // From saved_segments[case].lab_features (most common in this project).
      const savedSegments = payload?.saved_segments;
      if (savedSegments && typeof savedSegments === "object") {
        for (const [casePath, segObj] of Object.entries(savedSegments)) {
          if (!segObj || typeof segObj !== "object") continue;
          if (casePath === "lab_features") continue;
          const sid = extractSubjectIdFromCasePath(casePath);
          if (!sid) continue;
          const lf = segObj.lab_features;
          if (!lf || typeof lf !== "object") continue;
          for (const [k, v] of Object.entries(lf)) {
            addSubjectFeature(sid, `lab_features.${k}`, v);
          }
        }
      }

      // Also support top-level lab_features keyed by subject.
      const top = payload?.lab_features;
      if (top && typeof top === "object") {
        for (const [sidKey, obj] of Object.entries(top)) {
          if (!obj || typeof obj !== "object") continue;
          for (const [k, v] of Object.entries(obj)) {
            addSubjectFeature(sidKey, `lab_features.${k}`, v);
          }
        }
      }

      const out = {};
      for (const [sid, featMap] of Object.entries(agg)) {
        out[sid] = {};
        for (const [feat, vals] of Object.entries(featMap)) {
          if (!vals.length) continue;
          const first = vals[0];
          const hasConflict = vals.some((value) => Math.abs(value - first) > 1e-12);
          if (!hasConflict) out[sid][feat] = first;
        }
        if (Object.keys(out[sid]).length === 0) delete out[sid];
      }
      return out;
    }


    function flattenIntoRow(value, prefix, row) {
      if (value === null || value === undefined) return;
      if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i++) {
          flattenIntoRow(value[i], `${prefix}[${i}]`, row);
        }
        return;
      }
      if (typeof value === "object") {
        for (const [k, v] of Object.entries(value)) {
          flattenIntoRow(v, `${prefix}.${k}`, row);
        }
        return;
      }

      const asNum = parseMaybeNumber(value);
      if (asNum !== null) {
        row[prefix] = asNum;
      } else {
        row[prefix] = String(value);
      }
    }

    function flattenIntoExportFields(value, prefix, row) {
      if (value === undefined) return;
      if (value === null) {
        row[prefix] = null;
        return;
      }
      if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i++) {
          flattenIntoExportFields(value[i], `${prefix}[${i}]`, row);
        }
        return;
      }
      if (typeof value === "object") {
        for (const [k, v] of Object.entries(value)) {
          flattenIntoExportFields(v, `${prefix}.${k}`, row);
        }
        return;
      }

      const asNum = parseMaybeNumber(value);
      row[prefix] = asNum !== null ? asNum : String(value);
    }

    function addPairTimestampExportFields(pair, exportFields) {
      const selected = pair?.selected_peak_metadata;
      if (!selected || typeof selected !== "object") return;
      const timestampFields = ["trough1_timestamp", "peak_timestamp", "trough2_timestamp"];
      for (const signal of ["cgm", "seg", "sel"]) {
        const meta = selected[signal];
        if (!meta || typeof meta !== "object") continue;
        for (const field of timestampFields) {
          const value = meta[field];
          if (value !== undefined && value !== null && String(value).trim() !== "") {
            exportFields[`pair.timestamps.${signal}.${field}`] = String(value);
          }
        }
      }
    }

    function getNumericFeatures(rows, minNonMissing = 1) {
      const counts = new Map();
      for (const row of rows) {
        for (const [k, v] of Object.entries(row)) {
          if (["case_path", "instance_id", "pair_name", "category", "panel"].includes(k)) continue;
          if (typeof v === "number" && Number.isFinite(v)) {
            counts.set(k, (counts.get(k) || 0) + 1);
          }
        }
      }
      return Array.from(counts.entries())
        .filter(([, count]) => count >= minNonMissing)
        .map(([k]) => k)
        .sort();
    }

    function getExportableFeatures(rows) {
      const features = new Set();
      for (const row of rows) {
        for (const [k, v] of Object.entries(row)) {
          if (["case_path", "instance_id", "pair_name", "category", "panel", "subject_id", "__export_fields"].includes(k)) continue;
          if (v !== undefined && v !== null && String(v).trim() !== "") {
            features.add(k);
          }
        }
        const exportFields = row.__export_fields;
        if (exportFields && typeof exportFields === "object") {
          for (const k of Object.keys(exportFields)) {
            features.add(k);
          }
        }
      }
      return Array.from(features).sort();
    }

    function getFilterableFeatures(rows) {
      const features = new Set(["case_path", "instance_id", "pair_name", "category", "panel", "subject_id"]);
      for (const row of rows) {
        for (const [k, v] of Object.entries(row)) {
          if (v === null || v === undefined) continue;
          if (typeof v === "number" && Number.isFinite(v)) features.add(k);
          else if (typeof v === "string" && v.trim()) features.add(k);
        }
      }
      return Array.from(features).filter((feature) => rows.some((row) => row[feature] !== undefined && row[feature] !== null && String(row[feature]).trim() !== "")).sort();
    }

    const FEATURE_CATEGORY_ORDER = [
      "Subject metadata",
      "CGM",
      "SEG",
      "SEL",
      "Pair timestamps",
      "Pair summary",
      "Other",
    ];

    function featureCategory(feature) {
      const f = String(feature || "");
      if (f.startsWith("lab_features.") || f.startsWith("subject.")) return "Subject metadata";
      if (f.includes("pair.selected_peak_metadata.cgm.")) return "CGM";
      if (f.includes("pair.selected_peak_metadata.seg.")) return "SEG";
      if (f.includes("pair.selected_peak_metadata.sel.")) return "SEL";
      if (f.startsWith("pair.timestamps.")) return "Pair timestamps";
      if (f.startsWith("pair.cgm_") || f.includes("_cgm_")) return "CGM";
      if (f.startsWith("pair.seg_") || f.includes("_seg_")) return "SEG";
      if (f.startsWith("pair.sel_") || f.includes("_sel_")) return "SEL";
      if (f.startsWith("pair.")) return "Pair summary";
      return "Other";
    }

    function groupedFeatures(features, allowedCategories = null) {
      const groups = {};
      for (const feature of features) {
        const category = featureCategory(feature);
        if (allowedCategories && !allowedCategories.includes(category)) continue;
        if (!groups[category]) groups[category] = [];
        groups[category].push(feature);
      }
      for (const values of Object.values(groups)) values.sort();
      return groups;
    }

    function orderedVisibleFeatureCategories(groups) {
      return FEATURE_CATEGORY_ORDER.filter((cat) => Array.isArray(groups[cat]) && groups[cat].length > 0);
    }

    function createPredictorColumns(container, categoryCount) {
      const columnCount = Math.max(1, Math.min(6, categoryCount || 1));
      container.style.gridTemplateColumns = `repeat(${columnCount}, minmax(0, 1fr))`;
      return Array.from({ length: columnCount }, () => {
        const col = document.createElement("div");
        col.className = "predictor-column";
        container.appendChild(col);
        return col;
      });
    }

    function featureShortLabel(feature, category) {
      let out = String(feature || "");
      const replacements = {
        "Subject metadata": ["lab_features.", "subject."],
        "CGM": ["pair.selected_peak_metadata.cgm.", "pair."],
        "SEG": ["pair.selected_peak_metadata.seg.", "pair."],
        "SEL": ["pair.selected_peak_metadata.sel.", "pair."],
        "Pair timestamps": ["pair.timestamps."],
        "Pair summary": ["pair."],
      };
      for (const prefix of replacements[category] || []) {
        if (out.startsWith(prefix)) out = out.slice(prefix.length);
      }
      return out;
    }

    function populateGroupedFeaturePicker(groupSelect, featureSelect, features, preferredFeature, allowedCategories = null) {
      const groups = groupedFeatures(features, allowedCategories);
      const categories = FEATURE_CATEGORY_ORDER.filter((cat) => Array.isArray(groups[cat]) && groups[cat].length > 0);
      groupSelect.innerHTML = "";
      for (const category of categories) {
        const opt = document.createElement("option");
        opt.value = category;
        opt.textContent = `${category} (${groups[category].length})`;
        groupSelect.appendChild(opt);
      }
      const preferredCategory = preferredFeature && categories.includes(featureCategory(preferredFeature))
        ? featureCategory(preferredFeature)
        : categories[0];
      groupSelect.value = preferredCategory || "";

      function fillSpecificFeature(keepFeature = "") {
        const category = groupSelect.value;
        const values = groups[category] || [];
        featureSelect.innerHTML = "";
        for (const feature of values) {
          const opt = document.createElement("option");
          opt.value = feature;
          opt.textContent = featureShortLabel(feature, category);
          opt.title = feature;
          featureSelect.appendChild(opt);
        }
        if (keepFeature && values.includes(keepFeature)) {
          featureSelect.value = keepFeature;
        } else if (preferredFeature && values.includes(preferredFeature)) {
          featureSelect.value = preferredFeature;
        } else if (values.length) {
          featureSelect.value = values[0];
        }
      }

      fillSpecificFeature(preferredFeature);
      groupSelect.onchange = () => {
        fillSpecificFeature("");
        featureSelect.dispatchEvent(new Event("change"));
      };
    }

    function populateFeatureSelectors(features) {
      const defaultX =
        features.find(f => f === "lab_features.hba1c") ||
        features.find(f => f.endsWith("pair.selected_peak_metadata.sel.duration_trough1_to_trough2_seconds")) ||
        features.find(f => f.endsWith("pair.selected_peak_metadata.sel.point_values.peak.raw_delta_value")) ||
        features.find(f => f.endsWith("pair.selected_peak_metadata.seg.point_values.peak.converted_concentration")) ||
        features.find(f => f.endsWith("pair.selected_peak_metadata.seg.duration_trough1_to_trough2_seconds")) ||
        features[0];
      const defaultY =
        features.find(f => f.endsWith("pair.selected_peak_metadata.sel.duration_trough1_to_trough2_seconds")) ||
        features.find(f => f.endsWith("pair.selected_peak_metadata.sel.point_values.peak.raw_delta_value")) ||
        features.find(f => f.endsWith("pair.selected_peak_metadata.cgm.point_values.peak.cgm_value")) ||
        features.find(f => f.endsWith("pair.selected_peak_metadata.seg.point_values.peak.converted_concentration")) ||
        features.find(f => f.endsWith("pair.selected_peak_metadata.cgm.duration_trough1_to_trough2_seconds")) ||
        features[1] ||
        features[0];
      const yDefault = defaultY === defaultX && features.length > 1 ? features[1] : defaultY;
      populateGroupedFeaturePicker(xFeatureGroup, xFeature, features, defaultX);
      populateGroupedFeaturePicker(yFeatureGroup, yFeature, features, yDefault);
      const previousCorrSelection = corrSelectedFeatures instanceof Set ? corrSelectedFeatures : new Set();
      corrSelectedFeatures = previousCorrSelection.size
        ? new Set(features.filter((feature) => previousCorrSelection.has(feature)))
        : new Set(features);
      renderCorrelationFeatureSelector(features);
      populateGroupedFeaturePicker(mlrResponseGroup, mlrResponseFeature, features, yDefault || defaultY || defaultX);
      mlrIncludedFeatures = new Set(features.filter((feature) => feature !== mlrResponseFeature.value));
      renderMlrFeatureSelector();
      populateGroupedFeaturePicker(tsneColorFeatureGroup, tsneColorFeature, features, features[0]);
      renderTsneColorRuleTable();
      tsneSelectedFeatures = new Set(features);
      renderTsneFeatureSelector(features);
      populateGroupedFeaturePicker(pcaColorFeatureGroup, pcaColorFeature, features, features[0]);
      pcaSelectedFeatures = new Set(features);
      pcaColorRules = [];
      pcaAutoClusterMode = false;
      pcaAutoClusterInfo = null;
      renderPcaColorRuleTable();
      lastPcaRender = null;
      renderPcaFeatureSelector(features);
      pcaStatusEl.textContent = "PCA not run automatically. Click Run PCA to compute.";
      populateGroupedFeaturePicker(classifierClassFeatureGroup, classifierClassFeature, features, features[0]);
      renderClassifierClassRuleTable();
      const previousClassifierSelection = classifierSelectedFeatures instanceof Set ? classifierSelectedFeatures : new Set();
      classifierSelectedFeatures = previousClassifierSelection.size
        ? new Set(features.filter((feature) => previousClassifierSelection.has(feature)))
        : new Set(features);
      renderClassifierFeatureSelector(features);
      const previousExportSelection = exportSelectedFeatures instanceof Set ? exportSelectedFeatures : new Set();
      exportSelectedFeatures = previousExportSelection.size
        ? new Set(exportFeatures.filter((feature) => previousExportSelection.has(feature)))
        : new Set(exportFeatures);
      renderExportFeatureSelector(exportFeatures);
    }

    function renderCorrelationFeatureSelector(features = numericFeatures) {
      if (!corrFeatureListEl) return;
      const previous = corrSelectedFeatures instanceof Set ? corrSelectedFeatures : new Set(features);
      corrFeatureListEl.innerHTML = "";
      const groups = groupedFeatures(features);
      const categories = orderedVisibleFeatureCategories(groups);
      const columns = createPredictorColumns(corrFeatureListEl, categories.length);
      let renderedCategoryIndex = 0;
      for (const category of categories) {
        const groupFeatures = groups[category] || [];
        if (!groupFeatures.length) continue;
        const categoryBox = document.createElement("div");
        categoryBox.className = "feature-category";

        const header = document.createElement("div");
        header.className = "feature-category-header";
        const selectedCount = groupFeatures.filter((feature) => previous.has(feature)).length;
        const title = document.createElement("div");
        title.className = "feature-category-title";
        title.textContent = `${category} (${selectedCount}/${groupFeatures.length} selected)`;

        const actions = document.createElement("div");
        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "mini-btn";
        selectBtn.textContent = "Select category";
        selectBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => corrSelectedFeatures.add(feature));
          renderCorrelationFeatureSelector(features);
          corrStatusEl.textContent = `Selected ${groupFeatures.length} correlation features from ${category}. Click Run Correlation Filter to rescreen.`;
        });
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "mini-btn";
        clearBtn.textContent = "Clear category";
        clearBtn.style.marginLeft = "6px";
        clearBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => corrSelectedFeatures.delete(feature));
          renderCorrelationFeatureSelector(features);
          corrStatusEl.textContent = `Cleared ${groupFeatures.length} correlation features from ${category}. Click Run Correlation Filter to rescreen.`;
        });
        actions.appendChild(selectBtn);
        actions.appendChild(clearBtn);
        header.appendChild(title);
        header.appendChild(actions);
        categoryBox.appendChild(header);

        for (const feature of groupFeatures) {
          const id = `corr_feat_${feature.replace(/[^a-zA-Z0-9_]/g, "_")}`;
          const wrapper = document.createElement("label");
          wrapper.style.display = "block";
          wrapper.style.fontSize = "12px";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.id = id;
          input.dataset.corrFeature = feature;
          input.checked = previous.has(feature);
          wrapper.appendChild(input);
          wrapper.appendChild(document.createTextNode(` ${featureShortLabel(feature, category)}`));
          categoryBox.appendChild(wrapper);
        }
        columns[renderedCategoryIndex % columns.length].appendChild(categoryBox);
        renderedCategoryIndex += 1;
      }
      corrSelectedFeatures = new Set(features.filter((feature) => previous.has(feature)));
      corrFeatureListEl.querySelectorAll("input[data-corr-feature]").forEach((el) => {
        el.addEventListener("change", () => {
          const feature = el.getAttribute("data-corr-feature");
          if (!feature) return;
          if (el.checked) corrSelectedFeatures.add(feature);
          else corrSelectedFeatures.delete(feature);
          corrStatusEl.textContent = `Correlation feature selection changed. ${corrSelectedFeatures.size} features selected. Click Run Correlation Filter to rescreen.`;
        });
      });
    }

    function downloadCorrelationFeatureSelection() {
      const features = Array.from(corrSelectedFeatures).filter((feature) => numericFeatures.includes(feature)).sort();
      const payload = {
        type: "correlation_feature_selection",
        saved_at: new Date().toISOString(),
        selected_feature_count: features.length,
        selected_features: features,
      };
      if (downloadJson(payload, "analysis_correlation_feature_selection")) {
        corrStatusEl.textContent = `Downloaded ${features.length} selected correlation features.`;
      }
    }

    function applyCorrelationFeatureSelectionPayload(payload, sourceLabel = "JSON") {
      const rawFeatures = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.selected_features)
          ? payload.selected_features
          : Array.isArray(payload?.features)
            ? payload.features
            : [];
      const selected = rawFeatures.map((feature) => String(feature || "")).filter((feature) => numericFeatures.includes(feature));
      corrSelectedFeatures = new Set(selected);
      renderCorrelationFeatureSelector(numericFeatures);
      const dropped = rawFeatures.length - selected.length;
      corrStatusEl.textContent = `Reloaded ${selected.length} correlation features from ${sourceLabel}${dropped > 0 ? `; ignored ${dropped} unavailable/non-numeric features.` : "."}`;
    }

    function featureSelectionPayload(type, selectedSet, availableFeatures, extra = {}) {
      const features = Array.from(selectedSet || [])
        .filter((feature) => availableFeatures.includes(feature))
        .sort();
      return {
        type,
        saved_at: new Date().toISOString(),
        selected_feature_count: features.length,
        selected_features: features,
        ...extra,
      };
    }

    function selectedFeaturesFromPayload(payload) {
      return Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.selected_features)
          ? payload.selected_features
          : Array.isArray(payload?.features)
            ? payload.features
            : [];
    }

    function applyFeatureSelectionPayload(payload, availableFeatures, assignSelection, renderSelection, statusEl, sourceLabel = "JSON", label = "feature") {
      const rawFeatures = selectedFeaturesFromPayload(payload);
      const selected = rawFeatures.map((feature) => String(feature || "")).filter((feature) => availableFeatures.includes(feature));
      assignSelection(new Set(selected));
      renderSelection();
      const dropped = rawFeatures.length - selected.length;
      statusEl.textContent = `Reloaded ${selected.length} ${label} features from ${sourceLabel}${dropped > 0 ? `; ignored ${dropped} unavailable/non-numeric features.` : "."}`;
    }

    function bindFeatureSelectionFileInput(fileInput, applyPayload, statusEl, label) {
      fileInput.addEventListener("change", async (event) => {
        const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
        if (!file) return;
        try {
          const text = await file.text();
          applyPayload(JSON.parse(text), file.name || "JSON");
        } catch (err) {
          statusEl.textContent = `Failed to reload ${label} feature selection: ` + (err.message || err);
        } finally {
          fileInput.value = "";
        }
      });
    }

    function downloadMlrFeatureSelection() {
      const available = numericFeatures.filter((feature) => feature !== mlrResponseFeature.value);
      const payload = featureSelectionPayload("mlr_feature_selection", mlrIncludedFeatures, available, {
        response_feature: mlrResponseFeature.value || "",
      });
      if (downloadJson(payload, "analysis_mlr_feature_selection")) {
        mlrStatusEl.textContent = `Downloaded ${payload.selected_feature_count} selected MLR predictors.`;
      }
    }

    function downloadTsneFeatureSelection() {
      const payload = featureSelectionPayload("tsne_feature_selection", tsneSelectedFeatures, numericFeatures);
      if (downloadJson(payload, "analysis_tsne_feature_selection")) {
        tsneStatusEl.textContent = `Downloaded ${payload.selected_feature_count} selected t-SNE features.`;
      }
    }

    function downloadPcaFeatureSelection() {
      const payload = featureSelectionPayload("pca_feature_selection", pcaSelectedFeatures, numericFeatures, {
        color_feature: pcaColorFeature.value || "",
      });
      if (downloadJson(payload, "analysis_pca_feature_selection")) {
        pcaStatusEl.textContent = `Downloaded ${payload.selected_feature_count} selected PCA features.`;
      }
    }

    function downloadClassifierFeatureSelection() {
      const payload = featureSelectionPayload("classifier_feature_selection", classifierSelectedFeatures, numericFeatures, {
        class_feature: classifierClassFeature.value || "",
      });
      if (downloadJson(payload, "analysis_classifier_feature_selection")) {
        classifierStatusEl.textContent = `Downloaded ${payload.selected_feature_count} selected classifier features.`;
      }
    }

    function renderMlrFeatureSelector() {
      if (!mlrFeatureListEl) return;
      const response = mlrResponseFeature.value;
      const features = numericFeatures.filter((feature) => feature !== response);
      const previous = mlrIncludedFeatures instanceof Set ? mlrIncludedFeatures : new Set(features);
      mlrFeatureListEl.innerHTML = "";
      mlrFeatureElements.clear();
      const groups = groupedFeatures(features);
      const categories = orderedVisibleFeatureCategories(groups);
      const columns = createPredictorColumns(mlrFeatureListEl, categories.length);
      let renderedCategoryIndex = 0;
      for (const category of categories) {
        const groupFeatures = groups[category] || [];
        if (!groupFeatures.length) continue;
        const categoryBox = document.createElement("div");
        categoryBox.className = "feature-category";

        const header = document.createElement("div");
        header.className = "feature-category-header";
        const selectedCount = groupFeatures.filter((feature) => previous.has(feature)).length;
        const title = document.createElement("div");
        title.className = "feature-category-title";
        title.textContent = `${category} (${selectedCount}/${groupFeatures.length} selected)`;

        const actions = document.createElement("div");
        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "mini-btn";
        selectBtn.textContent = "Select category";
        selectBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => mlrIncludedFeatures.add(feature));
          renderMlrFeatureSelector();
          mlrStatusEl.textContent = `Selected ${groupFeatures.length} predictors from ${category}. Click Run Multilinear Regression to refit.`;
        });
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "mini-btn";
        clearBtn.textContent = "Clear category";
        clearBtn.style.marginLeft = "6px";
        clearBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => mlrIncludedFeatures.delete(feature));
          renderMlrFeatureSelector();
          mlrStatusEl.textContent = `Cleared ${groupFeatures.length} predictors from ${category}. Click Run Multilinear Regression to refit.`;
        });
        actions.appendChild(selectBtn);
        actions.appendChild(clearBtn);
        header.appendChild(title);
        header.appendChild(actions);
        categoryBox.appendChild(header);

        for (const feature of groupFeatures) {
          const id = `mlr_feat_${feature.replace(/[^a-zA-Z0-9_]/g, "_")}`;
          const wrapper = document.createElement("label");
          wrapper.dataset.mlrFeatureRow = feature;
          wrapper.style.display = "block";
          wrapper.style.fontSize = "12px";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.id = id;
          input.dataset.mlrFeature = feature;
          input.checked = previous.has(feature);
          wrapper.appendChild(input);
          wrapper.appendChild(document.createTextNode(` ${featureShortLabel(feature, category)}`));
          categoryBox.appendChild(wrapper);
          mlrFeatureElements.set(feature, wrapper);
        }
        columns[renderedCategoryIndex % columns.length].appendChild(categoryBox);
        renderedCategoryIndex += 1;
      }
      mlrIncludedFeatures = new Set(features.filter((feature) => previous.has(feature)));
      mlrFeatureListEl.querySelectorAll('input[data-mlr-feature]').forEach((el) => {
        el.addEventListener("change", () => {
          const feature = el.getAttribute("data-mlr-feature");
          if (!feature) return;
          if (el.checked) mlrIncludedFeatures.add(feature);
          else mlrIncludedFeatures.delete(feature);
          const included = mlrIncludedFeatures.size;
          mlrStatusEl.textContent = `Feature selection changed. ${included} candidate predictors selected. Click Run Multilinear Regression to refit.`;
        });
      });
    }

    function htmlEscape(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function filterFeatureLabel(feature) {
      const category = featureCategory(feature);
      return `${category}: ${featureShortLabel(feature, category)}`;
    }

    function isContinuousFilterFeature(feature) {
      return numericFeatures.includes(feature);
    }

    function uniqueFilterValues(feature) {
      const counts = new Map();
      const sourceRows = plottedRows.length ? plottedRows : allRows;
      for (const row of sourceRows) {
        const raw = row?.[feature];
        if (raw === null || raw === undefined) continue;
        const value = String(raw).trim();
        if (!value) continue;
        counts.set(value, (counts.get(value) || 0) + 1);
      }
      return Array.from(counts.entries())
        .sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true, sensitivity: "base" }))
        .slice(0, 500)
        .map(([value, count]) => ({ value, count }));
    }

    function populateRowFilterFeatureSelector() {
      if (!rowFilterFeature) return;
      const previous = rowFilterFeature.value;
      rowFilterFeature.innerHTML = "";
      if (!rowFilterFeatures.length) {
        rowFilterFeature.innerHTML = '<option value="">No filterable features</option>';
        renderRowFilterValueControl();
        return;
      }
      const groups = groupedFeatures(rowFilterFeatures);
      for (const category of FEATURE_CATEGORY_ORDER) {
        const values = groups[category] || [];
        if (!values.length) continue;
        const group = document.createElement("optgroup");
        group.label = category;
        for (const feature of values) {
          const opt = document.createElement("option");
          opt.value = feature;
          opt.textContent = featureShortLabel(feature, category);
          opt.title = feature;
          group.appendChild(opt);
        }
        rowFilterFeature.appendChild(group);
      }
      if (previous && rowFilterFeatures.includes(previous)) rowFilterFeature.value = previous;
      else rowFilterFeature.value = rowFilterFeatures[0] || "";
      renderRowFilterValueControl();
    }

    function renderRowFilterValueControl() {
      if (!rowFilterFeature || !rowFilterOperator || !rowFilterValue || !rowFilterCategoryValue) return;
      const feature = rowFilterFeature.value || "";
      const continuous = isContinuousFilterFeature(feature);
      rowFilterOperator.innerHTML = "";
      const operators = continuous
        ? [[">", "greater than"], [">=", "greater than or equal"], ["<", "smaller than"], ["<=", "smaller than or equal"]]
        : [["==", "is"], ["!=", "is not"]];
      for (const [value, label] of operators) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = label;
        rowFilterOperator.appendChild(opt);
      }
      rowFilterValue.style.display = continuous ? "" : "none";
      rowFilterCategoryValue.style.display = continuous ? "none" : "";
      if (continuous) {
        rowFilterValue.value = "";
        rowFilterValue.placeholder = "Threshold value";
        return;
      }
      const values = uniqueFilterValues(feature);
      rowFilterCategoryValue.innerHTML = "";
      if (!values.length) {
        rowFilterCategoryValue.innerHTML = '<option value="">No values available</option>';
        return;
      }
      for (const item of values) {
        const opt = document.createElement("option");
        opt.value = item.value;
        opt.textContent = `${item.value} (${item.count})`;
        rowFilterCategoryValue.appendChild(opt);
      }
    }

    function rowMatchesFilterRule(row, rule) {
      if (!rule || !rule.feature) return true;
      if (rule.kind === "continuous") {
        const actual = parseMaybeNumber(row?.[rule.feature]);
        const target = parseMaybeNumber(rule.value);
        if (actual === null || target === null) return false;
        if (rule.operator === ">") return actual > target;
        if (rule.operator === ">=") return actual >= target;
        if (rule.operator === "<") return actual < target;
        if (rule.operator === "<=") return actual <= target;
        return false;
      }
      const actual = row?.[rule.feature] === null || row?.[rule.feature] === undefined ? "" : String(row[rule.feature]);
      const target = String(rule.value ?? "");
      if (rule.operator === "!=") return actual !== target;
      return actual === target;
    }

    function rowHasSelMetadata(row) {
      if (!row || typeof row !== "object") return false;
      if (row["pair.sel_id"] !== undefined && row["pair.sel_id"] !== null && String(row["pair.sel_id"]).trim() !== "") return true;
      if (row["pair.selected_peak_metadata.sel.aggregate_id"] !== undefined && row["pair.selected_peak_metadata.sel.aggregate_id"] !== null && String(row["pair.selected_peak_metadata.sel.aggregate_id"]).trim() !== "") return true;
      const exportFields = row.__export_fields && typeof row.__export_fields === "object" ? row.__export_fields : {};
      if (exportFields["pair.sel_id"] !== undefined && exportFields["pair.sel_id"] !== null && String(exportFields["pair.sel_id"]).trim() !== "") return true;
      return Object.keys(row).some((key) => key.startsWith("pair.selected_peak_metadata.sel."))
        || Object.keys(exportFields).some((key) => key.startsWith("pair.selected_peak_metadata.sel."));
    }

    function rowMatchesSelPresence(row) {
      if (rowSelPresenceMode === "present") return rowHasSelMetadata(row);
      if (rowSelPresenceMode === "absent") return !rowHasSelMetadata(row);
      return true;
    }

    function rowMatchesAllFilterRules(row) {
      return rowMatchesSelPresence(row) && rowFilterRules.every((rule) => rowMatchesFilterRule(row, rule));
    }

    function rowFilterRuleDescription(rule) {
      const feature = filterFeatureLabel(rule.feature);
      const opText = {
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
        "==": "is",
        "!=": "is not",
      }[rule.operator] || rule.operator;
      return `${feature} ${opText} ${rule.value}`;
    }

    function renderRowFilterRules() {
      if (!rowFilterRulesEl) return;
      if (!rowFilterRules.length) {
        rowFilterRulesEl.innerHTML = "";
        return;
      }
      rowFilterRulesEl.innerHTML = rowFilterRules.map((rule) => `
        <div class="row-filter-rule">
          <span>${htmlEscape(rowFilterRuleDescription(rule))}</span>
          <button type="button" class="mini-btn" data-remove-row-filter="${htmlEscape(rule.id)}">Remove</button>
        </div>
      `).join("");
      rowFilterRulesEl.querySelectorAll("[data-remove-row-filter]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const id = btn.getAttribute("data-remove-row-filter");
          rowFilterRules = rowFilterRules.filter((rule) => rule.id !== id);
          renderRowFilterRules();
          recomputeRowVisibility(true);
        });
      });
    }

    function updateRowFilterStatus() {
      if (!rowFilterStatusEl) return;
      if (!plottedRows.length) {
        const hasSelMode = rowSelPresenceMode !== "all";
        rowFilterStatusEl.textContent = rowFilterRules.length || hasSelMode ? `${rowFilterRules.length + (hasSelMode ? 1 : 0)} active filter(s). No plotted rows loaded.` : "No row filters active.";
        return;
      }
      const filterMatched = rowFilterVisibility.filter(Boolean).length;
      const visible = rowVisibility.filter(Boolean).length;
      const selectedSubjects = new Set(
        plottedRows
          .filter((_, idx) => rowVisibility[idx])
          .map((row) => getSubjectId(row))
          .filter((sid) => sid && sid !== "Unknown")
      ).size;
      const activeFilterCount = rowFilterRules.length + (rowSelPresenceMode !== "all" ? 1 : 0);
      rowFilterStatusEl.textContent = activeFilterCount
        ? `${activeFilterCount} active filter(s). Filter matched ${filterMatched} / ${plottedRows.length}; showing ${visible} rows across ${selectedSubjects} unique subject(s) after manual row toggles.`
        : `No row filters active. Showing ${visible} / ${plottedRows.length} rows across ${selectedSubjects} unique subject(s) after manual row toggles.`;
    }

    function recomputeRowVisibility(refresh = true) {
      rowFilterVisibility = plottedRows.map((row) => rowMatchesAllFilterRules(row));
      if (rowManualVisibility.length !== plottedRows.length) rowManualVisibility = plottedRows.map(() => true);
      rowVisibility = plottedRows.map((_, idx) => rowManualVisibility[idx] !== false && rowFilterVisibility[idx] !== false);
      updateRowFilterStatus();
      if (refresh) {
        renderTable(plottedRows, currentXKey, currentYKey);
        rerenderFromVisibility();
      }
    }

    function refreshRowsToAllCases(message = "Refreshed plotted rows to all cases.") {
      plottedRows = allRows.slice();
      rowFilterRules = [];
      rowSelPresenceMode = "all";
      if (rowSelPresenceFilter) rowSelPresenceFilter.value = "all";
      rowManualVisibility = plottedRows.map(() => true);
      rowFilterVisibility = plottedRows.map(() => true);
      rowVisibility = plottedRows.map(() => true);
      renderRowFilterRules();
      renderRowFilterValueControl();
      updateRowStats(plottedRows);
      updateRowFilterStatus();
      renderTable(plottedRows, currentXKey, currentYKey);
      rerenderFromVisibility();
      setStatus(message);
    }

    function addRowFilterRule() {
      const feature = rowFilterFeature.value || "";
      if (!feature) {
        rowFilterStatusEl.textContent = "Select a feature before adding a filter.";
        return;
      }
      const continuous = isContinuousFilterFeature(feature);
      const operator = rowFilterOperator.value || (continuous ? ">" : "==");
      const value = continuous ? parseMaybeNumber(rowFilterValue.value) : rowFilterCategoryValue.value;
      if (continuous && value === null) {
        rowFilterStatusEl.textContent = "Enter a numeric threshold for the continuous filter.";
        return;
      }
      if (!continuous && (value === null || value === undefined || String(value).trim() === "")) {
        rowFilterStatusEl.textContent = "Select a categorical value before adding the filter.";
        return;
      }
      rowFilterRules.push({
        id: `row_filter_${Date.now()}_${rowFilterRules.length + 1}`,
        feature,
        kind: continuous ? "continuous" : "categorical",
        operator,
        value,
      });
      renderRowFilterRules();
      recomputeRowVisibility(true);
    }

    function renderExportFeatureSelector(features) {
      if (!exportFeatureListEl) return;
      exportFeatureListEl.innerHTML = "";
      if (!features.length) {
        exportStatusEl.textContent = "No annotation feature columns available for export.";
        return;
      }

      const groups = groupedFeatures(features);
      let rendered = 0;
      for (const category of FEATURE_CATEGORY_ORDER) {
        const groupFeatures = groups[category] || [];
        if (!groupFeatures.length) continue;
        const categoryBox = document.createElement("div");
        categoryBox.className = "feature-category";

        const header = document.createElement("div");
        header.className = "feature-category-header";
        const selectedCount = groupFeatures.filter((feature) => exportSelectedFeatures.has(feature)).length;
        const title = document.createElement("div");
        title.className = "feature-category-title";
        title.textContent = `${category} (${selectedCount}/${groupFeatures.length} selected)`;

        const actions = document.createElement("div");
        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "mini-btn";
        selectBtn.textContent = "Select";
        selectBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => exportSelectedFeatures.add(feature));
          renderExportFeatureSelector(features);
        });
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "mini-btn";
        clearBtn.textContent = "Clear";
        clearBtn.style.marginLeft = "6px";
        clearBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => exportSelectedFeatures.delete(feature));
          renderExportFeatureSelector(features);
        });
        actions.appendChild(selectBtn);
        actions.appendChild(clearBtn);
        header.appendChild(title);
        header.appendChild(actions);
        categoryBox.appendChild(header);

        for (const feature of groupFeatures) {
          const id = `export_feat_${feature.replace(/[^a-zA-Z0-9_]/g, "_")}`;
          const wrapper = document.createElement("label");
          wrapper.style.display = "block";
          wrapper.style.fontSize = "12px";
          wrapper.title = feature;
          const input = document.createElement("input");
          input.type = "checkbox";
          input.id = id;
          input.dataset.exportFeature = feature;
          input.checked = exportSelectedFeatures.has(feature);
          wrapper.appendChild(input);
          wrapper.appendChild(document.createTextNode(` ${featureShortLabel(feature, category)}`));
          categoryBox.appendChild(wrapper);
        }
        exportFeatureListEl.appendChild(categoryBox);
        rendered += 1;
      }

      exportFeatureListEl.querySelectorAll('input[data-export-feature]').forEach((el) => {
        el.addEventListener("change", () => {
          const feature = el.getAttribute("data-export-feature");
          if (!feature) return;
          if (el.checked) exportSelectedFeatures.add(feature);
          else exportSelectedFeatures.delete(feature);
          updateExportStatus();
        });
      });
      if (!rendered) exportFeatureListEl.innerHTML = `<div class="small">No grouped numeric features available.</div>`;
      updateExportStatus();
    }

    function updateExportStatus(message = "") {
      if (!exportStatusEl) return;
      const selectedCount = Array.from(exportSelectedFeatures).filter((feature) => exportFeatures.includes(feature)).length;
      exportStatusEl.textContent = message || `${allRows.length} observation rows ready. ${selectedCount} feature columns selected.`;
    }

    function exportIdentifierColumns(rows) {
      const candidates = ["case_path", "instance_id", "pair_name", "category", "panel", "subject_id"];
      return candidates.filter((col) => rows.some((row) => row[col] !== undefined && row[col] !== null && String(row[col]).trim() !== ""));
    }

    function csvCell(value) {
      if (value === undefined || value === null) return "NA";
      if (typeof value === "number" && !Number.isFinite(value)) return "NA";
      const text = String(value);
      if (!text.trim()) return "NA";
      if (/[",\\n\\r]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
      return text;
    }

    function buildSelectedFeaturesCsv(rows = allRows, selectedFeatures = Array.from(exportSelectedFeatures)) {
      const features = selectedFeatures.filter((feature) => exportFeatures.includes(feature));
      const columns = [...exportIdentifierColumns(rows), ...features];
      if (!rows.length || !columns.length) return "";
      const lines = [columns.map(csvCell).join(",")];
      for (const row of rows) {
        lines.push(columns.map((col) => {
          const value = row[col] !== undefined ? row[col] : row.__export_fields?.[col];
          return csvCell(value);
        }).join(","));
      }
      return lines.join("\\n");
    }

    function buildRowsCsv(rows, preferredColumns = []) {
      if (!Array.isArray(rows) || !rows.length) return "";
      const columns = [];
      for (const col of preferredColumns) {
        if (col && !columns.includes(col)) columns.push(col);
      }
      for (const row of rows) {
        for (const col of Object.keys(row || {})) {
          if (!columns.includes(col)) columns.push(col);
        }
      }
      const lines = [columns.map(csvCell).join(",")];
      for (const row of rows) {
        lines.push(columns.map((col) => csvCell(row?.[col])).join(","));
      }
      return lines.join("\\n");
    }

    function downloadCsv(csv, filenameStem) {
      if (!csv) return false;
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${filenameStem}_${stamp}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      return true;
    }

    function downloadJson(payload, filenameStem) {
      if (!payload) return false;
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${filenameStem}_${stamp}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      return true;
    }

    function exportSelectedFeaturesCsv() {
      const selected = Array.from(exportSelectedFeatures).filter((feature) => exportFeatures.includes(feature));
      if (!allRows.length) {
        updateExportStatus("No observation rows loaded. Load an annotation JSON first.");
        return;
      }
      if (!selected.length) {
        updateExportStatus("Select at least one feature column before exporting.");
        return;
      }
      const csv = buildSelectedFeaturesCsv(allRows, selected);
      if (!csv) {
        updateExportStatus("CSV export could not be built from the current rows.");
        return;
      }
      downloadCsv(csv, "analysis_selected_features");
      updateExportStatus(`Exported ${allRows.length} rows with ${selected.length} selected feature columns.`);
    }

    function focusMlrFeature(feature) {
      if (!feature || !mlrFeatureListEl) return;
      mlrFeatureElements.forEach((el) => el.classList.remove("feature-hit"));
      const row = mlrFeatureElements.get(feature);
      if (!row) {
        mlrStatusEl.textContent = `Feature is not currently in the candidate predictor list: ${feature}`;
        return;
      }
      row.classList.add("feature-hit");
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      const input = row.querySelector('input[data-mlr-feature]');
      if (input) input.focus({ preventScroll: true });
      mlrStatusEl.textContent = `Located predictor in selection list: ${feature}`;
      window.setTimeout(() => row.classList.remove("feature-hit"), 3500);
    }

    function populateDistFeatureSelector(features) {
      distFeatureOptions = [...features];
      const dflt = features.find((f) => f === "lab_features.hba1c") || features[0];
      populateGroupedFeaturePicker(distFeatureGroup, distFeature, features, dflt);
    }

    function getSubjectId(row) {
      if (row.subject_id !== undefined && row.subject_id !== null && String(row.subject_id).trim() !== "") {
        return String(row.subject_id);
      }
      return extractSubjectIdFromCasePath(row.case_path || "") || "Unknown";
    }

    function mean(arr) {
      if (!arr.length) return null;
      return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    function median(arr) {
      const vals = arr.filter((value) => value !== null && Number.isFinite(value)).slice().sort((a, b) => a - b);
      if (!vals.length) return null;
      const mid = Math.floor(vals.length / 2);
      return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
    }

    function sampleStd(arr) {
      if (arr.length < 2) return 0;
      const m = mean(arr);
      const v = arr.reduce((s, x) => s + (x - m) * (x - m), 0) / (arr.length - 1);
      return Math.sqrt(v);
    }

    function summarizeCompleteCase(meta) {
      if (!meta) return "";
      const droppedRows = meta.droppedRows?.length || 0;
      const droppedFeatures = meta.droppedFeatures?.length || 0;
      return `Complete-case analysis: used ${meta.usedRowCount || 0}/${meta.rowCount || 0} selected rows with observed values only; skipped ${droppedRows} rows with missing selected values and ${droppedFeatures} constant/unusable features.`;
    }

    function buildCompleteCaseNumericMatrix(rows, candidateFeatures, options = {}) {
      const standardize = options.standardize !== false;
      const sourceIndices = options.sourceIndices || rows.map((_, idx) => idx);
      const rawMatrix = [];
      const keptRows = [];
      const keptSourceIndices = [];
      const droppedRows = [];
      const droppedFeatures = [];

      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        const values = [];
        const missing = [];
        for (const feature of candidateFeatures) {
          const value = parseMaybeNumber(row[feature]);
          if (value === null) missing.push(feature);
          else values.push(value);
        }
        if (missing.length) {
          droppedRows.push({
            row_index: sourceIndices[i],
            case_path: row.case_path || "",
            instance_id: row.instance_id || "",
            pair_name: row.pair_name || "",
            category: row.category || "",
            panel: row.panel || "",
            missing_features: missing.join(", "),
          });
          continue;
        }
        rawMatrix.push(values);
        keptRows.push(row);
        keptSourceIndices.push(sourceIndices[i]);
      }

      if (!rawMatrix.length) {
        return {
          matrix: [],
          rows: [],
          rowIndices: [],
          features: [],
          droppedRows,
          droppedFeatures: candidateFeatures.map((feature) => ({ feature, reason: "no complete observed rows" })),
          rowCount: rows.length,
          usedRowCount: 0,
        };
      }

      const keepCols = [];
      const means = [];
      const stds = [];
      for (let j = 0; j < candidateFeatures.length; j++) {
        const vals = rawMatrix.map((row) => row[j]);
        const m = mean(vals);
        const sd = sampleStd(vals);
        if (!Number.isFinite(sd) || sd <= 1e-12) {
          droppedFeatures.push({ feature: candidateFeatures[j], reason: "no variation in complete rows" });
          continue;
        }
        keepCols.push(j);
        means.push(m);
        stds.push(sd);
      }

      const features = keepCols.map((idx) => candidateFeatures[idx]);
      const matrix = rawMatrix.map((row) =>
        keepCols.map((colIdx, outIdx) => standardize ? (row[colIdx] - means[outIdx]) / stds[outIdx] : row[colIdx])
      );

      return {
        matrix,
        rows: keptRows,
        rowIndices: keptSourceIndices,
        features,
        means,
        stds,
        droppedRows,
        droppedFeatures,
        rowCount: rows.length,
        usedRowCount: keptRows.length,
      };
    }

    function buildCompleteCaseRowsForRegression(rows, responseFeature, candidateFeatures) {
      const features = [responseFeature].concat(candidateFeatures);
      const complete = buildCompleteCaseNumericMatrix(rows, features, {
        standardize: true,
      });
      if (!complete.features.includes(responseFeature)) {
        return {
          rows: [],
          predictors: [],
          completeCaseMeta: complete,
        };
      }
      const modelRows = complete.rows.map((row, rowIdx) => {
        const out = { __source_row: row };
        complete.features.forEach((feature, colIdx) => {
          out[feature] = complete.matrix[rowIdx][colIdx];
        });
        return out;
      });
      return {
        rows: modelRows,
        predictors: complete.features.filter((feature) => feature !== responseFeature),
        completeCaseMeta: complete,
      };
    }

    function populateCohortFeatureSelector(features) {
      const defaultF = features.find((f) => f === "lab_features.hba1c") || features[0];
      populateGroupedFeaturePicker(cohortFeatureGroup, cohortFeature, features, defaultF, ["Subject metadata"]);
    }

    function rerenderCohortHistogram() {
      const feature = cohortFeature.value;
      const subjects = Object.keys(subjectLabMap || {}).sort();
      if (!feature || !subjects.length) {
        cohortStatsEl.textContent = "No cohort lab data available.";
        safeNewPlot("cohortPlot", [], { title: "No cohort data" }, { responsive: true });
        return;
      }

      const allVals = [];
      const subVals = {};
      for (const sid of subjects) {
        const v = parseMaybeNumber(subjectLabMap[sid]?.[feature]);
        if (v === null) continue;
        allVals.push(v);
        subVals[sid] = v;
      }
      if (!allVals.length) {
        cohortStatsEl.textContent = `No numeric values for ${feature} across subjects.`;
        safeNewPlot("cohortPlot", [], { title: "No numeric data" }, { responsive: true });
        return;
      }

      const selectedSids = new Set(
        plottedRows
          .filter((_, i) => rowVisibility[i])
          .map((r) => getSubjectId(r))
          .filter((sid) => !!sid)
      );
      const selected = Array.from(selectedSids).filter((sid) => subVals[sid] !== undefined);

      const traces = [
        {
          type: "histogram",
          x: allVals,
          nbinsx: Math.min(15, Math.max(6, Math.round(Math.sqrt(allVals.length)))),
          marker: { color: "rgba(59,130,246,0.55)" },
          name: "All subjects",
        },
      ];

      const shapes = [];
      const annotations = [];
      selected.forEach((sid, idx) => {
        const x = subVals[sid];
        shapes.push({
          type: "line",
          x0: x,
          x1: x,
          y0: 0,
          y1: 1,
          xref: "x",
          yref: "paper",
          line: { color: "rgba(220,38,38,0.85)", width: 2, dash: "dash" },
        });
        annotations.push({
          x: x,
          y: 1.02 + (idx % 2) * 0.06,
          xref: "x",
          yref: "paper",
          text: sid,
          showarrow: false,
          font: { size: 10, color: "rgba(153,27,27,0.95)" },
        });
      });

      const sortedVals = [...allVals].sort((a, b) => a - b);
      const median = sortedVals[Math.floor(sortedVals.length / 2)];
      const selectedDesc = selected.length
        ? selected.map((sid) => `${sid}:${subVals[sid].toFixed(3)}`).join(", ")
        : "none";
      cohortStatsEl.textContent =
        `Feature: ${feature} | Cohort subjects with value: ${allVals.length} | Median: ${median.toFixed(3)} | Selected subjects: ${selectedDesc}`;

      safeNewPlot("cohortPlot", traces, {
        title: `${feature}: selected subject positions within all subjects`,
        xaxis: { title: feature },
        yaxis: { title: "Count" },
        shapes,
        annotations,
        margin: { t: 58, r: 20, b: 56, l: 64 },
      }, { responsive: true });
    }

    function rerenderFeatureDistribution() {
      const feature = distFeature.value;
      const level = distLevel.value || "both";
      const visibleRows = plottedRows.filter((_, i) => rowVisibility[i]);
      if (!feature) {
        distStatsEl.textContent = "Select a feature.";
        safeNewPlot("distPlot", [], { title: "No feature selected" }, { responsive: true });
        safeNewPlot("distHistPlot", [], { title: "No feature selected" }, { responsive: true });
        return;
      }
      const boxTraces = [];
      const histTraces = [];

      // Individual-level distribution from visible rows, grouped by subject.
      const bySubject = {};
      for (const row of visibleRows) {
        const v = parseMaybeNumber(row[feature]);
        if (v === null) continue;
        const sid = getSubjectId(row);
        if (!bySubject[sid]) bySubject[sid] = [];
        bySubject[sid].push(v);
      }
      const individualSubjects = Object.keys(bySubject).sort();
      const individualVals = [];
      if (level === "individual" || level === "both") {
        for (const sid of individualSubjects) {
          individualVals.push(...bySubject[sid]);
          boxTraces.push({
            type: "box",
            name: `ind:${sid}`,
            y: bySubject[sid],
            boxpoints: "all",
            jitter: 0.25,
            pointpos: 0,
            marker: { size: 4, opacity: 0.6 },
          });
        }
        if (individualVals.length) {
          histTraces.push({
            type: "histogram",
            name: "individual rows",
            x: individualVals,
            opacity: 0.65,
            nbinsx: Math.min(30, Math.max(6, Math.round(Math.sqrt(individualVals.length)))),
            marker: { color: "rgba(14,116,144,0.62)" },
          });
        }
      }

      // Cohort-level distribution from all subjects' lab data.
      const cohortVals = Object.values(subjectLabMap || {})
        .map((m) => parseMaybeNumber(m?.[feature]))
        .filter((v) => v !== null);
      if ((level === "cohort" || level === "both") && cohortVals.length) {
        boxTraces.push({
          type: "box",
          name: "cohort",
          y: cohortVals,
          boxpoints: "all",
          jitter: 0.2,
          pointpos: 0,
          marker: { size: 4, opacity: 0.55, color: "rgba(37,99,235,0.8)" },
          line: { color: "rgba(37,99,235,0.9)" },
        });
        histTraces.push({
          type: "histogram",
          name: "cohort",
          x: cohortVals,
          opacity: 0.5,
          nbinsx: Math.min(30, Math.max(6, Math.round(Math.sqrt(cohortVals.length)))),
          marker: { color: "rgba(37,99,235,0.58)" },
        });
      }

      if (!boxTraces.length && !histTraces.length) {
        distStatsEl.textContent = `No numeric data available for feature ${feature} at selected level (${level}).`;
        safeNewPlot("distPlot", [], { title: "No numeric data" }, { responsive: true });
        safeNewPlot("distHistPlot", [], { title: "No numeric data" }, { responsive: true });
        return;
      }

      const indN = individualSubjects.reduce((n, sid) => n + bySubject[sid].length, 0);
      const subjectMeans = individualSubjects.map((sid) => mean(bySubject[sid])).filter((v) => v !== null);
      const subjectStds = individualSubjects.map((sid) => sampleStd(bySubject[sid]));
      const interSubjectSD = sampleStd(subjectMeans);
      const intraSubjectSD = mean(subjectStds);
      distStatsEl.textContent =
        `Feature: ${feature} | Level: ${level} | Individual subjects: ${individualSubjects.length}, points: ${indN} | Cohort subjects with value: ${cohortVals.length} | Inter-subject SD (subject means): ${interSubjectSD.toFixed(4)} | Intra-subject SD (avg within subject): ${intraSubjectSD === null ? "-" : intraSubjectSD.toFixed(4)}`;

      safeNewPlot("distPlot", boxTraces, {
        title: `${feature} box/whisker distribution (${level})`,
        yaxis: { title: feature },
        xaxis: { title: "" },
        margin: { t: 48, r: 20, b: 40, l: 64 },
      }, { responsive: true });
      safeNewPlot("distHistPlot", histTraces, {
        title: `${feature} histogram distribution (${level})`,
        xaxis: { title: feature },
        yaxis: { title: "Count" },
        barmode: "overlay",
        bargap: 0.05,
        margin: { t: 48, r: 20, b: 56, l: 64 },
      }, { responsive: true });
    }

    function renderTsneFeatureSelector(features) {
      tsneFeatureListEl.innerHTML = "";
      const groups = groupedFeatures(features);
      const categories = orderedVisibleFeatureCategories(groups);
      const columns = createPredictorColumns(tsneFeatureListEl, categories.length);
      let renderedCategoryIndex = 0;
      for (const category of categories) {
        const groupFeatures = groups[category] || [];
        if (!groupFeatures.length) continue;
        const categoryBox = document.createElement("div");
        categoryBox.className = "feature-category";

        const header = document.createElement("div");
        header.className = "feature-category-header";
        const selectedCount = groupFeatures.filter((feature) => tsneSelectedFeatures.has(feature)).length;
        const title = document.createElement("div");
        title.className = "feature-category-title";
        title.textContent = `${category} (${selectedCount}/${groupFeatures.length} selected)`;

        const actions = document.createElement("div");
        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "mini-btn";
        selectBtn.textContent = "Select category";
        selectBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => tsneSelectedFeatures.add(feature));
          renderTsneFeatureSelector(features);
          tsneStatusEl.textContent = `Selected ${groupFeatures.length} ${category} features. Click Run t-SNE to recompute.`;
        });
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "mini-btn";
        clearBtn.textContent = "Clear category";
        clearBtn.style.marginLeft = "6px";
        clearBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => tsneSelectedFeatures.delete(feature));
          renderTsneFeatureSelector(features);
          tsneStatusEl.textContent = `Cleared ${groupFeatures.length} ${category} features. Click Run t-SNE to recompute.`;
        });
        actions.appendChild(selectBtn);
        actions.appendChild(clearBtn);
        header.appendChild(title);
        header.appendChild(actions);
        categoryBox.appendChild(header);

        for (const feature of groupFeatures) {
          const id = `tsne_feat_${feature.replace(/[^a-zA-Z0-9_]/g, "_")}`;
          const wrapper = document.createElement("label");
          wrapper.style.display = "block";
          wrapper.style.fontSize = "12px";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.id = id;
          input.dataset.tsneFeature = feature;
          input.checked = tsneSelectedFeatures.has(feature);
          wrapper.appendChild(input);
          wrapper.appendChild(document.createTextNode(` ${featureShortLabel(feature, category)}`));
          categoryBox.appendChild(wrapper);
        }
        columns[renderedCategoryIndex % columns.length].appendChild(categoryBox);
        renderedCategoryIndex += 1;
      }
      tsneFeatureListEl.querySelectorAll('input[data-tsne-feature]').forEach((el) => {
        el.addEventListener("change", () => {
          const f = el.getAttribute("data-tsne-feature");
          if (!f) return;
          if (el.checked) tsneSelectedFeatures.add(f);
          else tsneSelectedFeatures.delete(f);
          tsneStatusEl.textContent = "t-SNE feature selection changed. Click Run t-SNE to recompute.";
        });
      });
    }

    function renderPcaFeatureSelector(features) {
      if (!pcaFeatureListEl) return;
      pcaFeatureListEl.innerHTML = "";
      const groups = groupedFeatures(features);
      const categories = orderedVisibleFeatureCategories(groups);
      const columns = createPredictorColumns(pcaFeatureListEl, categories.length);
      let renderedCategoryIndex = 0;
      for (const category of categories) {
        const groupFeatures = groups[category] || [];
        if (!groupFeatures.length) continue;
        const categoryBox = document.createElement("div");
        categoryBox.className = "feature-category";

        const header = document.createElement("div");
        header.className = "feature-category-header";
        const selectedCount = groupFeatures.filter((feature) => pcaSelectedFeatures.has(feature)).length;
        const title = document.createElement("div");
        title.className = "feature-category-title";
        title.textContent = `${category} (${selectedCount}/${groupFeatures.length} selected)`;

        const actions = document.createElement("div");
        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "mini-btn";
        selectBtn.textContent = "Select category";
        selectBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => pcaSelectedFeatures.add(feature));
          renderPcaFeatureSelector(features);
          pcaStatusEl.textContent = `Selected ${groupFeatures.length} ${category} features. Click Run PCA to recompute.`;
        });
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "mini-btn";
        clearBtn.textContent = "Clear category";
        clearBtn.style.marginLeft = "6px";
        clearBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => pcaSelectedFeatures.delete(feature));
          renderPcaFeatureSelector(features);
          pcaStatusEl.textContent = `Cleared ${groupFeatures.length} ${category} features. Click Run PCA to recompute.`;
        });
        actions.appendChild(selectBtn);
        actions.appendChild(clearBtn);
        header.appendChild(title);
        header.appendChild(actions);
        categoryBox.appendChild(header);

        for (const feature of groupFeatures) {
          const id = `pca_feat_${feature.replace(/[^a-zA-Z0-9_]/g, "_")}`;
          const wrapper = document.createElement("label");
          wrapper.style.display = "block";
          wrapper.style.fontSize = "12px";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.id = id;
          input.dataset.pcaFeature = feature;
          input.checked = pcaSelectedFeatures.has(feature);
          wrapper.appendChild(input);
          wrapper.appendChild(document.createTextNode(` ${featureShortLabel(feature, category)}`));
          categoryBox.appendChild(wrapper);
        }
        columns[renderedCategoryIndex % columns.length].appendChild(categoryBox);
        renderedCategoryIndex += 1;
      }
      pcaFeatureListEl.querySelectorAll('input[data-pca-feature]').forEach((el) => {
        el.addEventListener("change", () => {
          const f = el.getAttribute("data-pca-feature");
          if (!f) return;
          if (el.checked) pcaSelectedFeatures.add(f);
          else pcaSelectedFeatures.delete(f);
          pcaStatusEl.textContent = "PCA feature selection changed. Click Run PCA to recompute.";
        });
      });
    }

    function renderClassifierFeatureSelector(features) {
      if (!classifierFeatureListEl) return;
      classifierFeatureListEl.innerHTML = "";
      const groups = groupedFeatures(features);
      const categories = orderedVisibleFeatureCategories(groups);
      const columns = createPredictorColumns(classifierFeatureListEl, categories.length);
      let renderedCategoryIndex = 0;
      for (const category of categories) {
        const groupFeatures = groups[category] || [];
        if (!groupFeatures.length) continue;
        const categoryBox = document.createElement("div");
        categoryBox.className = "feature-category";

        const header = document.createElement("div");
        header.className = "feature-category-header";
        const selectedCount = groupFeatures.filter((feature) => classifierSelectedFeatures.has(feature)).length;
        const title = document.createElement("div");
        title.className = "feature-category-title";
        title.textContent = `${category} (${selectedCount}/${groupFeatures.length} selected)`;

        const actions = document.createElement("div");
        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "mini-btn";
        selectBtn.textContent = "Select category";
        selectBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => classifierSelectedFeatures.add(feature));
          renderClassifierFeatureSelector(features);
          classifierStatusEl.textContent = `Selected ${groupFeatures.length} classifier features from ${category}.`;
        });
        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "mini-btn";
        clearBtn.textContent = "Clear category";
        clearBtn.style.marginLeft = "6px";
        clearBtn.addEventListener("click", () => {
          groupFeatures.forEach((feature) => classifierSelectedFeatures.delete(feature));
          renderClassifierFeatureSelector(features);
          classifierStatusEl.textContent = `Cleared ${groupFeatures.length} classifier features from ${category}.`;
        });
        actions.appendChild(selectBtn);
        actions.appendChild(clearBtn);
        header.appendChild(title);
        header.appendChild(actions);
        categoryBox.appendChild(header);

        for (const feature of groupFeatures) {
          const id = `classifier_feat_${feature.replace(/[^a-zA-Z0-9_]/g, "_")}`;
          const wrapper = document.createElement("label");
          wrapper.style.display = "block";
          wrapper.style.fontSize = "12px";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.id = id;
          input.dataset.classifierFeature = feature;
          input.checked = classifierSelectedFeatures.has(feature);
          wrapper.appendChild(input);
          wrapper.appendChild(document.createTextNode(` ${featureShortLabel(feature, category)}`));
          categoryBox.appendChild(wrapper);
        }
        columns[renderedCategoryIndex % columns.length].appendChild(categoryBox);
        renderedCategoryIndex += 1;
      }
      classifierFeatureListEl.querySelectorAll('input[data-classifier-feature]').forEach((el) => {
        el.addEventListener("change", () => {
          const f = el.getAttribute("data-classifier-feature");
          if (!f) return;
          if (el.checked) classifierSelectedFeatures.add(f);
          else classifierSelectedFeatures.delete(f);
          classifierStatusEl.textContent = "Classifier feature selection changed. Click Run Classification to refit.";
        });
      });
    }

    function defaultTsneColorFeatureLabel(feature) {
      return featureShortLabel(feature, featureCategory(feature));
    }

    function normalizedTsneColorRule(rule) {
      if (!rule || rule.enabled === false || !rule.label) return null;
      let lowerOperator = rule.lowerOperator || "";
      let lowerValue = parseMaybeNumber(rule.lowerValue);
      let upperOperator = rule.upperOperator || "";
      let upperValue = parseMaybeNumber(rule.upperValue);

      // Backward compatibility for older one-threshold rules.
      if (lowerValue === null && upperValue === null && rule.operator && parseMaybeNumber(rule.value) !== null) {
        if (rule.operator === ">=" || rule.operator === ">") {
          lowerOperator = rule.operator;
          lowerValue = parseMaybeNumber(rule.value);
        } else if (rule.operator === "<=" || rule.operator === "<") {
          upperOperator = rule.operator;
          upperValue = parseMaybeNumber(rule.value);
        }
      }

      const hasLower = (lowerOperator === ">" || lowerOperator === ">=") && lowerValue !== null;
      const hasUpper = (upperOperator === "<" || upperOperator === "<=") && upperValue !== null;
      if (!hasLower && !hasUpper) return null;
      return { ...rule, lowerOperator, lowerValue, upperOperator, upperValue, hasLower, hasUpper };
    }

    function tsneColorRuleMatches(rawValue, rule) {
      if (rawValue === null || !rule) return false;
      if (rule.hasLower) {
        if (rule.lowerOperator === ">" && !(rawValue > rule.lowerValue)) return false;
        if (rule.lowerOperator === ">=" && !(rawValue >= rule.lowerValue)) return false;
      }
      if (rule.hasUpper) {
        if (rule.upperOperator === "<" && !(rawValue < rule.upperValue)) return false;
        if (rule.upperOperator === "<=" && !(rawValue <= rule.upperValue)) return false;
      }
      return true;
    }

    function tsneColorRuleDescription(rule) {
      const parts = [];
      const normalized = normalizedTsneColorRule(rule);
      if (!normalized) return "no active bound";
      if (normalized.hasLower) parts.push(`${normalized.lowerOperator} ${normalized.lowerValue}`);
      if (normalized.hasUpper) parts.push(`${normalized.upperOperator} ${normalized.upperValue}`);
      return parts.join(" and ");
    }

    function buildRuleColorGroups(rows, colorFeature, rules) {
      const feature = colorFeature || "";
      const assignments = [];
      const groups = new Map();
      const activeRules = (rules || [])
        .map(normalizedTsneColorRule)
        .filter((rule) => rule !== null);

      function addGroup(key, label, color) {
        if (!groups.has(key)) {
          groups.set(key, {
            key,
            label,
            color,
            count: 0,
          });
        }
        groups.get(key).count += 1;
      }

      for (let i = 0; i < rows.length; i++) {
        const rawValue = parseMaybeNumber(rows[i]?.[feature]);
        let match = null;
        if (rawValue !== null) {
          for (const rule of activeRules) {
            if (tsneColorRuleMatches(rawValue, rule)) {
              match = rule;
              break;
            }
          }
        }
        if (match) {
          const key = match.id;
          addGroup(key, match.label, match.color || "#1f77b4");
          assignments.push({ key, rawValue });
        } else {
          const key = "__unlabeled__";
          addGroup(key, "Unlabeled", "#9ca3af");
          assignments.push({ key, rawValue: rawValue === null ? "missing" : rawValue });
        }
      }
      return { assignments, groups: Array.from(groups.values()) };
    }

    function buildTsneColorGroups(rows, colorFeature) {
      return buildRuleColorGroups(rows, colorFeature, tsneColorRules);
    }

    function buildClassifierClassGroups(rows, classFeature) {
      return buildRuleColorGroups(rows, classFeature, classifierClassRules);
    }

    function buildPcaColorGroups(rows, colorFeature) {
      return buildRuleColorGroups(rows, colorFeature, pcaColorRules);
    }

    function renderTsneColorRuleTable(groups = []) {
      const counts = new Map((groups || []).map((group) => [group.key, group.count]));
      tsneColorHead.innerHTML = ["enabled", "label", "lower", "lower value", "upper", "upper value", "color", "matched", ""].map((c) => `<th>${c}</th>`).join("");
      if (!tsneColorRules.length) {
        tsneColorBody.innerHTML = `<tr><td colspan="9">No color rules yet. Add a rule to color selected points; otherwise points stay unlabeled/gray.</td></tr>`;
        tsneColorLabelStatusEl.textContent = "No color rules defined. Points without a matching rule are shown as Unlabeled.";
        return;
      }
      tsneColorBody.innerHTML = "";
      for (const rule of tsneColorRules) {
        const tr = document.createElement("tr");
        const enabledTd = document.createElement("td");
        const enabled = document.createElement("input");
        enabled.type = "checkbox";
        enabled.checked = rule.enabled !== false;
        enabled.addEventListener("change", () => {
          rule.enabled = enabled.checked;
          if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
        });
        enabledTd.appendChild(enabled);

        const labelTd = document.createElement("td");
        const labelInput = document.createElement("input");
        labelInput.type = "text";
        labelInput.value = rule.label || "";
        labelInput.style.width = "120px";
        labelInput.addEventListener("change", () => {
          rule.label = labelInput.value.trim() || "Label";
          if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
        });
        labelTd.appendChild(labelInput);

        if (!rule.lowerOperator && !rule.upperOperator && rule.operator && rule.value !== undefined) {
          if (rule.operator === ">=" || rule.operator === ">") {
            rule.lowerOperator = rule.operator;
            rule.lowerValue = rule.value;
          } else if (rule.operator === "<=" || rule.operator === "<") {
            rule.upperOperator = rule.operator;
            rule.upperValue = rule.value;
          }
        }

        const lowerOpTd = document.createElement("td");
        const lowerOp = document.createElement("select");
        ["", ">", ">="].forEach((value) => {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = value || "none";
          lowerOp.appendChild(opt);
        });
        lowerOp.value = rule.lowerOperator || "";
        lowerOp.addEventListener("change", () => {
          rule.lowerOperator = lowerOp.value;
          if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
        });
        lowerOpTd.appendChild(lowerOp);

        const lowerValueTd = document.createElement("td");
        const lowerValueInput = document.createElement("input");
        lowerValueInput.type = "number";
        lowerValueInput.step = "any";
        lowerValueInput.value = rule.lowerValue ?? "";
        lowerValueInput.style.width = "90px";
        lowerValueInput.placeholder = "min";
        lowerValueInput.addEventListener("change", () => {
          rule.lowerValue = lowerValueInput.value;
          if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
        });
        lowerValueTd.appendChild(lowerValueInput);

        const upperOpTd = document.createElement("td");
        const upperOp = document.createElement("select");
        ["", "<", "<="].forEach((value) => {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = value || "none";
          upperOp.appendChild(opt);
        });
        upperOp.value = rule.upperOperator || "";
        upperOp.addEventListener("change", () => {
          rule.upperOperator = upperOp.value;
          if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
        });
        upperOpTd.appendChild(upperOp);

        const upperValueTd = document.createElement("td");
        const upperValueInput = document.createElement("input");
        upperValueInput.type = "number";
        upperValueInput.step = "any";
        upperValueInput.value = rule.upperValue ?? "";
        upperValueInput.style.width = "90px";
        upperValueInput.placeholder = "max";
        upperValueInput.addEventListener("change", () => {
          rule.upperValue = upperValueInput.value;
          if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
        });
        upperValueTd.appendChild(upperValueInput);

        const colorTd = document.createElement("td");
        const color = document.createElement("input");
        color.type = "color";
        color.value = rule.color || "#1f77b4";
        color.addEventListener("change", () => {
          rule.color = color.value;
          if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
        });
        colorTd.appendChild(color);

        const countTd = document.createElement("td");
        countTd.textContent = String(counts.get(rule.id) || 0);

        const actionTd = document.createElement("td");
        const del = document.createElement("button");
        del.type = "button";
        del.textContent = "Delete";
        del.className = "mini-btn";
        del.addEventListener("click", () => {
          tsneColorRules = tsneColorRules.filter((item) => item.id !== rule.id);
          if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
          else renderTsneColorRuleTable();
        });
        actionTd.appendChild(del);

        tr.appendChild(enabledTd);
        tr.appendChild(labelTd);
        tr.appendChild(lowerOpTd);
        tr.appendChild(lowerValueTd);
        tr.appendChild(upperOpTd);
        tr.appendChild(upperValueTd);
        tr.appendChild(colorTd);
        tr.appendChild(countTd);
        tr.appendChild(actionTd);
        tsneColorBody.appendChild(tr);
      }
      const unlabeled = counts.get("__unlabeled__") || 0;
      tsneColorLabelStatusEl.textContent = `${tsneColorRules.length} user-defined rules. Unmatched points: ${unlabeled}.`;
    }

    function renderClassifierClassRuleTable(groups = []) {
      const counts = new Map((groups || []).map((group) => [group.key, group.count]));
      classifierClassRuleHead.innerHTML = ["enabled", "class label", "lower", "lower value", "upper", "upper value", "color", "matched", ""].map((c) => `<th>${c}</th>`).join("");
      if (!classifierClassRules.length) {
        classifierClassRuleBody.innerHTML = `<tr><td colspan="9">No classification class rules yet. Add a class rule before running classification.</td></tr>`;
        classifierClassRuleStatusEl.textContent = "No classification-specific classes defined yet.";
        return;
      }
      classifierClassRuleBody.innerHTML = "";
      for (const rule of classifierClassRules) {
        const tr = document.createElement("tr");
        const enabledTd = document.createElement("td");
        const enabled = document.createElement("input");
        enabled.type = "checkbox";
        enabled.checked = rule.enabled !== false;
        enabled.addEventListener("change", () => {
          rule.enabled = enabled.checked;
          renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
        });
        enabledTd.appendChild(enabled);

        const labelTd = document.createElement("td");
        const labelInput = document.createElement("input");
        labelInput.type = "text";
        labelInput.value = rule.label || "";
        labelInput.style.width = "120px";
        labelInput.addEventListener("change", () => {
          rule.label = labelInput.value.trim() || "Class";
          renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
        });
        labelTd.appendChild(labelInput);

        const lowerOpTd = document.createElement("td");
        const lowerOp = document.createElement("select");
        ["", ">", ">="].forEach((value) => {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = value || "none";
          lowerOp.appendChild(opt);
        });
        lowerOp.value = rule.lowerOperator || "";
        lowerOp.addEventListener("change", () => {
          rule.lowerOperator = lowerOp.value;
          renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
        });
        lowerOpTd.appendChild(lowerOp);

        const lowerValueTd = document.createElement("td");
        const lowerValueInput = document.createElement("input");
        lowerValueInput.type = "number";
        lowerValueInput.step = "any";
        lowerValueInput.value = rule.lowerValue ?? "";
        lowerValueInput.style.width = "90px";
        lowerValueInput.placeholder = "min";
        lowerValueInput.addEventListener("change", () => {
          rule.lowerValue = lowerValueInput.value;
          renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
        });
        lowerValueTd.appendChild(lowerValueInput);

        const upperOpTd = document.createElement("td");
        const upperOp = document.createElement("select");
        ["", "<", "<="].forEach((value) => {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = value || "none";
          upperOp.appendChild(opt);
        });
        upperOp.value = rule.upperOperator || "";
        upperOp.addEventListener("change", () => {
          rule.upperOperator = upperOp.value;
          renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
        });
        upperOpTd.appendChild(upperOp);

        const upperValueTd = document.createElement("td");
        const upperValueInput = document.createElement("input");
        upperValueInput.type = "number";
        upperValueInput.step = "any";
        upperValueInput.value = rule.upperValue ?? "";
        upperValueInput.style.width = "90px";
        upperValueInput.placeholder = "max";
        upperValueInput.addEventListener("change", () => {
          rule.upperValue = upperValueInput.value;
          renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
        });
        upperValueTd.appendChild(upperValueInput);

        const colorTd = document.createElement("td");
        const color = document.createElement("input");
        color.type = "color";
        color.value = rule.color || "#1f77b4";
        color.addEventListener("change", () => {
          rule.color = color.value;
          renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
        });
        colorTd.appendChild(color);

        const countTd = document.createElement("td");
        countTd.textContent = String(counts.get(rule.id) || 0);

        const actionTd = document.createElement("td");
        const del = document.createElement("button");
        del.type = "button";
        del.textContent = "Delete";
        del.className = "mini-btn";
        del.addEventListener("click", () => {
          classifierClassRules = classifierClassRules.filter((item) => item.id !== rule.id);
          renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
        });
        actionTd.appendChild(del);

        tr.appendChild(enabledTd);
        tr.appendChild(labelTd);
        tr.appendChild(lowerOpTd);
        tr.appendChild(lowerValueTd);
        tr.appendChild(upperOpTd);
        tr.appendChild(upperValueTd);
        tr.appendChild(colorTd);
        tr.appendChild(countTd);
        tr.appendChild(actionTd);
        classifierClassRuleBody.appendChild(tr);
      }
      const unlabeled = counts.get("__unlabeled__") || 0;
      classifierClassRuleStatusEl.textContent = `${classifierClassRules.length} classification class rules. Unmatched visible rows: ${unlabeled}.`;
    }

    function renderPcaColorRuleTable(groups = []) {
      const counts = new Map((groups || []).map((group) => [group.key, group.count]));
      pcaColorRuleHead.innerHTML = ["enabled", "label", "lower", "lower value", "upper", "upper value", "color", "matched", ""].map((c) => `<th>${c}</th>`).join("");
      if (!pcaColorRules.length) {
        pcaColorRuleBody.innerHTML = `<tr><td colspan="9">No PCA color rules yet. Add a class rule to color PCA points; unmatched points stay gray.</td></tr>`;
        pcaColorRuleStatusEl.textContent = "No PCA color rules defined. Points without a matching rule are shown as Unlabeled.";
        return;
      }
      pcaColorRuleBody.innerHTML = "";
      for (const rule of pcaColorRules) {
        const tr = document.createElement("tr");
        const rerender = () => {
          if (lastPcaRender) renderPcaPanel();
          else renderPcaColorRuleTable();
        };

        const enabledTd = document.createElement("td");
        const enabled = document.createElement("input");
        enabled.type = "checkbox";
        enabled.checked = rule.enabled !== false;
        enabled.addEventListener("change", () => {
          rule.enabled = enabled.checked;
          rerender();
        });
        enabledTd.appendChild(enabled);

        const labelTd = document.createElement("td");
        const labelInput = document.createElement("input");
        labelInput.type = "text";
        labelInput.value = rule.label || "";
        labelInput.style.width = "120px";
        labelInput.addEventListener("change", () => {
          rule.label = labelInput.value.trim() || "Class";
          rerender();
        });
        labelTd.appendChild(labelInput);

        const lowerOpTd = document.createElement("td");
        const lowerOp = document.createElement("select");
        ["", ">", ">="].forEach((value) => {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = value || "none";
          lowerOp.appendChild(opt);
        });
        lowerOp.value = rule.lowerOperator || "";
        lowerOp.addEventListener("change", () => {
          rule.lowerOperator = lowerOp.value;
          rerender();
        });
        lowerOpTd.appendChild(lowerOp);

        const lowerValueTd = document.createElement("td");
        const lowerValueInput = document.createElement("input");
        lowerValueInput.type = "number";
        lowerValueInput.step = "any";
        lowerValueInput.value = rule.lowerValue ?? "";
        lowerValueInput.style.width = "90px";
        lowerValueInput.placeholder = "min";
        lowerValueInput.addEventListener("change", () => {
          rule.lowerValue = lowerValueInput.value;
          rerender();
        });
        lowerValueTd.appendChild(lowerValueInput);

        const upperOpTd = document.createElement("td");
        const upperOp = document.createElement("select");
        ["", "<", "<="].forEach((value) => {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = value || "none";
          upperOp.appendChild(opt);
        });
        upperOp.value = rule.upperOperator || "";
        upperOp.addEventListener("change", () => {
          rule.upperOperator = upperOp.value;
          rerender();
        });
        upperOpTd.appendChild(upperOp);

        const upperValueTd = document.createElement("td");
        const upperValueInput = document.createElement("input");
        upperValueInput.type = "number";
        upperValueInput.step = "any";
        upperValueInput.value = rule.upperValue ?? "";
        upperValueInput.style.width = "90px";
        upperValueInput.placeholder = "max";
        upperValueInput.addEventListener("change", () => {
          rule.upperValue = upperValueInput.value;
          rerender();
        });
        upperValueTd.appendChild(upperValueInput);

        const colorTd = document.createElement("td");
        const color = document.createElement("input");
        color.type = "color";
        color.value = rule.color || "#1f77b4";
        color.addEventListener("change", () => {
          rule.color = color.value;
          rerender();
        });
        colorTd.appendChild(color);

        const countTd = document.createElement("td");
        countTd.textContent = String(counts.get(rule.id) || 0);

        const actionTd = document.createElement("td");
        const del = document.createElement("button");
        del.type = "button";
        del.textContent = "Delete";
        del.className = "mini-btn";
        del.addEventListener("click", () => {
          pcaColorRules = pcaColorRules.filter((item) => item.id !== rule.id);
          rerender();
        });
        actionTd.appendChild(del);

        tr.appendChild(enabledTd);
        tr.appendChild(labelTd);
        tr.appendChild(lowerOpTd);
        tr.appendChild(lowerValueTd);
        tr.appendChild(upperOpTd);
        tr.appendChild(upperValueTd);
        tr.appendChild(colorTd);
        tr.appendChild(countTd);
        tr.appendChild(actionTd);
        pcaColorRuleBody.appendChild(tr);
      }
      const unlabeled = counts.get("__unlabeled__") || 0;
      pcaColorRuleStatusEl.textContent = `${pcaColorRules.length} PCA color rules. Unmatched PCA points: ${unlabeled}.`;
    }

    function pcaClusterPalette(k) {
      const base = ["#0f766e", "#b45309", "#be123c", "#2563eb", "#7c3aed", "#15803d", "#c2410c", "#0891b2", "#a21caf", "#4d7c0f", "#334155", "#db2777"];
      return Array.from({ length: k }, (_, idx) => base[idx % base.length]);
    }

    function kMeans2d(points, k, maxIter = 80) {
      const n = points.length;
      if (n < 2) return null;
      k = Math.max(2, Math.min(k, n));
      const sorted = points
        .map((point, idx) => ({ point, idx }))
        .sort((a, b) => a.point[0] - b.point[0] || a.point[1] - b.point[1]);
      let centers = Array.from({ length: k }, (_, clusterIdx) => {
        const pos = Math.round((clusterIdx * (sorted.length - 1)) / Math.max(1, k - 1));
        return sorted[pos].point.slice();
      });
      let assignments = Array(n).fill(0);
      for (let iter = 0; iter < maxIter; iter++) {
        let changed = false;
        for (let i = 0; i < n; i++) {
          let best = 0;
          let bestDist = Infinity;
          for (let c = 0; c < k; c++) {
            const dx = points[i][0] - centers[c][0];
            const dy = points[i][1] - centers[c][1];
            const dist = dx * dx + dy * dy;
            if (dist < bestDist) {
              best = c;
              bestDist = dist;
            }
          }
          if (assignments[i] !== best) {
            assignments[i] = best;
            changed = true;
          }
        }
        const sums = Array.from({ length: k }, () => [0, 0, 0]);
        for (let i = 0; i < n; i++) {
          const c = assignments[i];
          sums[c][0] += points[i][0];
          sums[c][1] += points[i][1];
          sums[c][2] += 1;
        }
        for (let c = 0; c < k; c++) {
          if (sums[c][2] > 0) centers[c] = [sums[c][0] / sums[c][2], sums[c][1] / sums[c][2]];
        }
        if (!changed) break;
      }
      return { assignments, centers };
    }

    function buildPcaAutoClusterGroups(scores, requestedK) {
      const points = scores.map((row) => [row[0], row[1]]);
      const result = kMeans2d(points, requestedK);
      if (!result) return null;
      const k = Math.max(...result.assignments) + 1;
      const colors = pcaClusterPalette(k);
      const counts = Array(k).fill(0);
      result.assignments.forEach((clusterIdx) => { counts[clusterIdx] += 1; });
      const groups = Array.from({ length: k }, (_, idx) => ({
        key: `cluster_${idx + 1}`,
        label: `Cluster ${idx + 1}`,
        color: colors[idx],
        count: counts[idx],
      }));
      const assignments = result.assignments.map((clusterIdx) => ({
        key: `cluster_${clusterIdx + 1}`,
        rawValue: clusterIdx + 1,
      }));
      return { assignments, groups, centers: result.centers, k };
    }

    function renderTsnePlotFromEmbedding(renderState) {
      const { Y, rowIndices, dataRowsLength, selectedCount, perplexity, selectedFeatures, dataRows } = renderState;
      const colorFeature = tsneColorFeature.value || "category";
      const colorRows = rowIndices.map((idx) => plottedRows[idx]);
      const colorInfo = buildTsneColorGroups(colorRows, colorFeature);
      renderTsneColorRuleTable(colorInfo.groups);
      const groupLookup = new Map(colorInfo.groups.map((group) => [group.key, group]));
      const byGroup = {};
      for (let i = 0; i < Y.length; i++) {
        const rowIdx = rowIndices[i];
        const row = plottedRows[rowIdx];
        const assignment = colorInfo.assignments[i] || { key: "missing", rawValue: "" };
        const group = groupLookup.get(assignment.key) || { label: assignment.key };
        const label = group.label;
        if (!byGroup[label]) byGroup[label] = [];
        byGroup[label].push({ x: Y[i][0], y: Y[i][1], idx: rowIdx, row, rawValue: assignment.rawValue, color: group.color });
      }
      lastTsneExportRows = Y.map((coords, i) => {
        const rowIdx = rowIndices[i];
        const row = plottedRows[rowIdx] || {};
        const assignment = colorInfo.assignments[i] || { key: "missing", rawValue: "" };
        const group = groupLookup.get(assignment.key) || { label: assignment.key, color: "" };
        const matchedRule = tsneColorRules.find((rule) => rule.id === assignment.key);
        const out = {
          visual_component: "t_sne",
          row_index: rowIdx,
          case_path: row.case_path || "",
          instance_id: row.instance_id || "",
          pair_name: row.pair_name || "",
          category: row.category || "",
          panel: row.panel || "",
          subject_id: row.subject_id || "",
          tsne_1: coords[0],
          tsne_2: coords[1],
          color_feature: colorFeature,
          color_value: assignment.rawValue,
          color_label: group.label || assignment.key,
          color_key: assignment.key,
          color_rule: matchedRule ? tsneColorRuleDescription(matchedRule) : "",
          color_hex: group.color || "",
          perplexity,
          selected_feature_count: selectedCount,
          selected_features: (selectedFeatures || []).join("; "),
        };
        (selectedFeatures || []).forEach((feature, featureIdx) => {
          out[`input.${feature}`] = dataRows?.[i]?.[featureIdx] ?? null;
        });
        return out;
      });

      const traces = Object.entries(byGroup).map(([label, pts]) => ({
        type: "scatter",
        mode: "markers",
        name: label,
        x: pts.map((p) => p.x),
        y: pts.map((p) => p.y),
        customdata: pts.map((p) => p.idx),
        text: pts.map((p) => `${p.row.pair_name || ""}<br>${p.row.case_path || ""}<br>Color value: ${p.rawValue}`),
        hovertemplate: "%{text}<br>tSNE1=%{x:.3f}<br>tSNE2=%{y:.3f}<extra></extra>",
        marker: { size: 9, opacity: 0.85, color: pts[0]?.color || "#9ca3af" }
      }));

      safeNewPlot("tsnePlot", traces, {
        title: `t-SNE (rows=${dataRowsLength}, features=${selectedCount}, perplexity=${perplexity}, color=${defaultTsneColorFeatureLabel(colorFeature)})`,
        xaxis: { title: "tSNE-1" },
        yaxis: { title: "tSNE-2" },
        margin: { t: 48, r: 20, b: 56, l: 64 }
      }, { responsive: true });

      const tsnePlotEl = document.getElementById("tsnePlot");
      if (tsnePlotEl && typeof tsnePlotEl.on === "function") {
        tsnePlotEl.on("plotly_click", (eventData) => {
          const point = eventData?.points?.[0];
          if (!point || point.customdata === undefined) return;
          highlightRow(Number(point.customdata));
        });
      }
    }

    function dotVector(a, b) {
      let out = 0;
      for (let i = 0; i < a.length; i++) out += a[i] * b[i];
      return out;
    }

    function buildClusterSeparationDataset() {
      if (!lastTsneRender) {
        return { error: "Run t-SNE first so there is an embedding and matching labeled rows." };
      }
      const candidateFeatures = Array.from(tsneSelectedFeatures).filter((feature) => numericFeatures.includes(feature));
      if (candidateFeatures.length < 1) {
        return { error: "Select at least one t-SNE input feature to use as SVM predictors." };
      }
      const colorFeature = tsneColorFeature.value || "";
      if (!colorFeature) {
        return { error: "Select a numeric color feature and define color rules first." };
      }
      const colorRows = lastTsneRender.rowIndices.map((idx) => plottedRows[idx]);
      const colorInfo = buildTsneColorGroups(colorRows, colorFeature);
      const groupLookup = new Map(colorInfo.groups.map((group) => [group.key, group]));
      const labeled = [];
      for (let i = 0; i < colorRows.length; i++) {
        const assignment = colorInfo.assignments[i];
        if (!assignment || assignment.key === "__unlabeled__") continue;
        const group = groupLookup.get(assignment.key);
        if (!group) continue;
        labeled.push({
          row: colorRows[i],
          label: group.label,
          color: group.color || "#1f77b4",
        });
      }
      const labels = Array.from(new Set(labeled.map((item) => item.label)));
      if (labels.length < 2) {
        return { error: "Need at least two user-labeled color groups with matched points. Add rules for two or more clusters." };
      }
      if (labeled.length < labels.length + 3) {
        return { error: "Not enough labeled points to train a separator." };
      }

      const complete = buildCompleteCaseNumericMatrix(labeled.map((item) => item.row), candidateFeatures, {
        standardize: true,
      });
      if (complete.features.length < 1 || complete.matrix.length < labels.length + 3) {
        return { error: "Not enough complete observed rows for the selected cluster-separation features. Narrow the feature selection or include more rows." };
      }
      const x = complete.matrix;
      const y = complete.rows.map((row) => {
        const item = labeled.find((entry) => entry.row === row);
        return item ? item.label : "";
      });
      const completeLabels = Array.from(new Set(y));
      if (completeLabels.length < 2) {
        return { error: "Complete-case cluster-separation data contain fewer than two labeled groups. Narrow selected predictors or include more rows." };
      }
      return {
        x,
        y,
        labels: completeLabels,
        features: complete.features,
        colors: Object.fromEntries(labeled.map((item) => [item.label, item.color])),
        labeledCount: complete.rows.length,
        completeCaseMeta: complete,
      };
    }

    function trainLinearSvmOneVsRest(x, y, labels) {
      const n = x.length;
      const p = x[0]?.length || 0;
      const models = [];
      const epochs = 260;
      const lambda = 0.001;
      const baseLr = 0.04;
      for (const label of labels) {
        const w = Array(p).fill(0);
        let b = 0;
        for (let epoch = 0; epoch < epochs; epoch++) {
          const lr = baseLr / (1 + epoch * 0.01);
          for (let i = 0; i < n; i++) {
            const yi = y[i] === label ? 1 : -1;
            const margin = yi * (dotVector(w, x[i]) + b);
            for (let j = 0; j < p; j++) w[j] *= (1 - lr * lambda);
            if (margin < 1) {
              for (let j = 0; j < p; j++) w[j] += lr * yi * x[i][j];
              b += lr * yi;
            }
          }
        }
        models.push({ label, w, b });
      }
      return models;
    }

    function predictSvmLabel(models, row) {
      let best = null;
      for (const model of models) {
        const score = dotVector(model.w, row) + model.b;
        if (!best || score > best.score) best = { label: model.label, score };
      }
      return best ? best.label : "";
    }

    function svmAccuracy(models, x, y) {
      const predictions = x.map((row) => predictSvmLabel(models, row));
      const correct = predictions.reduce((sum, pred, i) => sum + (pred === y[i] ? 1 : 0), 0);
      return {
        predictions,
        correct,
        accuracy: correct / Math.max(1, y.length),
      };
    }

    function buildPrincipalComponentDataset(dataset) {
      const maxComponents = Math.min(8, dataset.features.length, Math.max(1, dataset.x.length - 1));
      if (dataset.features.length < 2 || maxComponents < 1) return null;
      const cov = covarianceMatrixFromStandardized(dataset.x);
      const components = topEigenComponentsSymmetric(cov, maxComponents);
      if (!components.length) return null;
      const xPc = dataset.x.map((row) => {
        return components.map((component) => dot(row, component.loadings));
      });
      const pcFeatures = components.map((_, idx) => `PC${idx + 1}`);
      const totalVariance = dataset.features.length;
      const pcMeta = components.map((component, idx) => {
        const loadings = component.loadings
          .map((loading, featureIdx) => ({ feature: dataset.features[featureIdx], loading }))
          .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading));
        return {
          component: pcFeatures[idx],
          eigenvalue: component.eigenvalue,
          variancePct: totalVariance > 0 ? (100 * component.eigenvalue) / totalVariance : null,
          topLoadings: loadings.slice(0, 6),
        };
      });
      return { x: xPc, features: pcFeatures, pcMeta };
    }

    function renderPcaPanel() {
      const selected = Array.from(pcaSelectedFeatures).filter((feature) => numericFeatures.includes(feature));
      lastPcaRender = null;
      if (selected.length < 2) {
        pcaStatusEl.textContent = "Select at least two PCA features.";
        pcaHead.innerHTML = "";
        pcaBody.innerHTML = "";
        safeNewPlot("pcaPlot", [], { title: "PCA: not enough selected features" }, { responsive: true });
        return;
      }
      const visibleRows = plottedRows.filter((_, idx) => rowVisibility[idx]);
      const visibleSourceIndices = plottedRows.map((_, idx) => idx).filter((idx) => rowVisibility[idx]);
      if (visibleRows.length < 3) {
        pcaStatusEl.textContent = "PCA needs at least three visible rows.";
        pcaHead.innerHTML = "";
        pcaBody.innerHTML = "";
        safeNewPlot("pcaPlot", [], { title: "PCA: not enough rows" }, { responsive: true });
        return;
      }
      const complete = buildCompleteCaseNumericMatrix(visibleRows, selected, {
        standardize: true,
        sourceIndices: visibleSourceIndices,
      });
      if (complete.matrix.length < 3 || complete.features.length < 2) {
        pcaStatusEl.textContent = "Not enough complete observed rows/features for PCA. Narrow feature selection or include more rows.";
        pcaHead.innerHTML = "";
        pcaBody.innerHTML = "";
        safeNewPlot("pcaPlot", [], { title: "PCA: insufficient usable data" }, { responsive: true });
        return;
      }
      const cov = covarianceMatrixFromStandardized(complete.matrix);
      const components = topEigenComponentsSymmetric(cov, Math.min(6, complete.features.length, complete.matrix.length - 1));
      if (components.length < 2) {
        pcaStatusEl.textContent = "PCA could not find two usable components from the selected features.";
        pcaHead.innerHTML = "";
        pcaBody.innerHTML = "";
        safeNewPlot("pcaPlot", [], { title: "PCA: fewer than two components" }, { responsive: true });
        return;
      }
      const scores = complete.matrix.map((row) => components.map((component) => dot(row, component.loadings)));
      const totalVariance = complete.features.length;
      const variancePct = components.map((component) => totalVariance > 0 ? (100 * component.eigenvalue) / totalVariance : null);
      const colorFeature = pcaColorFeature.value || "";
      const colorRows = complete.rowIndices.map((idx) => plottedRows[idx]);
      let colorInfo = null;
      let colorModeLabel = "manual class rules";
      if (pcaAutoClusterMode) {
        const k = Math.max(2, Math.min(Number(pcaClusterCountInput.value || 3), complete.matrix.length));
        colorInfo = buildPcaAutoClusterGroups(scores, k);
        pcaAutoClusterInfo = colorInfo;
        colorModeLabel = `auto K-means clusters (K=${colorInfo?.k || k})`;
        pcaColorRuleStatusEl.textContent = colorInfo
          ? `Auto PCA clustering active: ${colorInfo.groups.map((group) => `${group.label} n=${group.count}`).join("; ")}.`
          : "Auto PCA clustering could not be computed.";
      } else {
        colorInfo = buildPcaColorGroups(colorRows, colorFeature);
        pcaAutoClusterInfo = null;
        renderPcaColorRuleTable(colorInfo.groups);
      }
      const groupLookup = new Map(colorInfo.groups.map((group) => [group.key, group]));
      const text = complete.rowIndices.map((rowIdx, i) => {
        const row = plottedRows[rowIdx] || {};
        const assignment = colorInfo.assignments[i] || { key: "__unlabeled__", rawValue: "missing" };
        const group = groupLookup.get(assignment.key) || { label: "Unlabeled" };
        return [
          `row=${rowIdx + 1}`,
          `subject=${row.subject_id || "NA"}`,
          `case=${row.case_path || "NA"}`,
          `pair=${row.pair_name || row.instance_id || "NA"}`,
          `PC1=${scores[i][0].toFixed(4)}`,
          `PC2=${scores[i][1].toFixed(4)}`,
          `class=${group.label}`,
          colorFeature ? `${colorFeature}=${assignment.rawValue === null ? "NA" : assignment.rawValue}` : "",
        ].filter(Boolean).join("<br>");
      });
      const traces = colorInfo.groups.map((group) => {
        const indices = colorInfo.assignments
          .map((assignment, idx) => assignment.key === group.key ? idx : -1)
          .filter((idx) => idx >= 0);
        return {
          type: "scatter",
          mode: "markers",
          name: `${group.label} (${indices.length})`,
          x: indices.map((idx) => scores[idx][0]),
          y: indices.map((idx) => scores[idx][1]),
          text: indices.map((idx) => text[idx]),
          marker: {
            size: 9,
            color: group.color || "#9ca3af",
            line: { color: "rgba(15,23,42,0.35)", width: 0.7 },
          },
          hovertemplate: "%{text}<extra></extra>",
        };
      });
      safeNewPlot("pcaPlot", traces, {
        title: `PCA projection (${complete.features.length} features, ${complete.matrix.length} rows)`,
        xaxis: { title: `PC1 (${variancePct[0] === null ? "-" : variancePct[0].toFixed(1)}% variance)` },
        yaxis: { title: `PC2 (${variancePct[1] === null ? "-" : variancePct[1].toFixed(1)}% variance)` },
        margin: { t: 52, r: 24, b: 64, l: 72 },
        legend: { orientation: "v", x: 1.02, y: 1 },
      }, { responsive: true });

      const loadingRows = [];
      components.slice(0, 4).forEach((component, componentIdx) => {
        component.loadings
          .map((loading, featureIdx) => ({ feature: complete.features[featureIdx], loading }))
          .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading))
          .slice(0, 10)
          .forEach((item, rank) => {
            loadingRows.push({
              component: `PC${componentIdx + 1}`,
              variancePct: variancePct[componentIdx],
              rank: rank + 1,
              feature: item.feature,
              loading: item.loading,
            });
          });
      });
      pcaHead.innerHTML = ["component", "variance %", "rank", "feature", "loading"].map((col) => `<th>${col}</th>`).join("");
      pcaBody.innerHTML = loadingRows.map((row) => `
        <tr>
          <td>${htmlEscape(row.component)}</td>
          <td>${row.variancePct === null ? "-" : row.variancePct.toFixed(2)}</td>
          <td>${row.rank}</td>
          <td>${htmlEscape(row.feature)}</td>
          <td>${row.loading.toFixed(6)}</td>
        </tr>
      `).join("");
      lastPcaRender = { complete, scores, components, variancePct, colorInfo, colorModeLabel };
      pcaStatusEl.textContent = `PCA computed on ${complete.matrix.length}/${visibleRows.length} visible rows using ${complete.features.length} usable features from ${selected.length} selected features. Color mode: ${colorModeLabel}${pcaAutoClusterMode ? "" : ` from ${colorFeature || "none"}`}. ${summarizeCompleteCase(complete)}`;
    }

    function buildPcaLabeledRowsForSupervision() {
      if (pcaAutoClusterMode) {
        if (!lastPcaRender || !lastPcaRender.colorInfo) {
          return { error: "Run PCA with auto clustering first, then rank class-separating features." };
        }
        const groupLookup = new Map(lastPcaRender.colorInfo.groups.map((group) => [group.key, group]));
        const labeled = [];
        for (let i = 0; i < lastPcaRender.complete.rowIndices.length; i++) {
          const rowIdx = lastPcaRender.complete.rowIndices[i];
          const assignment = lastPcaRender.colorInfo.assignments[i];
          const group = groupLookup.get(assignment?.key);
          if (!group) continue;
          labeled.push({
            row: plottedRows[rowIdx],
            label: group.label,
            color: group.color || "#1f77b4",
            rawValue: assignment.rawValue,
          });
        }
        const labels = Array.from(new Set(labeled.map((item) => item.label)));
        if (labels.length < 2) return { error: "Auto PCA clustering produced fewer than two clusters." };
        return { labeled, labels, colorFeature: `auto_clusters_k_${labels.length}`, colorInfo: lastPcaRender.colorInfo };
      }
      const visibleRows = plottedRows.filter((_, idx) => rowVisibility[idx]);
      if (visibleRows.length < 3) return { error: "Need at least three visible rows for supervised PCA feature ranking." };
      const colorFeature = pcaColorFeature.value || "";
      if (!colorFeature) return { error: "Select a PCA color feature and define color rules first." };
      if (!pcaColorRules.some((rule) => normalizedTsneColorRule(rule))) {
        return { error: "Define at least two active PCA color class rules before ranking features." };
      }
      const colorInfo = buildPcaColorGroups(visibleRows, colorFeature);
      renderPcaColorRuleTable(colorInfo.groups);
      const groupLookup = new Map(colorInfo.groups.map((group) => [group.key, group]));
      const labeled = [];
      for (let i = 0; i < visibleRows.length; i++) {
        const assignment = colorInfo.assignments[i];
        if (!assignment || assignment.key === "__unlabeled__") continue;
        const group = groupLookup.get(assignment.key);
        if (!group) continue;
        labeled.push({
          row: visibleRows[i],
          label: group.label,
          color: group.color || "#1f77b4",
          rawValue: assignment.rawValue,
        });
      }
      const labels = Array.from(new Set(labeled.map((item) => item.label)));
      if (labels.length < 2) return { error: "Need at least two PCA classes with matched rows." };
      if (labeled.length < labels.length + 2) return { error: "Not enough labeled rows to score class separation." };
      return { labeled, labels, colorFeature, colorInfo };
    }

    function supervisedFeatureSeparationRows() {
      const labeledInfo = buildPcaLabeledRowsForSupervision();
      if (labeledInfo.error) return labeledInfo;
      const candidateFeatures = Array.from(pcaSelectedFeatures).filter((feature) => numericFeatures.includes(feature));
      if (candidateFeatures.length < 1) return { error: "Select at least one PCA feature to score." };
      const rows = [];
      for (const feature of candidateFeatures) {
        const byClass = new Map();
        for (const item of labeledInfo.labeled) {
          const value = parseMaybeNumber(item.row[feature]);
          if (value === null) continue;
          if (!byClass.has(item.label)) byClass.set(item.label, []);
          byClass.get(item.label).push(value);
        }
        const classEntries = Array.from(byClass.entries()).filter(([, vals]) => vals.length > 0);
        if (classEntries.length < 2) continue;
        const allVals = classEntries.flatMap(([, vals]) => vals);
        const n = allVals.length;
        const k = classEntries.length;
        if (n <= k) continue;
        const overall = mean(allVals);
        let betweenSS = 0;
        let withinSS = 0;
        const classMeans = {};
        const classCounts = {};
        for (const [label, vals] of classEntries) {
          const m = mean(vals);
          classMeans[label] = m;
          classCounts[label] = vals.length;
          betweenSS += vals.length * Math.pow(m - overall, 2);
          withinSS += vals.reduce((sum, value) => sum + Math.pow(value - m, 2), 0);
        }
        const dfBetween = k - 1;
        const dfWithin = n - k;
        const msBetween = betweenSS / Math.max(1, dfBetween);
        const msWithin = withinSS / Math.max(1, dfWithin);
        const fScore = msWithin > 1e-12 ? msBetween / msWithin : (betweenSS > 0 ? Infinity : 0);
        const separationRatio = withinSS > 1e-12 ? betweenSS / withinSS : (betweenSS > 0 ? Infinity : 0);
        const etaSquared = (betweenSS + withinSS) > 1e-12 ? betweenSS / (betweenSS + withinSS) : null;
        const orderedMeans = Object.entries(classMeans).sort((a, b) => b[1] - a[1]);
        rows.push({
          feature,
          fScore,
          separationRatio,
          etaSquared,
          betweenSS,
          withinSS,
          labeledRows: n,
          classCount: k,
          classMeans,
          classCounts,
          strongestHighClass: orderedMeans[0]?.[0] || "",
          strongestLowClass: orderedMeans[orderedMeans.length - 1]?.[0] || "",
        });
      }
      rows.sort((a, b) => {
        const af = Number.isFinite(a.fScore) ? a.fScore : Number.MAX_VALUE;
        const bf = Number.isFinite(b.fScore) ? b.fScore : Number.MAX_VALUE;
        return bf - af || (b.etaSquared || 0) - (a.etaSquared || 0);
      });
      if (!rows.length) return { error: "No selected PCA features had enough observed values across at least two classes." };
      return { rows, ...labeledInfo };
    }

    function pcaSubsetCompleteCaseCount(features, rows = null) {
      const sourceRows = rows || plottedRows.filter((_, idx) => rowVisibility[idx]);
      if (!features.length) return 0;
      let count = 0;
      for (const row of sourceRows) {
        let complete = true;
        for (const feature of features) {
          if (parseMaybeNumber(row?.[feature]) === null) {
            complete = false;
            break;
          }
        }
        if (complete) count += 1;
      }
      return count;
    }

    function choosePcaRankedFeatureSubset(rankedRows, minCompleteCases, maxFeatures) {
      const ordered = rankedRows.map((row) => row.feature).filter((feature) => numericFeatures.includes(feature));
      const sourceRows = plottedRows.filter((_, idx) => rowVisibility[idx]);
      let best = [];
      let bestCount = 0;
      for (let n = 1; n <= Math.min(maxFeatures, ordered.length); n++) {
        const trial = ordered.slice(0, n);
        const count = pcaSubsetCompleteCaseCount(trial, sourceRows);
        if (count >= minCompleteCases) {
          best = trial;
          bestCount = count;
        } else if (best.length) {
          break;
        }
      }
      if (best.length) return { features: best, completeCases: bestCount, metConstraint: true };
      for (let n = 1; n <= Math.min(maxFeatures, ordered.length); n++) {
        const trial = ordered.slice(0, n);
        const count = pcaSubsetCompleteCaseCount(trial, sourceRows);
        if (count > bestCount || (count === bestCount && trial.length > best.length)) {
          best = trial;
          bestCount = count;
        }
      }
      return { features: best, completeCases: bestCount, metConstraint: false };
    }

    function renderPcaSupervisedFeatureRanking() {
      const result = supervisedFeatureSeparationRows();
      if (result.error) {
        lastPcaSupervisedRows = [];
        pcaSupervisedStatusEl.textContent = result.error;
        pcaSupervisedHead.innerHTML = "";
        pcaSupervisedBody.innerHTML = "";
        safeNewPlot("pcaSupervisedPlot", [], { title: "No supervised PCA feature ranking" }, { responsive: true });
        return;
      }
      lastPcaSupervisedRows = result.rows;
      const topN = Math.max(2, Math.min(100, Number(pcaTopNFeaturesInput.value || 10)));
      const minCompleteCases = Math.max(3, Number(pcaMinCompleteCasesInput.value || 10));
      const suggested = choosePcaRankedFeatureSubset(result.rows, minCompleteCases, topN);
      const suggestedSet = new Set(suggested.features);
      pcaSupervisedHead.innerHTML = ["rank", "apply", "feature", "F score", "η²", "between SS", "within SS", "feature rows/classes", "class means"].map((col) => `<th>${col}</th>`).join("");
      pcaSupervisedBody.innerHTML = result.rows.map((row, idx) => `
        <tr>
          <td>${idx + 1}</td>
          <td><input type="checkbox" data-pca-ranked-feature="${htmlEscape(row.feature)}" ${suggestedSet.has(row.feature) ? "checked" : ""}></td>
          <td>${htmlEscape(row.feature)}</td>
          <td>${Number.isFinite(row.fScore) ? row.fScore.toFixed(4) : "Infinity"}</td>
          <td>${row.etaSquared === null ? "-" : row.etaSquared.toFixed(4)}</td>
          <td>${row.betweenSS.toFixed(4)}</td>
          <td>${row.withinSS.toFixed(4)}</td>
          <td>${row.labeledRows}/${row.classCount}</td>
          <td>${htmlEscape(Object.entries(row.classMeans).map(([label, value]) => `${label}: ${value.toFixed(4)} (n=${row.classCounts[label] || 0})`).join("; "))}</td>
        </tr>
      `).join("");
      const top = result.rows.slice(0, 20).reverse();
      safeNewPlot("pcaSupervisedPlot", [{
        type: "bar",
        orientation: "h",
        y: top.map((row) => featureShortLabel(row.feature, featureCategory(row.feature))),
        x: top.map((row) => Number.isFinite(row.fScore) ? row.fScore : 0),
        text: top.map((row) => `${row.feature}<br>F=${Number.isFinite(row.fScore) ? row.fScore.toFixed(4) : "Infinity"}<br>η²=${row.etaSquared === null ? "-" : row.etaSquared.toFixed(4)}<br>high=${row.strongestHighClass}<br>low=${row.strongestLowClass}`),
        hovertemplate: "%{text}<extra></extra>",
        marker: { color: "rgba(15,118,110,0.75)" },
      }], {
        title: "Top class-separating features for PCA color classes",
        xaxis: { title: "ANOVA-like F score" },
        yaxis: { automargin: true },
        margin: { t: 48, r: 20, b: 56, l: 220 },
      }, { responsive: true });
      pcaSupervisedStatusEl.textContent = `Ranked ${result.rows.length} selected PCA features using ${result.labeled.length} labeled visible rows across ${result.labels.length} classes from ${result.colorFeature}. Suggested ${suggested.features.length} features retain ${suggested.completeCases} complete cases${suggested.metConstraint ? "" : `, below requested minimum ${minCompleteCases}`}.`;
    }

    function applyPcaSupervisedTopFeatures() {
      if (!lastPcaSupervisedRows.length) {
        renderPcaSupervisedFeatureRanking();
      }
      if (!lastPcaSupervisedRows.length) return;
      const checked = Array.from(pcaSupervisedBody.querySelectorAll("input[data-pca-ranked-feature]:checked"))
        .map((el) => el.getAttribute("data-pca-ranked-feature"))
        .filter(Boolean);
      const topN = Math.max(2, Math.min(100, Number(pcaTopNFeaturesInput.value || 10)));
      const minCompleteCases = Math.max(3, Number(pcaMinCompleteCasesInput.value || 10));
      const suggested = choosePcaRankedFeatureSubset(lastPcaSupervisedRows, minCompleteCases, topN);
      const features = checked.length ? checked : suggested.features;
      const completeCases = pcaSubsetCompleteCaseCount(features);
      pcaSelectedFeatures = new Set(features);
      renderPcaFeatureSelector(numericFeatures);
      pcaStatusEl.textContent = `Applied ${features.length} supervised class-separating features to PCA selection, retaining ${completeCases} complete visible cases${completeCases < minCompleteCases ? ` (below requested minimum ${minCompleteCases})` : ""}. Click Run PCA to recompute.`;
    }

    function renderClusterSeparationAnalysis() {
      const dataset = buildClusterSeparationDataset();
      if (dataset.error) {
        tsneSvmStatusEl.textContent = dataset.error;
        tsneSvmHead.innerHTML = "";
        tsneSvmBody.innerHTML = "";
        safeNewPlot("tsneSvmPlot", [], { title: "No cluster separation model" }, { responsive: true });
        return;
      }
      const models = trainLinearSvmOneVsRest(dataset.x, dataset.y, dataset.labels);
      const originalFit = svmAccuracy(models, dataset.x, dataset.y);
      const pcDataset = buildPrincipalComponentDataset(dataset);
      const pcModels = pcDataset ? trainLinearSvmOneVsRest(pcDataset.x, dataset.y, dataset.labels) : [];
      const pcFit = pcDataset ? svmAccuracy(pcModels, pcDataset.x, dataset.y) : null;

      const featureImportance = dataset.features.map((feature, idx) => {
        const perClass = models.map((model) => ({ label: model.label, weight: model.w[idx] }));
        const maxAbsWeight = Math.max(...perClass.map((item) => Math.abs(item.weight)));
        return { feature, maxAbsWeight, perClass };
      }).sort((a, b) => b.maxAbsWeight - a.maxAbsWeight);
      const pcImportance = pcDataset ? pcDataset.features.map((feature, idx) => {
        const perClass = pcModels.map((model) => ({ label: model.label, weight: model.w[idx] }));
        const maxAbsWeight = Math.max(...perClass.map((item) => Math.abs(item.weight)));
        const meta = pcDataset.pcMeta[idx] || {};
        return { feature, maxAbsWeight, perClass, meta };
      }).sort((a, b) => b.maxAbsWeight - a.maxAbsWeight) : [];

      tsneSvmStatusEl.textContent =
        `Linear SVM-style separator trained on ${dataset.labeledCount} labeled points, ${dataset.features.length} selected features, ${dataset.labels.length} classes. Original-feature accuracy: ${(originalFit.accuracy * 100).toFixed(1)}%. Principal-component accuracy: ${pcFit ? `${(pcFit.accuracy * 100).toFixed(1)}% using ${pcDataset.features.length} PCs` : "not available"}.`;

      const tableRows = [];
      for (const model of models) {
        const top = dataset.features.map((feature, idx) => ({
          modelSpace: "selected_features",
          classLabel: model.label,
          feature,
          weight: model.w[idx],
          absWeight: Math.abs(model.w[idx]),
          variancePct: null,
          topLoadings: "",
        })).sort((a, b) => b.absWeight - a.absWeight).slice(0, 12);
        tableRows.push(...top);
      }
      if (pcDataset) {
        for (const model of pcModels) {
          const top = pcDataset.features.map((feature, idx) => {
            const meta = pcDataset.pcMeta[idx] || {};
            return {
              modelSpace: "principal_components",
              classLabel: model.label,
              feature,
              weight: model.w[idx],
              absWeight: Math.abs(model.w[idx]),
              variancePct: meta.variancePct,
              topLoadings: (meta.topLoadings || []).map((item) => `${item.feature} (${item.loading.toFixed(3)})`).join("; "),
            };
          }).sort((a, b) => b.absWeight - a.absWeight).slice(0, 8);
          tableRows.push(...top);
        }
      }
      tsneSvmHead.innerHTML = ["model space", "cluster label", "feature/component", "weight", "abs_weight", "PC variance %", "top PC loadings"].map((c) => `<th>${c}</th>`).join("");
      tsneSvmBody.innerHTML = tableRows.map((row) => `
        <tr>
          <td>${htmlEscape(row.modelSpace)}</td>
          <td>${htmlEscape(row.classLabel)}</td>
          <td>${htmlEscape(row.feature)}</td>
          <td>${row.weight.toFixed(6)}</td>
          <td>${row.absWeight.toFixed(6)}</td>
          <td>${row.variancePct === null || row.variancePct === undefined ? "-" : row.variancePct.toFixed(2)}</td>
          <td>${htmlEscape(row.topLoadings || "")}</td>
        </tr>
      `).join("");

      const originalPlot = featureImportance.slice(0, 14).map((item) => ({ ...item, plotLabel: featureShortLabel(item.feature, featureCategory(item.feature)), color: "rgba(190,24,93,0.72)" }));
      const pcPlot = pcImportance.slice(0, 8).map((item) => ({ ...item, plotLabel: `${item.feature} (PC)`, color: "rgba(37,99,235,0.68)" }));
      const topPlot = originalPlot.concat(pcPlot).sort((a, b) => b.maxAbsWeight - a.maxAbsWeight).slice(0, 22).reverse();
      safeNewPlot("tsneSvmPlot", [{
        type: "bar",
        orientation: "h",
        y: topPlot.map((item) => item.plotLabel),
        x: topPlot.map((item) => item.maxAbsWeight),
        text: topPlot.map((item) => `${item.feature}<br>max |weight|=${item.maxAbsWeight.toFixed(4)}<br>${item.meta?.topLoadings ? `top loadings: ${item.meta.topLoadings.map((x) => `${x.feature} (${x.loading.toFixed(3)})`).join("; ")}<br>` : ""}${item.perClass.map((x) => `${x.label}: ${x.weight.toFixed(3)}`).join("<br>")}`),
        hovertemplate: "%{text}<extra></extra>",
        marker: { color: topPlot.map((item) => item.color) },
      }], {
        title: "Top selected features and PCs separating user-labeled t-SNE color groups",
        xaxis: { title: "max absolute linear separator weight" },
        yaxis: { automargin: true },
        margin: { t: 48, r: 20, b: 56, l: 220 },
      }, { responsive: true });
    }

    function seededRandom(seed) {
      let t = seed >>> 0;
      return function() {
        t += 0x6D2B79F5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
      };
    }

    function seededShuffle(values, rng) {
      const out = values.slice();
      for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
      }
      return out;
    }

    function classifierSeed() {
      const parsed = Number(classifierSeedInput?.value);
      return Number.isFinite(parsed) ? Math.trunc(parsed) : 20260606;
    }

    function labelCounts(y, indices) {
      const counts = {};
      for (const idx of indices) counts[y[idx]] = (counts[y[idx]] || 0) + 1;
      return counts;
    }

    function giniFromCounts(counts) {
      const total = Object.values(counts).reduce((a, b) => a + b, 0);
      if (!total) return 0;
      let impurity = 1;
      for (const count of Object.values(counts)) {
        const p = count / total;
        impurity -= p * p;
      }
      return impurity;
    }

    function leafProbs(y, indices, labels) {
      const counts = labelCounts(y, indices);
      const total = Math.max(1, indices.length);
      const probs = {};
      for (const label of labels) probs[label] = (counts[label] || 0) / total;
      return probs;
    }

    function trainDecisionTree(x, y, labels, indices, featureIndices, depth, options, rng) {
      const counts = labelCounts(y, indices);
      if (depth >= options.maxDepth || indices.length <= options.minLeaf * 2 || Object.keys(counts).length <= 1) {
        return { type: "leaf", probs: leafProbs(y, indices, labels), n: indices.length };
      }

      const shuffled = seededShuffle(featureIndices, rng);
      const candidates = shuffled.slice(0, Math.max(1, Math.min(options.mtry, shuffled.length)));
      let best = null;
      for (const featureIdx of candidates) {
        const values = Array.from(new Set(indices.map((idx) => x[idx][featureIdx]).filter((v) => Number.isFinite(v)))).sort((a, b) => a - b);
        if (values.length < 2) continue;
        const step = Math.max(1, Math.floor(values.length / 12));
        for (let vi = 1; vi < values.length; vi += step) {
          const threshold = (values[vi - 1] + values[vi]) / 2;
          const left = [];
          const right = [];
          for (const idx of indices) {
            if (x[idx][featureIdx] <= threshold) left.push(idx);
            else right.push(idx);
          }
          if (left.length < options.minLeaf || right.length < options.minLeaf) continue;
          const impurity = (left.length * giniFromCounts(labelCounts(y, left)) + right.length * giniFromCounts(labelCounts(y, right))) / indices.length;
          if (!best || impurity < best.impurity) best = { featureIdx, threshold, left, right, impurity };
        }
      }
      if (!best) return { type: "leaf", probs: leafProbs(y, indices, labels), n: indices.length };
      return {
        type: "node",
        featureIdx: best.featureIdx,
        threshold: best.threshold,
        left: trainDecisionTree(x, y, labels, best.left, featureIndices, depth + 1, options, rng),
        right: trainDecisionTree(x, y, labels, best.right, featureIndices, depth + 1, options, rng),
      };
    }

    function trainRandomForestClassifier(x, y, labels, options = {}) {
      const n = x.length;
      const p = x[0]?.length || 0;
      const rng = seededRandom(options.seed || 20260606);
      const featureIndices = Array.from({ length: p }, (_, i) => i);
      const treeOptions = {
        maxDepth: options.maxDepth || 4,
        minLeaf: options.minLeaf || 2,
        mtry: options.mtry || Math.max(1, Math.round(Math.sqrt(Math.max(1, p)))),
      };
      const trees = [];
      for (let t = 0; t < (options.nTrees || 80); t++) {
        const indices = Array.from({ length: n }, () => Math.floor(rng() * n));
        trees.push(trainDecisionTree(x, y, labels, indices, featureIndices, 0, treeOptions, rng));
      }
      return { trees, labels, featureCount: p, options: { ...treeOptions, nTrees: options.nTrees || 80 } };
    }

    function predictTreeProbs(tree, row) {
      let node = tree;
      while (node && node.type !== "leaf") {
        node = row[node.featureIdx] <= node.threshold ? node.left : node.right;
      }
      return node?.probs || {};
    }

    function predictForestProbs(model, row) {
      const probs = {};
      for (const label of model.labels) probs[label] = 0;
      for (const tree of model.trees) {
        const treeProbs = predictTreeProbs(tree, row);
        for (const label of model.labels) probs[label] += treeProbs[label] || 0;
      }
      for (const label of model.labels) probs[label] /= Math.max(1, model.trees.length);
      return probs;
    }

    function predictForestLabel(model, row) {
      const probs = predictForestProbs(model, row);
      let best = "";
      let bestValue = -Infinity;
      for (const label of model.labels) {
        if ((probs[label] || 0) > bestValue) {
          best = label;
          bestValue = probs[label] || 0;
        }
      }
      return best;
    }

    function trainKnnClassifier(x, y, labels, options = {}) {
      return { type: "knn", x, y, labels, k: Math.max(1, Math.min(options.k || 5, x.length)) };
    }

    function predictKnnProbs(model, row) {
      const distances = model.x.map((trainRow, idx) => {
        const d2 = trainRow.reduce((sum, value, j) => sum + Math.pow(value - row[j], 2), 0);
        return { idx, distance: Math.sqrt(d2) };
      }).sort((a, b) => a.distance - b.distance).slice(0, model.k);
      const probs = {};
      for (const label of model.labels) probs[label] = 0;
      let weightSum = 0;
      for (const item of distances) {
        const weight = 1 / (item.distance + 1e-9);
        probs[model.y[item.idx]] += weight;
        weightSum += weight;
      }
      for (const label of model.labels) probs[label] /= Math.max(1e-12, weightSum);
      return probs;
    }

    function trainGaussianNbClassifier(x, y, labels) {
      const p = x[0]?.length || 0;
      const classStats = {};
      for (const label of labels) {
        const rows = x.filter((_, idx) => y[idx] === label);
        const prior = rows.length / Math.max(1, x.length);
        const means = [];
        const variances = [];
        for (let j = 0; j < p; j++) {
          const vals = rows.map((row) => row[j]);
          const m = mean(vals);
          const variance = vals.reduce((sum, value) => sum + Math.pow(value - m, 2), 0) / Math.max(1, vals.length - 1);
          means.push(m);
          variances.push(Math.max(variance, 1e-6));
        }
        classStats[label] = { prior, means, variances };
      }
      return { type: "gaussian_nb", labels, classStats };
    }

    function predictGaussianNbProbs(model, row) {
      const logs = {};
      for (const label of model.labels) {
        const stats = model.classStats[label];
        let logp = Math.log(Math.max(stats.prior, 1e-12));
        for (let j = 0; j < row.length; j++) {
          const variance = stats.variances[j];
          const diff = row[j] - stats.means[j];
          logp += -0.5 * Math.log(2 * Math.PI * variance) - (diff * diff) / (2 * variance);
        }
        logs[label] = logp;
      }
      const maxLog = Math.max(...Object.values(logs));
      const probs = {};
      let sum = 0;
      for (const label of model.labels) {
        probs[label] = Math.exp(logs[label] - maxLog);
        sum += probs[label];
      }
      for (const label of model.labels) probs[label] /= Math.max(sum, 1e-12);
      return probs;
    }

    function trainLinearSvmClassifier(x, y, labels) {
      return { type: "linear_svm", labels, models: trainLinearSvmOneVsRest(x, y, labels) };
    }

    function predictLinearSvmProbs(model, row) {
      const scores = model.models.map((m) => ({ label: m.label, score: dotVector(m.w, row) + m.b }));
      const maxScore = Math.max(...scores.map((item) => item.score));
      const probs = {};
      let sum = 0;
      for (const item of scores) {
        probs[item.label] = Math.exp(item.score - maxScore);
        sum += probs[item.label];
      }
      for (const label of model.labels) probs[label] = (probs[label] || 0) / Math.max(sum, 1e-12);
      return probs;
    }

    function classifierModelLabel(type) {
      const labels = {
        random_forest: "Random Forest-style",
        knn: "k-nearest neighbors",
        gaussian_nb: "Gaussian Naive Bayes",
        linear_svm: "Linear SVM-style",
      };
      return labels[type] || labels.random_forest;
    }

    function trainClassifierModel(type, x, y, labels, seed) {
      if (type === "knn") return trainKnnClassifier(x, y, labels, { k: Math.min(5, Math.max(1, Math.floor(Math.sqrt(x.length)))) });
      if (type === "gaussian_nb") return trainGaussianNbClassifier(x, y, labels);
      if (type === "linear_svm") return trainLinearSvmClassifier(x, y, labels);
      const rf = trainRandomForestClassifier(x, y, labels, {
        nTrees: 90,
        maxDepth: 4,
        minLeaf: 2,
        seed,
      });
      return { type: "random_forest", labels, rf };
    }

    function predictClassifierProbs(model, row) {
      if (model.type === "knn") return predictKnnProbs(model, row);
      if (model.type === "gaussian_nb") return predictGaussianNbProbs(model, row);
      if (model.type === "linear_svm") return predictLinearSvmProbs(model, row);
      return predictForestProbs(model.rf || model, row);
    }

    function predictClassifierLabel(model, row) {
      const probs = predictClassifierProbs(model, row);
      let best = "";
      let bestValue = -Infinity;
      const labels = model.labels || Object.keys(probs);
      for (const label of labels) {
        if ((probs[label] || 0) > bestValue) {
          best = label;
          bestValue = probs[label] || 0;
        }
      }
      return best;
    }

    function evaluateClassifier(model, x, y) {
      const predictions = x.map((row) => predictClassifierLabel(model, row));
      const probabilities = x.map((row) => predictClassifierProbs(model, row));
      const correct = predictions.reduce((sum, pred, i) => sum + (pred === y[i] ? 1 : 0), 0);
      return { predictions, probabilities, correct, accuracy: correct / Math.max(1, y.length) };
    }

    function binaryAuroc(scores, positives) {
      const posCount = positives.filter(Boolean).length;
      const negCount = positives.length - posCount;
      if (posCount === 0 || negCount === 0) return null;
      const pairs = scores.map((score, idx) => ({ score, positive: positives[idx] ? 1 : 0 }))
        .sort((a, b) => a.score - b.score);
      let rankSum = 0;
      let i = 0;
      while (i < pairs.length) {
        let j = i + 1;
        while (j < pairs.length && pairs[j].score === pairs[i].score) j += 1;
        const avgRank = (i + 1 + j) / 2;
        for (let k = i; k < j; k++) {
          if (pairs[k].positive) rankSum += avgRank;
        }
        i = j;
      }
      return (rankSum - (posCount * (posCount + 1)) / 2) / (posCount * negCount);
    }

    function binaryRocCurve(scores, positives) {
      const posCount = positives.filter(Boolean).length;
      const negCount = positives.length - posCount;
      if (posCount === 0 || negCount === 0) return [];
      const thresholds = Array.from(new Set(scores.filter((score) => Number.isFinite(score)))).sort((a, b) => b - a);
      const points = [{ threshold: "above_max", fpr: 0, tpr: 0 }];
      for (const threshold of thresholds) {
        let tp = 0;
        let fp = 0;
        for (let i = 0; i < scores.length; i++) {
          if ((scores[i] || 0) >= threshold) {
            if (positives[i]) tp += 1;
            else fp += 1;
          }
        }
        points.push({
          threshold,
          fpr: fp / negCount,
          tpr: tp / posCount,
        });
      }
      const last = points[points.length - 1];
      if (!last || last.fpr < 1 || last.tpr < 1) points.push({ threshold: "below_min", fpr: 1, tpr: 1 });
      return points;
    }

    function classifierRocRows(labels, truth, probabilities) {
      const rows = [];
      for (const label of labels) {
        const scores = probabilities.map((prob) => prob[label] || 0);
        const positives = truth.map((actual) => actual === label);
        const auroc = binaryAuroc(scores, positives);
        binaryRocCurve(scores, positives).forEach((point, idx) => {
          rows.push({
            class_label: label,
            point_index: idx,
            threshold: point.threshold,
            fpr: point.fpr,
            tpr: point.tpr,
            auroc,
          });
        });
      }
      return rows;
    }

    function classificationMetricRows(labels, truth, predictions, probabilities) {
      const rows = labels.map((label) => {
        let tp = 0;
        let fp = 0;
        let tn = 0;
        let fn = 0;
        for (let i = 0; i < truth.length; i++) {
          const actualPositive = truth[i] === label;
          const predictedPositive = predictions[i] === label;
          if (actualPositive && predictedPositive) tp += 1;
          else if (!actualPositive && predictedPositive) fp += 1;
          else if (!actualPositive && !predictedPositive) tn += 1;
          else if (actualPositive && !predictedPositive) fn += 1;
        }
        const sensitivity = tp + fn > 0 ? tp / (tp + fn) : null;
        const specificity = tn + fp > 0 ? tn / (tn + fp) : null;
        const precision = tp + fp > 0 ? tp / (tp + fp) : null;
        const f1 = precision !== null && sensitivity !== null && precision + sensitivity > 0
          ? (2 * precision * sensitivity) / (precision + sensitivity)
          : null;
        const auroc = binaryAuroc(
          probabilities.map((prob) => prob[label] || 0),
          truth.map((actual) => actual === label)
        );
        return { label, tp, fp, tn, fn, sensitivity, specificity, precision, f1, auroc };
      });
      const avg = (key) => {
        const vals = rows.map((row) => row[key]).filter((v) => v !== null && Number.isFinite(v));
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      };
      return {
        rows,
        macro: {
          sensitivity: avg("sensitivity"),
          specificity: avg("specificity"),
          precision: avg("precision"),
          f1: avg("f1"),
          auroc: avg("auroc"),
        },
      };
    }

    function formatMetric(value) {
      return value === null || !Number.isFinite(value) ? "-" : value.toFixed(4);
    }

    function formatPercentMetric(value) {
      return value === null || !Number.isFinite(value) ? "-" : `${(value * 100).toFixed(1)}%`;
    }

    function renderClassificationQualitySummary(fit, looFit, metrics, validationLabel = "LOO") {
      if (!classificationQualityCards) return;
      const cards = [
        {
          label: "Accuracy",
          value: formatPercentMetric(fit?.accuracy),
          note: looFit ? `${validationLabel} ${formatPercentMetric(looFit.accuracy)} (${looFit.correct}/${looFit.total})` : `${validationLabel} not computed`,
        },
        { label: "Macro AUROC", value: formatMetric(metrics?.macro?.auroc), note: "one-vs-rest" },
        { label: "Sensitivity", value: formatPercentMetric(metrics?.macro?.sensitivity), note: "macro recall" },
        { label: "Specificity", value: formatPercentMetric(metrics?.macro?.specificity), note: "macro TNR" },
        { label: "F1 Score", value: formatPercentMetric(metrics?.macro?.f1), note: "macro F1" },
      ];
      classificationQualityCards.innerHTML = cards.map((card) => `
        <div class="quality-card">
          <div class="quality-label">${htmlEscape(card.label)}</div>
          <div class="quality-value">${htmlEscape(card.value)}</div>
          <div class="quality-note">${htmlEscape(card.note)}</div>
        </div>
      `).join("");
    }

    function clearClassificationQualitySummary() {
      if (!classificationQualityCards) return;
      classificationQualityCards.innerHTML = `
        <div class="quality-card"><div class="quality-label">Accuracy</div><div class="quality-value">-</div><div class="quality-note">training / LOO</div></div>
        <div class="quality-card"><div class="quality-label">Macro AUROC</div><div class="quality-value">-</div><div class="quality-note">one-vs-rest</div></div>
        <div class="quality-card"><div class="quality-label">Sensitivity</div><div class="quality-value">-</div><div class="quality-note">macro recall</div></div>
        <div class="quality-card"><div class="quality-label">Specificity</div><div class="quality-value">-</div><div class="quality-note">macro true negative rate</div></div>
        <div class="quality-card"><div class="quality-label">F1 Score</div><div class="quality-value">-</div><div class="quality-note">macro harmonic mean</div></div>
      `;
    }

    function leaveOneOutClassifierAccuracy(x, y, labels, seed, modelType) {
      if (x.length < 4 || x.length > 80) return null;
      let correct = 0;
      for (let holdout = 0; holdout < x.length; holdout++) {
        const trainX = [];
        const trainY = [];
        for (let i = 0; i < x.length; i++) {
          if (i === holdout) continue;
          trainX.push(x[i]);
          trainY.push(y[i]);
        }
        const model = trainClassifierModel(modelType, trainX, trainY, labels, seed + holdout + 1);
        const pred = predictClassifierLabel(model, x[holdout]);
        if (pred === y[holdout]) correct += 1;
      }
      return { correct, total: x.length, accuracy: correct / Math.max(1, x.length) };
    }

    function classifierSubjectKey(meta, idx) {
      const sid = meta?.subject_id && String(meta.subject_id).trim() ? String(meta.subject_id).trim() : "";
      if (sid && sid !== "Unknown") return sid;
      const fallback = meta?.case_path || meta?.instance_id || meta?.pair_name || "";
      return fallback ? `Unknown:${fallback}` : `Unknown:row_${idx + 1}`;
    }

    function chooseClassifierVoteLabel(labels, counts, scoreSums = null) {
      let best = labels[0] || "";
      let bestCount = -1;
      let bestScore = -Infinity;
      labels.forEach((label) => {
        const count = counts[label] || 0;
        const score = scoreSums ? (scoreSums[label] || 0) : 0;
        if (count > bestCount || (count === bestCount && score > bestScore)) {
          best = label;
          bestCount = count;
          bestScore = score;
        }
      });
      return best;
    }

    function subjectVoteEvaluation(labels, truth, predictions, probabilities, rowMeta) {
      const groups = new Map();
      for (let i = 0; i < truth.length; i++) {
        const meta = rowMeta?.[i] || {};
        const key = classifierSubjectKey(meta, i);
        if (!groups.has(key)) {
          groups.set(key, {
            subject_id: meta.subject_id || key,
            rows: [],
            truthCounts: Object.fromEntries(labels.map((label) => [label, 0])),
            predCounts: Object.fromEntries(labels.map((label) => [label, 0])),
            probSums: Object.fromEntries(labels.map((label) => [label, 0])),
          });
        }
        const group = groups.get(key);
        group.rows.push(i);
        group.truthCounts[truth[i]] = (group.truthCounts[truth[i]] || 0) + 1;
        group.predCounts[predictions[i]] = (group.predCounts[predictions[i]] || 0) + 1;
        labels.forEach((label) => {
          group.probSums[label] = (group.probSums[label] || 0) + ((probabilities[i] || {})[label] || 0);
        });
      }
      const rows = [];
      const votedTruth = [];
      const votedPredictions = [];
      const votedProbabilities = [];
      for (const group of groups.values()) {
        const truthLabel = chooseClassifierVoteLabel(labels, group.truthCounts);
        const predLabel = chooseClassifierVoteLabel(labels, group.predCounts, group.probSums);
        const avgProb = Object.fromEntries(labels.map((label) => [
          label,
          (group.probSums[label] || 0) / Math.max(1, group.rows.length),
        ]));
        votedTruth.push(truthLabel);
        votedPredictions.push(predLabel);
        votedProbabilities.push(avgProb);
        rows.push({
          subject_id: group.subject_id,
          case_count: group.rows.length,
          truth: truthLabel,
          prediction: predLabel,
          correct: truthLabel === predLabel ? 1 : 0,
          truth_vote_counts: labels.map((label) => `${label}:${group.truthCounts[label] || 0}`).join("; "),
          prediction_vote_counts: labels.map((label) => `${label}:${group.predCounts[label] || 0}`).join("; "),
          mean_probabilities: labels.map((label) => `${label}:${avgProb[label].toFixed(6)}`).join("; "),
        });
      }
      const correct = votedPredictions.reduce((sum, pred, idx) => sum + (pred === votedTruth[idx] ? 1 : 0), 0);
      return {
        y: votedTruth,
        predictions: votedPredictions,
        probabilities: votedProbabilities,
        rows,
        correct,
        total: votedTruth.length,
        accuracy: correct / Math.max(1, votedTruth.length),
      };
    }

    function leaveOneSubjectOutClassifierAccuracy(x, y, labels, rowMeta, seed, modelType) {
      const subjectKeys = x.map((_, idx) => classifierSubjectKey(rowMeta?.[idx] || {}, idx));
      const uniqueSubjects = Array.from(new Set(subjectKeys));
      if (uniqueSubjects.length < 3 || x.length > 160) return null;
      let correct = 0;
      let total = 0;
      uniqueSubjects.forEach((subjectKey, foldIdx) => {
        const trainX = [];
        const trainY = [];
        const testX = [];
        const testY = [];
        const testMeta = [];
        for (let i = 0; i < x.length; i++) {
          if (subjectKeys[i] === subjectKey) {
            testX.push(x[i]);
            testY.push(y[i]);
            testMeta.push(rowMeta?.[i] || {});
          } else {
            trainX.push(x[i]);
            trainY.push(y[i]);
          }
        }
        if (new Set(trainY).size < 2 || !testX.length) return;
        const model = trainClassifierModel(modelType, trainX, trainY, labels, seed + foldIdx + 1);
        const foldPredictions = testX.map((row) => predictClassifierLabel(model, row));
        const foldProbabilities = testX.map((row) => predictClassifierProbs(model, row));
        const voted = subjectVoteEvaluation(labels, testY, foldPredictions, foldProbabilities, testMeta);
        if (!voted.total) return;
        correct += voted.correct;
        total += voted.total;
      });
      return total ? { correct, total, accuracy: correct / total } : null;
    }

    function confusionRows(labels, truth, predictions) {
      const rows = [];
      for (const actual of labels) {
        for (const predicted of labels) {
          let count = 0;
          for (let i = 0; i < truth.length; i++) {
            if (truth[i] === actual && predictions[i] === predicted) count += 1;
          }
          rows.push({ actual, predicted, count });
        }
      }
      return rows;
    }

    function classifierShapStyleImportance(model, x, y, featureNames) {
      if (!x.length) return [];
      const p = featureNames.length;
      const means = Array.from({ length: p }, (_, j) => mean(x.map((row) => row[j])));
      const rows = featureNames.map((feature, j) => {
        let signed = 0;
        let abs = 0;
        for (let i = 0; i < x.length; i++) {
          const label = y[i];
          const base = predictClassifierProbs(model, x[i])[label] || 0;
          const occluded = x[i].slice();
          occluded[j] = means[j];
          const changed = predictClassifierProbs(model, occluded)[label] || 0;
          const contribution = base - changed;
          signed += contribution;
          abs += Math.abs(contribution);
        }
        return {
          feature,
          meanContribution: signed / x.length,
          meanAbsContribution: abs / x.length,
        };
      });
      return rows.sort((a, b) => b.meanAbsContribution - a.meanAbsContribution);
    }

    function buildClassificationDatasetFromVisibleRows() {
      const candidateFeatures = Array.from(classifierSelectedFeatures).filter((feature) => numericFeatures.includes(feature));
      if (candidateFeatures.length < 1) {
        return { error: "Select at least one classifier feature." };
      }
      const classFeature = classifierClassFeature.value || "";
      if (!classFeature) {
        return { error: "Select a classification class feature and define class rules first." };
      }
      if (!classifierClassRules.some((rule) => normalizedTsneColorRule(rule))) {
        return { error: "Define at least two classification-specific class rules before running classification." };
      }
      const visibleRows = plottedRows.filter((_, i) => rowVisibility[i]);
      if (visibleRows.length < 3) {
        return { error: "Need at least three visible rows for classification." };
      }
      const colorInfo = buildClassifierClassGroups(visibleRows, classFeature);
      renderClassifierClassRuleTable(colorInfo.groups);
      const groupLookup = new Map(colorInfo.groups.map((group) => [group.key, group]));
      const labeled = [];
      for (let i = 0; i < visibleRows.length; i++) {
        const assignment = colorInfo.assignments[i];
        if (!assignment || assignment.key === "__unlabeled__") continue;
        const group = groupLookup.get(assignment.key);
        if (!group) continue;
        labeled.push({
          row: visibleRows[i],
          label: group.label,
          color: group.color || "#1f77b4",
        });
      }
      const labels = Array.from(new Set(labeled.map((item) => item.label)));
      if (labels.length < 2) {
        return { error: "Need at least two user-defined classes with matched visible rows." };
      }
      if (labeled.length < labels.length + 3) {
        return { error: "Not enough labeled visible rows to train a classifier." };
      }

      const complete = buildCompleteCaseNumericMatrix(labeled.map((item) => item.row), candidateFeatures, {
        standardize: true,
      });
      if (complete.features.length < 1) {
        return { error: "No selected classifier features have complete observed variation for classification." };
      }

      const keptByRow = new Map(labeled.map((item) => [item.row, item]));
      const x = complete.matrix;
      const y = complete.rows.map((row) => keptByRow.get(row)?.label || "");
      const rowMeta = complete.rows.map((row) => ({
        subject_id: getSubjectId(row),
        case_path: row.case_path || "",
        instance_id: row.instance_id || "",
        pair_name: row.pair_name || "",
      }));
      const completeLabels = Array.from(new Set(y));
      if (completeLabels.length < 2) {
        return { error: "Complete-case classifier data contain fewer than two classes. Narrow selected predictors or include more rows." };
      }
      if (x.length < completeLabels.length + 3) {
        return { error: "Not enough complete observed rows to train a classifier. Narrow selected predictors or include more rows." };
      }
      return {
        x,
        y,
        labels: completeLabels,
        features: complete.features,
        colors: Object.fromEntries(labeled.map((item) => [item.label, item.color])),
        rowMeta,
        labeledCount: complete.rows.length,
        visibleCount: visibleRows.length,
        classFeature,
        completeCaseMeta: complete,
      };
    }

    function prepareClassifierDataset() {
      const base = buildClassificationDatasetFromVisibleRows();
      if (base.error) return base;
      const space = classifierFeatureSpace.value || "selected_features";
      if (space === "principal_components") {
        const pc = buildPrincipalComponentDataset(base);
        if (!pc) return { error: "Principal components are not available for the selected labeled dataset." };
        return { ...base, x: pc.x, features: pc.features, featureSpace: "principal_components", pcMeta: pc.pcMeta };
      }
      return { ...base, featureSpace: "selected_features" };
    }

    function renderClassificationPanel() {
      const dataset = prepareClassifierDataset();
      if (dataset.error) {
        lastClassifierExportRows = [];
        classifierStatusEl.textContent = dataset.error;
        classifierHead.innerHTML = "";
        classifierBody.innerHTML = "";
        clearClassificationQualitySummary();
        safeNewPlot("classifierMetricPlot", [], { title: "No classification quality metrics" }, { responsive: true });
        safeNewPlot("classifierRocPlot", [], { title: "No ROC curve" }, { responsive: true });
        safeNewPlot("classifierPerfPlot", [], { title: "No classification model" }, { responsive: true });
        safeNewPlot("classifierShapPlot", [], { title: "No SHAP-style explainer" }, { responsive: true });
        return;
      }
      const seed = classifierSeed();
      const modelType = classifierModelType.value || "random_forest";
      const modelLabel = classifierModelLabel(modelType);
      const evaluationUnit = classifierEvaluationUnit?.value || "row";
      const evaluationLabel = evaluationUnit === "subject" ? "Unique subjects (majority vote)" : "Rows / cases";
      const model = trainClassifierModel(modelType, dataset.x, dataset.y, dataset.labels, seed);
      const rowFit = evaluateClassifier(model, dataset.x, dataset.y);
      const voteFit = evaluationUnit === "subject"
        ? subjectVoteEvaluation(dataset.labels, dataset.y, rowFit.predictions, rowFit.probabilities, dataset.rowMeta)
        : null;
      const fit = voteFit || rowFit;
      const evaluationTruth = voteFit ? voteFit.y : dataset.y;
      const evaluationPredictions = voteFit ? voteFit.predictions : rowFit.predictions;
      const evaluationProbabilities = voteFit ? voteFit.probabilities : rowFit.probabilities;
      const looFit = evaluationUnit === "subject"
        ? leaveOneSubjectOutClassifierAccuracy(dataset.x, dataset.y, dataset.labels, dataset.rowMeta, seed, modelType)
        : leaveOneOutClassifierAccuracy(dataset.x, dataset.y, dataset.labels, seed, modelType);
      const metrics = classificationMetricRows(dataset.labels, evaluationTruth, evaluationPredictions, evaluationProbabilities);
      const shapRows = classifierShapStyleImportance(model, dataset.x, dataset.y, dataset.features);
      const rocRows = classifierRocRows(dataset.labels, evaluationTruth, evaluationProbabilities);
      const predictorList = dataset.features.join("; ");
      lastClassifierExportRows = [{
        section: "model_summary",
        model_type: modelType,
        model_label: modelLabel,
        seed,
        class_feature: dataset.classFeature || "",
        feature_space: dataset.featureSpace,
        evaluation_unit: evaluationUnit,
        evaluation_label: evaluationLabel,
        class_count: dataset.labels.length,
        classes: dataset.labels.join("; "),
        labeled_rows: dataset.y.length,
        visible_rows: dataset.visibleCount || dataset.y.length,
        evaluated_units: fit.total || evaluationTruth.length,
        subject_count: voteFit ? voteFit.total : Array.from(new Set(dataset.rowMeta.map((meta, idx) => classifierSubjectKey(meta, idx)))).length,
        predictor_count: dataset.features.length,
        predictors_used: predictorList,
        training_accuracy: fit.accuracy,
        row_training_accuracy: rowFit.accuracy,
        loo_accuracy: looFit ? looFit.accuracy : null,
        loo_correct: looFit ? looFit.correct : null,
        loo_total: looFit ? looFit.total : null,
        macro_auroc: metrics.macro.auroc,
        macro_sensitivity: metrics.macro.sensitivity,
        macro_specificity: metrics.macro.specificity,
        macro_precision: metrics.macro.precision,
        macro_f1: metrics.macro.f1,
        complete_case_rows_used: dataset.completeCaseMeta?.usedRowCount || dataset.y.length,
        rows_skipped_missing_selected_values: dataset.completeCaseMeta?.droppedRows?.length || 0,
      }];
      for (const row of metrics.rows) {
        lastClassifierExportRows.push({
          section: "class_metric",
          model_type: modelType,
          model_label: modelLabel,
          seed,
          class_feature: dataset.classFeature || "",
          feature_space: dataset.featureSpace,
          evaluation_unit: evaluationUnit,
          evaluation_label: evaluationLabel,
          class_label: row.label,
          tp: row.tp,
          fp: row.fp,
          tn: row.tn,
          fn: row.fn,
          auroc: row.auroc,
          sensitivity: row.sensitivity,
          specificity: row.specificity,
          precision: row.precision,
          f1: row.f1,
          predictors_used: predictorList,
        });
      }
      shapRows.forEach((row, idx) => {
        let details = "";
        if (dataset.featureSpace === "principal_components" && dataset.pcMeta) {
          const meta = dataset.pcMeta.find((item) => item.component === row.feature);
          details = meta ? `variance=${meta.variancePct.toFixed(2)}%; ${meta.topLoadings.map((item) => `${item.feature} (${item.loading.toFixed(3)})`).join("; ")}` : "";
        }
        lastClassifierExportRows.push({
          section: "shap_style_importance",
          model_type: modelType,
          model_label: modelLabel,
          seed,
          class_feature: dataset.classFeature || "",
          feature_space: dataset.featureSpace,
          evaluation_unit: evaluationUnit,
          evaluation_label: evaluationLabel,
          rank: idx + 1,
          predictor: row.feature,
          mean_abs_contribution: row.meanAbsContribution,
          mean_signed_contribution: row.meanContribution,
          details,
          predictors_used: predictorList,
        });
      });
      dataset.features.forEach((feature, idx) => {
        lastClassifierExportRows.push({
          section: "predictor_used",
          model_type: modelType,
          model_label: modelLabel,
          seed,
          class_feature: dataset.classFeature || "",
          feature_space: dataset.featureSpace,
          evaluation_unit: evaluationUnit,
          evaluation_label: evaluationLabel,
          rank: idx + 1,
          predictor: feature,
          predictors_used: predictorList,
        });
      });
      for (const row of rocRows) {
        lastClassifierExportRows.push({
          section: "roc_curve_point",
          model_type: modelType,
          model_label: modelLabel,
          seed,
          class_feature: dataset.classFeature || "",
          feature_space: dataset.featureSpace,
          evaluation_unit: evaluationUnit,
          evaluation_label: evaluationLabel,
          class_label: row.class_label,
          point_index: row.point_index,
          threshold: row.threshold,
          fpr: row.fpr,
          tpr: row.tpr,
          auroc: row.auroc,
          predictors_used: predictorList,
        });
      }
      classifierClassRules.forEach((rule, idx) => {
        lastClassifierExportRows.push({
          section: "class_rule",
          model_type: modelType,
          model_label: modelLabel,
          seed,
          class_feature: dataset.classFeature || "",
          evaluation_unit: evaluationUnit,
          evaluation_label: evaluationLabel,
          rank: idx + 1,
          class_label: rule.label || "",
          color: rule.color || "",
          enabled: rule.enabled === false ? "false" : "true",
          rule: tsneColorRuleDescription(rule),
          lower_operator: rule.lowerOperator || "",
          lower_value: rule.lowerValue || "",
          upper_operator: rule.upperOperator || "",
          upper_value: rule.upperValue || "",
        });
      });
      if (voteFit) {
        voteFit.rows.forEach((row, idx) => {
          lastClassifierExportRows.push({
            section: "subject_vote",
            model_type: modelType,
            model_label: modelLabel,
            seed,
            class_feature: dataset.classFeature || "",
            feature_space: dataset.featureSpace,
            evaluation_unit: evaluationUnit,
            evaluation_label: evaluationLabel,
            rank: idx + 1,
            subject_id: row.subject_id,
            case_count: row.case_count,
            actual: row.truth,
            predicted: row.prediction,
            correct: row.correct,
            truth_vote_counts: row.truth_vote_counts,
            prediction_vote_counts: row.prediction_vote_counts,
            mean_probabilities: row.mean_probabilities,
            predictors_used: predictorList,
          });
        });
      }
      classifierStatusEl.textContent =
        `${modelLabel} classifier | seed=${seed} | evaluation=${evaluationLabel} | class feature=${dataset.classFeature || "-"} | classes=${dataset.labels.length} | complete observed labeled rows=${dataset.y.length}/${dataset.visibleCount || dataset.y.length} | evaluated units=${fit.total || evaluationTruth.length}${voteFit ? ` subjects from ${dataset.y.length} case predictions` : " rows"} | inputs=${dataset.features.length} (${dataset.featureSpace}) | accuracy=${(fit.accuracy * 100).toFixed(1)}% | ${evaluationUnit === "subject" ? "leave-one-subject-out" : "leave-one-out"} accuracy=${looFit ? `${(looFit.accuracy * 100).toFixed(1)}% (${looFit.correct}/${looFit.total})` : "not computed for this data size"} | macro AUROC=${formatMetric(metrics.macro.auroc)} | macro sensitivity=${formatMetric(metrics.macro.sensitivity)} | macro specificity=${formatMetric(metrics.macro.specificity)} | macro F1=${formatMetric(metrics.macro.f1)}. ${summarizeCompleteCase(dataset.completeCaseMeta)} SHAP-style values remain case-level feature-occlusion contributions to true-class probability.`;
      renderClassificationQualitySummary(fit, looFit, metrics, evaluationUnit === "subject" ? "LOSO" : "LOO");

      const qualityMetricNames = ["Accuracy", "Macro AUROC", "Sensitivity", "Specificity", "F1"];
      const qualityMetricValues = [fit.accuracy, metrics.macro.auroc, metrics.macro.sensitivity, metrics.macro.specificity, metrics.macro.f1];
      safeNewPlot("classifierMetricPlot", [{
        type: "bar",
        x: qualityMetricNames,
        y: qualityMetricValues.map((v) => v === null ? 0 : v),
        text: qualityMetricValues.map((v) => formatPercentMetric(v)),
        textposition: "auto",
        marker: { color: ["#0f766e", "#2563eb", "#16a34a", "#b45309", "#be123c"] },
        hovertemplate: "%{x}: %{text}<extra></extra>",
      }], {
        title: `Classification quality summary (${evaluationLabel})`,
        yaxis: { title: "Score", range: [0, 1] },
        margin: { t: 48, r: 20, b: 60, l: 64 },
      }, { responsive: true });

      const rocTraces = dataset.labels.map((label) => {
        const curve = rocRows.filter((row) => row.class_label === label);
        const metric = metrics.rows.find((row) => row.label === label);
        return {
          type: "scatter",
          mode: "lines+markers",
          x: curve.map((row) => row.fpr),
          y: curve.map((row) => row.tpr),
          name: `${label} (AUROC ${formatMetric(metric ? metric.auroc : null)})`,
          text: curve.map((row) => `class=${label}<br>threshold=${row.threshold}<br>FPR=${formatMetric(row.fpr)}<br>TPR=${formatMetric(row.tpr)}`),
          hovertemplate: "%{text}<extra></extra>",
        };
      });
      rocTraces.push({
        type: "scatter",
        mode: "lines",
        x: [0, 1],
        y: [0, 1],
        name: "Chance",
        line: { color: "rgba(100,116,139,0.65)", dash: "dash" },
        hovertemplate: "chance line<extra></extra>",
      });
      safeNewPlot("classifierRocPlot", rocTraces, {
        title: `ROC curve by class (one-vs-rest, ${evaluationLabel})`,
        xaxis: { title: "False positive rate", range: [0, 1] },
        yaxis: { title: "True positive rate", range: [0, 1] },
        margin: { t: 48, r: 16, b: 58, l: 62 },
        legend: { orientation: "v", x: 0.03, y: 0.97, bgcolor: "rgba(255,255,255,0.72)" },
      }, { responsive: true });

      const conf = confusionRows(dataset.labels, evaluationTruth, evaluationPredictions);
      const confTotals = Object.fromEntries(dataset.labels.map((actual) => [
        actual,
        conf.filter((row) => row.actual === actual).reduce((sum, row) => sum + row.count, 0),
      ]));
      for (const row of conf) {
        const actualTotal = confTotals[row.actual] || 0;
        lastClassifierExportRows.push({
          section: "confusion_matrix",
          model_type: modelType,
          model_label: modelLabel,
          seed,
          class_feature: dataset.classFeature || "",
          feature_space: dataset.featureSpace,
          evaluation_unit: evaluationUnit,
          evaluation_label: evaluationLabel,
          actual: row.actual,
          predicted: row.predicted,
          count: row.count,
          row_normalized_percent: actualTotal > 0 ? (row.count / actualTotal) * 100 : null,
          predictors_used: predictorList,
        });
      }
      const confusionZ = dataset.labels.map((actual) => dataset.labels.map((predicted) => {
        const row = conf.find((item) => item.actual === actual && item.predicted === predicted);
        const total = confTotals[actual] || 0;
        return row && total > 0 ? (row.count / total) * 100 : 0;
      }));
      const confusionText = dataset.labels.map((actual) => dataset.labels.map((predicted) => {
        const row = conf.find((item) => item.actual === actual && item.predicted === predicted);
        const total = confTotals[actual] || 0;
        const count = row ? row.count : 0;
        const pct = total > 0 ? (count / total) * 100 : 0;
        return `actual=${actual}<br>predicted=${predicted}<br>count=${count}/${total}<br>row-normalized=${pct.toFixed(1)}%`;
      }));
      safeNewPlot("classifierPerfPlot", [{
        type: "heatmap",
        x: dataset.labels,
        y: dataset.labels,
        z: confusionZ,
        text: confusionText,
        colorscale: "Blues",
        zmin: 0,
        zmax: 100,
        showscale: true,
        colorbar: { title: "% of actual class" },
        hovertemplate: "%{text}<extra></extra>",
      }], {
        title: `Classification confusion matrix (${evaluationLabel}, row-normalized)`,
        xaxis: { title: "Predicted class" },
        yaxis: { title: "Actual class" },
        margin: { t: 48, r: 18, b: 64, l: 78 },
      }, { responsive: true });

      const top = shapRows.slice(0, 20).reverse();
      safeNewPlot("classifierShapPlot", [{
        type: "bar",
        orientation: "h",
        y: top.map((row) => row.feature.startsWith("PC") ? row.feature : featureShortLabel(row.feature, featureCategory(row.feature))),
        x: top.map((row) => row.meanAbsContribution),
        text: top.map((row) => `${row.feature}<br>mean |contribution|=${row.meanAbsContribution.toFixed(4)}<br>mean signed contribution=${row.meanContribution.toFixed(4)}`),
        hovertemplate: "%{text}<extra></extra>",
        marker: { color: "rgba(22,101,52,0.72)" },
      }], {
        title: "SHAP-style feature importance by occlusion",
        xaxis: { title: "mean absolute contribution to true-class probability" },
        yaxis: { automargin: true },
        margin: { t: 48, r: 20, b: 58, l: 220 },
      }, { responsive: true });

      classifierHead.innerHTML = ["section", "label/rank", "feature/metric", "value_1", "value_2", "details"].map((c) => `<th>${c}</th>`).join("");
      const metricRows = metrics.rows.map((row) => `
        <tr>
          <td>classification_metric</td>
          <td>${htmlEscape(row.label)}</td>
          <td>AUROC / sensitivity / specificity / F1</td>
          <td>AUROC=${formatMetric(row.auroc)}; F1=${formatMetric(row.f1)}</td>
          <td>sensitivity=${formatMetric(row.sensitivity)}; specificity=${formatMetric(row.specificity)}</td>
          <td>TP=${row.tp}; FP=${row.fp}; TN=${row.tn}; FN=${row.fn}; precision=${formatMetric(row.precision)}</td>
        </tr>
      `).join("") + `
        <tr>
          <td>classification_metric</td>
          <td>macro_average</td>
          <td>macro AUROC / sensitivity / specificity / F1</td>
          <td>AUROC=${formatMetric(metrics.macro.auroc)}; F1=${formatMetric(metrics.macro.f1)}</td>
          <td>sensitivity=${formatMetric(metrics.macro.sensitivity)}; specificity=${formatMetric(metrics.macro.specificity)}</td>
          <td>Macro average across one-vs-rest class metrics.</td>
        </tr>
      `;
      const shapTableRows = shapRows.slice(0, 40).map((row, idx) => {
        let details = "";
        if (dataset.featureSpace === "principal_components" && dataset.pcMeta) {
          const meta = dataset.pcMeta.find((item) => item.component === row.feature);
          details = meta ? `variance=${meta.variancePct.toFixed(2)}%; ${meta.topLoadings.map((item) => `${item.feature} (${item.loading.toFixed(3)})`).join("; ")}` : "";
        }
        return `
          <tr>
            <td>shap_style_importance</td>
            <td>${idx + 1}</td>
            <td>${htmlEscape(row.feature)}</td>
            <td>${row.meanAbsContribution.toFixed(6)}</td>
            <td>${row.meanContribution.toFixed(6)}</td>
            <td>${htmlEscape(details)}</td>
          </tr>
        `;
      }).join("");
      classifierBody.innerHTML = metricRows + shapTableRows;
    }

    function computeR2(rows, xKey, yKey) {
      const x = [];
      const y = [];
      for (const row of rows) {
        const xv = parseMaybeNumber(row[xKey]);
        const yv = parseMaybeNumber(row[yKey]);
        if (xv === null || yv === null) continue;
        x.push(xv);
        y.push(yv);
      }
      if (x.length < 2) return null;

      const meanX = x.reduce((a, b) => a + b, 0) / x.length;
      const meanY = y.reduce((a, b) => a + b, 0) / y.length;
      let num = 0;
      let denX = 0;
      let denY = 0;
      for (let i = 0; i < x.length; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
      }
      if (denX <= 0 || denY <= 0) return null;
      const r = num / Math.sqrt(denX * denY);
      return r * r;
    }

    function renderTable(rows, xKey, yKey) {
      const cols = Array.from(new Set(["case_path", "instance_id", "pair_name", "category", "panel", xKey, yKey, "subject_id"].filter(Boolean)));
      const displayRows = rows.slice(0, 250);
      rowsHead.innerHTML = ["show"].concat(cols).map(c => `<th>${c}</th>`).join("");
      rowsBody.innerHTML = displayRows.map((r, idx) => {
        const checked = rowVisibility[idx] ? "checked" : "";
        const disabled = rowFilterVisibility[idx] === false ? "disabled" : "";
        const title = rowFilterVisibility[idx] === false ? "Filtered out by row filter rules" : "Manually include/exclude this row";
        const cells = cols.map(c => `<td>${htmlEscape(r[c] ?? "")}</td>`).join("");
        return `<tr data-row-index="${idx}"><td><input type="checkbox" data-row-toggle="${idx}" ${checked} ${disabled} title="${title}"></td>${cells}</tr>`;
      }).join("");
      if (rows.length > displayRows.length) {
        rowsBody.innerHTML += `<tr><td colspan="${cols.length + 1}">Showing first ${displayRows.length} rows of ${rows.length}. Plots/statistics still use all visible rows.</td></tr>`;
      }

      rowsBody.querySelectorAll('input[data-row-toggle]').forEach((inputEl) => {
        inputEl.addEventListener("change", () => {
          const idx = Number(inputEl.getAttribute("data-row-toggle"));
          rowManualVisibility[idx] = !!inputEl.checked;
          rowVisibility[idx] = rowManualVisibility[idx] && rowFilterVisibility[idx] !== false;
          updateRowFilterStatus();
          rerenderFromVisibility();
        });
      });
    }

    function renderPlot(rows, xKey, yKey) {
      const tracesByCategory = {};
      let scatterPointCount = 0;
      for (let i = 0; i < rows.length; i++) {
        if (!rowVisibility[i]) continue;
        const row = rows[i];
        const xValue = parseMaybeNumber(row[xKey]);
        const yValue = parseMaybeNumber(row[yKey]);
        if (xValue === null || yValue === null) continue;
        const cat = row.category || "uncategorized";
        if (!tracesByCategory[cat]) tracesByCategory[cat] = [];
        tracesByCategory[cat].push({ row, idx: i, xValue, yValue });
        scatterPointCount += 1;
      }

      const traces = Object.entries(tracesByCategory).map(([cat, catRows]) => ({
        type: "scatter",
        mode: "markers",
        name: cat,
        x: catRows.map(obj => obj.xValue),
        y: catRows.map(obj => obj.yValue),
        text: catRows.map(obj => `${obj.row.pair_name || ""}<br>${obj.row.case_path || ""}`),
        customdata: catRows.map(obj => obj.idx),
        hovertemplate: "%{text}<br>X=%{x}<br>Y=%{y}<extra></extra>",
        marker: { size: 10, opacity: 0.85 }
      }));

      const visibleRows = rows.filter((_, i) => rowVisibility[i]);
      const reg = fitLinearRegression(visibleRows, xKey, yKey);
      const r2 = computeR2(visibleRows, xKey, yKey);
      lastScatterExportRows = [];
      for (let i = 0; i < rows.length; i++) {
        if (!rowVisibility[i]) continue;
        const row = rows[i];
        const xValue = parseMaybeNumber(row[xKey]);
        const yValue = parseMaybeNumber(row[yKey]);
        if (xValue === null || yValue === null) continue;
        lastScatterExportRows.push({
          visual_component: "scatter_plot",
          row_index: i,
          case_path: row.case_path || "",
          instance_id: row.instance_id || "",
          pair_name: row.pair_name || "",
          category: row.category || "",
          panel: row.panel || "",
          subject_id: row.subject_id || "",
          x_feature: xKey,
          y_feature: yKey,
          x_value: xValue,
          y_value: yValue,
          regression_slope: reg ? reg.slope : null,
          regression_intercept: reg ? reg.intercept : null,
          regression_predicted_y: reg && xValue !== null ? reg.slope * xValue + reg.intercept : null,
          r2,
        });
      }
      if (reg) {
        traces.push({
          type: "scatter",
          mode: "lines",
          name: "regression",
          x: [reg.minX, reg.maxX],
          y: [reg.slope * reg.minX + reg.intercept, reg.slope * reg.maxX + reg.intercept],
          line: { width: 2 }
        });
      }

      safeNewPlot("plot", traces, {
        title: `${xKey} vs ${yKey} (${scatterPointCount} plot-ready rows)`,
        xaxis: { title: xKey },
        yaxis: { title: yKey },
        margin: { t: 48, r: 20, b: 64, l: 64 }
      }, { responsive: true });

      const plotEl = document.getElementById("plot");
      if (plotEl && typeof plotEl.on === "function") {
        plotEl.on("plotly_click", (eventData) => {
          const point = eventData?.points?.[0];
          if (!point || point.customdata === undefined) return;
          const rowIdx = Number(point.customdata);
          highlightRow(rowIdx);
        });
      }
    }

    function fitLinearRegression(rows, xKey, yKey) {
      const x = [];
      const y = [];
      for (const row of rows) {
        const xv = parseMaybeNumber(row[xKey]);
        const yv = parseMaybeNumber(row[yKey]);
        if (xv === null || yv === null) continue;
        x.push(xv);
        y.push(yv);
      }
      if (x.length < 2) return null;

      const meanX = x.reduce((a, b) => a + b, 0) / x.length;
      const meanY = y.reduce((a, b) => a + b, 0) / y.length;
      let num = 0;
      let den = 0;
      for (let i = 0; i < x.length; i++) {
        const dx = x[i] - meanX;
        num += dx * (y[i] - meanY);
        den += dx * dx;
      }
      if (den <= 0) return null;
      const slope = num / den;
      const intercept = meanY - slope * meanX;
      return {
        slope,
        intercept,
        minX: Math.min(...x),
        maxX: Math.max(...x),
      };
    }

    function pearsonForFeature(rows, xKey, yKey) {
      const x = [];
      const y = [];
      for (const row of rows) {
        const xv = parseMaybeNumber(row[xKey]);
        const yv = parseMaybeNumber(row[yKey]);
        if (xv !== null && yv !== null) {
          x.push(xv);
          y.push(yv);
        }
      }
      if (x.length < 3) return null;
      const mx = mean(x);
      const my = mean(y);
      let num = 0;
      let dx2 = 0;
      let dy2 = 0;
      for (let i = 0; i < x.length; i++) {
        const dx = x[i] - mx;
        const dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
      }
      if (dx2 <= 0 || dy2 <= 0) return null;
      return num / Math.sqrt(dx2 * dy2);
    }

    function pearsonWithCount(rows, xKey, yKey) {
      const x = [];
      const y = [];
      for (const row of rows) {
        const xv = parseMaybeNumber(row[xKey]);
        const yv = parseMaybeNumber(row[yKey]);
        if (xv !== null && yv !== null) {
          x.push(xv);
          y.push(yv);
        }
      }
      if (x.length < 3) return null;
      const mx = mean(x);
      const my = mean(y);
      let num = 0;
      let dx2 = 0;
      let dy2 = 0;
      for (let i = 0; i < x.length; i++) {
        const dx = x[i] - mx;
        const dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
      }
      if (dx2 <= 0 || dy2 <= 0) return null;
      const r = num / Math.sqrt(dx2 * dy2);
      return { r, absR: Math.abs(r), n: x.length };
    }

    function setGroupedFeatureValue(groupSelect, featureSelect, feature, silent = false) {
      if (!feature || !groupSelect || !featureSelect) return false;
      const category = featureCategory(feature);
      groupSelect.value = category;
      if (silent) {
        const options = numericFeatures.filter((item) => featureCategory(item) === category);
        featureSelect.innerHTML = "";
        for (const optionFeature of options) {
          const opt = document.createElement("option");
          opt.value = optionFeature;
          opt.textContent = featureShortLabel(optionFeature, category);
          opt.title = optionFeature;
          featureSelect.appendChild(opt);
        }
      } else {
        groupSelect.dispatchEvent(new Event("change"));
      }
      featureSelect.value = feature;
      return featureSelect.value === feature;
    }

    function plotCorrelationFeaturePair(featureA, featureB) {
      const okX = setGroupedFeatureValue(xFeatureGroup, xFeature, featureA, true);
      const okY = setGroupedFeatureValue(yFeatureGroup, yFeature, featureB, true);
      if (!okX || !okY) {
        corrStatusEl.textContent = `Could not locate selected feature pair in scatter selectors: ${featureA} vs ${featureB}`;
        return;
      }
      refreshAnalysis();
      corrStatusEl.textContent = `Plotted selected correlation pair: ${featureA} vs ${featureB}`;
      const plotEl = document.getElementById("plot");
      if (plotEl) plotEl.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function swapScatterAxes() {
      const oldX = xFeature.value;
      const oldY = yFeature.value;
      if (!oldX || !oldY) {
        setStatus("Select both X and Y features before swapping axes.");
        return;
      }
      const okX = setGroupedFeatureValue(xFeatureGroup, xFeature, oldY, true);
      const okY = setGroupedFeatureValue(yFeatureGroup, yFeature, oldX, true);
      if (!okX || !okY) {
        setStatus("Could not swap axes because one selected feature is unavailable.");
        return;
      }
      refreshAnalysis();
    }

    function selectedCorrelationExportRows() {
      const inputs = Array.from(corrBody.querySelectorAll("input[data-corr-export-index]:checked"));
      if (!inputs.length) return [];
      return inputs
        .map((inputEl) => lastCorrelationExportRows[Number(inputEl.getAttribute("data-corr-export-index"))])
        .filter((row) => row && typeof row === "object");
    }

    function renderCorrelationFilterPanel() {
      lastCorrelationExportRows = [];
      const visibleRows = plottedRows.filter((_, i) => rowVisibility[i]);
      const selectedFeatures = Array.from(corrSelectedFeatures).filter((feature) => numericFeatures.includes(feature));
      const thresholdRaw = parseMaybeNumber(corrThresholdInput.value);
      const threshold = thresholdRaw === null ? 0.5 : Math.max(0, Math.min(1, thresholdRaw));
      corrThresholdInput.value = threshold;
      const r2ThresholdRaw = parseMaybeNumber(corrR2ThresholdInput.value);
      const r2Threshold = r2ThresholdRaw === null ? 0 : Math.max(0, Math.min(1, r2ThresholdRaw));
      corrR2ThresholdInput.value = r2Threshold;
      const useAbs = !!corrUseAbsInput.checked;
      if (visibleRows.length < 3) {
        corrStatusEl.textContent = "Need at least 3 visible rows to compute correlations.";
        corrHead.innerHTML = "";
        corrBody.innerHTML = "";
        return;
      }
      if (selectedFeatures.length < 2) {
        corrStatusEl.textContent = "Select at least 2 correlation features to compute feature-to-feature correlations.";
        corrHead.innerHTML = "";
        corrBody.innerHTML = "";
        return;
      }
      const rows = [];
      for (let i = 0; i < selectedFeatures.length; i++) {
        for (let j = i + 1; j < selectedFeatures.length; j++) {
          const featureA = selectedFeatures[i];
          const featureB = selectedFeatures[j];
          const stat = pearsonWithCount(visibleRows, featureA, featureB);
          if (!stat) continue;
          const score = useAbs ? stat.absR : stat.r;
          const rSquared = stat.r * stat.r;
          if (score >= threshold && rSquared >= r2Threshold) {
            rows.push({
              feature_a: featureA,
              feature_b: featureB,
              correlation_r: stat.r,
              abs_correlation: stat.absR,
              r_squared: rSquared,
              complete_rows: stat.n,
              visible_rows: visibleRows.length,
              screened_feature_count: selectedFeatures.length,
              screened_features: selectedFeatures.join("; "),
              threshold,
              r_squared_threshold: r2Threshold,
              threshold_mode: useAbs ? "absolute_r" : "signed_positive_r",
            });
          }
        }
      }
      rows.sort((a, b) => b.abs_correlation - a.abs_correlation || b.complete_rows - a.complete_rows);
      lastCorrelationExportRows = rows;
      corrStatusEl.textContent = `Found ${rows.length} feature pairs with ${useAbs ? "|r|" : "r"} >= ${threshold.toFixed(2)} and R² >= ${r2Threshold.toFixed(2)} across ${visibleRows.length} visible rows and ${selectedFeatures.length} checked features. Checked rows will be exported; click a row to plot that pair.`;
      corrHead.innerHTML = ["export", "rank", "feature A", "feature B", "r", "|r|", "R²", "complete rows"].map((c) => `<th>${c}</th>`).join("");
      corrBody.innerHTML = rows.map((row, idx) => `
        <tr data-corr-feature-a="${htmlEscape(row.feature_a)}" data-corr-feature-b="${htmlEscape(row.feature_b)}" style="cursor:pointer;">
          <td><input type="checkbox" data-corr-export-index="${idx}" checked title="Include this correlation pair in CSV export"></td>
          <td>${idx + 1}</td>
          <td>${htmlEscape(row.feature_a)}</td>
          <td>${htmlEscape(row.feature_b)}</td>
          <td>${row.correlation_r.toFixed(4)}</td>
          <td>${row.abs_correlation.toFixed(4)}</td>
          <td>${row.r_squared.toFixed(4)}</td>
          <td>${row.complete_rows}</td>
        </tr>
      `).join("");
      corrBody.querySelectorAll("tr[data-corr-feature-a]").forEach((tr) => {
        tr.addEventListener("click", () => {
          plotCorrelationFeaturePair(tr.getAttribute("data-corr-feature-a"), tr.getAttribute("data-corr-feature-b"));
        });
      });
      corrBody.querySelectorAll("input[data-corr-export-index]").forEach((inputEl) => {
        inputEl.addEventListener("click", (event) => event.stopPropagation());
        inputEl.addEventListener("change", () => {
          const selectedCount = corrBody.querySelectorAll("input[data-corr-export-index]:checked").length;
          corrStatusEl.textContent = `Found ${rows.length} matching feature pairs. ${selectedCount} checked for export. Click a row to plot that pair.`;
        });
      });
    }

    function completeRowsForFeatures(rows, features) {
      return rows.filter((row) => features.every((feature) => parseMaybeNumber(row[feature]) !== null));
    }

    function transpose(matrix) {
      return matrix[0].map((_, col) => matrix.map((row) => row[col]));
    }

    function multiplyMatrices(a, b) {
      const out = Array.from({ length: a.length }, () => Array(b[0].length).fill(0));
      for (let i = 0; i < a.length; i++) {
        for (let k = 0; k < b.length; k++) {
          const aik = a[i][k];
          if (aik === 0) continue;
          for (let j = 0; j < b[0].length; j++) out[i][j] += aik * b[k][j];
        }
      }
      return out;
    }

    function multiplyMatrixVector(a, v) {
      return a.map((row) => row.reduce((sum, value, i) => sum + value * v[i], 0));
    }

    function invertMatrix(matrix) {
      const n = matrix.length;
      const aug = matrix.map((row, i) => row.slice().concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
      for (let col = 0; col < n; col++) {
        let pivot = col;
        for (let r = col + 1; r < n; r++) {
          if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r;
        }
        if (Math.abs(aug[pivot][col]) < 1e-10) return null;
        if (pivot !== col) {
          const tmp = aug[col];
          aug[col] = aug[pivot];
          aug[pivot] = tmp;
        }
        const div = aug[col][col];
        for (let j = 0; j < 2 * n; j++) aug[col][j] /= div;
        for (let r = 0; r < n; r++) {
          if (r === col) continue;
          const factor = aug[r][col];
          if (factor === 0) continue;
          for (let j = 0; j < 2 * n; j++) aug[r][j] -= factor * aug[col][j];
        }
      }
      return aug.map((row) => row.slice(n));
    }

    function fitOls(rows, responseFeature, predictorFeatures) {
      const n = rows.length;
      const p = predictorFeatures.length;
      if (n <= p + 1) return null;
      const y = rows.map((row) => parseMaybeNumber(row[responseFeature]));
      const x = rows.map((row) => [1].concat(predictorFeatures.map((feature) => parseMaybeNumber(row[feature]))));
      const xt = transpose(x);
      const xtx = multiplyMatrices(xt, x);
      let inv = invertMatrix(xtx);
      if (!inv) {
        // Tiny ridge only for numeric stability; ANOVA remains approximate if predictors are nearly collinear.
        for (let i = 1; i < xtx.length; i++) xtx[i][i] += 1e-8;
        inv = invertMatrix(xtx);
      }
      if (!inv) return null;
      const xty = multiplyMatrixVector(xt, y);
      const beta = multiplyMatrixVector(inv, xty);
      const fitted = x.map((row) => row.reduce((sum, value, i) => sum + value * beta[i], 0));
      const meanY = mean(y);
      const sse = y.reduce((sum, value, i) => sum + Math.pow(value - fitted[i], 2), 0);
      const sst = y.reduce((sum, value) => sum + Math.pow(value - meanY, 2), 0);
      const ssr = Math.max(0, sst - sse);
      const dfModel = p;
      const dfError = n - p - 1;
      const r2 = sst > 0 ? 1 - sse / sst : null;
      const adjR2 = r2 === null || dfError <= 0 ? null : 1 - (1 - r2) * ((n - 1) / dfError);
      return { n, p, beta, fitted, sse, sst, ssr, dfModel, dfError, r2, adjR2 };
    }

    function logGamma(z) {
      const coeff = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.001208650973866179, -0.000005395239384953];
      let x = z;
      let y = z;
      let tmp = x + 5.5;
      tmp -= (x + 0.5) * Math.log(tmp);
      let ser = 1.000000000190015;
      for (let j = 0; j < coeff.length; j++) ser += coeff[j] / ++y;
      return -tmp + Math.log(2.5066282746310005 * ser / x);
    }

    function betaContinuedFraction(a, b, x) {
      const maxIter = 100;
      const eps = 3e-7;
      const fpmin = 1e-30;
      let qab = a + b;
      let qap = a + 1;
      let qam = a - 1;
      let c = 1;
      let d = 1 - qab * x / qap;
      if (Math.abs(d) < fpmin) d = fpmin;
      d = 1 / d;
      let h = d;
      for (let m = 1; m <= maxIter; m++) {
        const m2 = 2 * m;
        let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
        d = 1 + aa * d;
        if (Math.abs(d) < fpmin) d = fpmin;
        c = 1 + aa / c;
        if (Math.abs(c) < fpmin) c = fpmin;
        d = 1 / d;
        h *= d * c;
        aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
        d = 1 + aa * d;
        if (Math.abs(d) < fpmin) d = fpmin;
        c = 1 + aa / c;
        if (Math.abs(c) < fpmin) c = fpmin;
        d = 1 / d;
        const del = d * c;
        h *= del;
        if (Math.abs(del - 1) < eps) break;
      }
      return h;
    }

    function regularizedBeta(x, a, b) {
      if (x <= 0) return 0;
      if (x >= 1) return 1;
      const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
      if (x < (a + 1) / (a + b + 2)) return (bt * betaContinuedFraction(a, b, x)) / a;
      return 1 - (bt * betaContinuedFraction(b, a, 1 - x)) / b;
    }

    function fSurvival(fValue, df1, df2) {
      if (!Number.isFinite(fValue) || fValue < 0 || df1 <= 0 || df2 <= 0) return null;
      const x = (df1 * fValue) / (df1 * fValue + df2);
      return Math.max(0, Math.min(1, 1 - regularizedBeta(x, df1 / 2, df2 / 2)));
    }

    function selectMlrPredictors(rows, responseFeature, candidateFeatures) {
      const dataset = buildCompleteCaseRowsForRegression(rows, responseFeature, candidateFeatures);
      const responseRows = dataset.rows;
      const ranked = dataset.predictors
        .map((feature) => ({ feature, r: pearsonForFeature(responseRows, feature, responseFeature) }))
        .filter((item) => item.r !== null && Number.isFinite(item.r))
        .sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
      const selected = [];
      let rejectedForFit = 0;
      for (const item of ranked) {
        const trial = selected.concat(item.feature);
        const trialModel = responseRows.length > trial.length + MIN_MLR_ERROR_DF + 1 ? fitOls(responseRows, responseFeature, trial) : null;
        if (trialModel && trialModel.dfError >= MIN_MLR_ERROR_DF) {
          selected.push(item.feature);
        } else {
          rejectedForFit += 1;
        }
        if (selected.length >= 30) break;
      }
      return {
        predictors: selected,
        rows: responseRows,
        droppedCount: Math.max(0, dataset.predictors.length - selected.length),
        rejectedForFit,
        completeCaseMeta: dataset.completeCaseMeta,
        mode: selected.length === dataset.predictors.length ? "observed complete-case fit-stable predictors" : "observed complete-case strongest fit-stable predictors",
      };
    }

    function summarizeModelFit(model) {
      if (!model) return null;
      return {
        n: model.n,
        p: model.p,
        r2: model.r2,
        adjR2: model.adjR2,
        sse: model.sse,
        dfError: model.dfError,
      };
    }

    function compareModelFits(current, baseline) {
      if (!current || !baseline) return "Baseline comparison: not available.";
      const deltaR2 = current.r2 !== null && baseline.r2 !== null ? current.r2 - baseline.r2 : null;
      const deltaAdjR2 = current.adjR2 !== null && baseline.adjR2 !== null ? current.adjR2 - baseline.adjR2 : null;
      const deltaSse = current.sse - baseline.sse;
      return `Baseline comparison: all selected automatically R²=${baseline.r2 === null ? "-" : baseline.r2.toFixed(4)}, adj R²=${baseline.adjR2 === null ? "-" : baseline.adjR2.toFixed(4)}, SSE=${baseline.sse.toFixed(4)}. Current minus baseline: ΔR²=${deltaR2 === null ? "-" : deltaR2.toFixed(4)}, Δadj R²=${deltaAdjR2 === null ? "-" : deltaAdjR2.toFixed(4)}, ΔSSE=${deltaSse.toFixed(4)}.`;
    }

    function formatPValue(value) {
      if (value === null || !Number.isFinite(value)) return "-";
      if (value < 0.0001) return value.toExponential(2);
      return value.toFixed(4);
    }

    function standardizePredictorMatrix(rows, predictorFeatures) {
      const means = [];
      const stds = [];
      for (const feature of predictorFeatures) {
        const vals = rows.map((row) => parseMaybeNumber(row[feature]));
        if (vals.some((value) => value === null)) return null;
        const m = mean(vals);
        const sd = sampleStd(vals);
        means.push(m);
        stds.push(sd > 1e-12 ? sd : 1);
      }
      const matrix = rows.map((row) =>
        predictorFeatures.map((feature, idx) => {
          const value = parseMaybeNumber(row[feature]);
          return (value - means[idx]) / stds[idx];
        })
      );
      return { matrix, means, stds };
    }

    function covarianceMatrixFromStandardized(matrix) {
      const n = matrix.length;
      const p = matrix[0]?.length || 0;
      const cov = Array.from({ length: p }, () => Array(p).fill(0));
      if (n < 2 || p < 1) return cov;
      for (const row of matrix) {
        for (let i = 0; i < p; i++) {
          for (let j = i; j < p; j++) {
            cov[i][j] += row[i] * row[j];
          }
        }
      }
      for (let i = 0; i < p; i++) {
        for (let j = i; j < p; j++) {
          cov[i][j] /= (n - 1);
          cov[j][i] = cov[i][j];
        }
      }
      return cov;
    }

    function vectorNorm(vector) {
      return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    }

    function matrixVectorProduct(matrix, vector) {
      return matrix.map((row) => row.reduce((sum, value, idx) => sum + value * vector[idx], 0));
    }

    function dot(a, b) {
      return a.reduce((sum, value, idx) => sum + value * b[idx], 0);
    }

    function topEigenComponentsSymmetric(matrix, maxComponents) {
      const p = matrix.length;
      if (!p) return [];
      let work = matrix.map((row) => row.slice());
      const components = [];
      for (let comp = 0; comp < Math.min(maxComponents, p); comp++) {
        let v = Array.from({ length: p }, (_, i) => (i === comp % p ? 1 : 0.25));
        let norm = vectorNorm(v);
        v = v.map((x) => x / norm);
        for (let iter = 0; iter < 200; iter++) {
          let next = matrixVectorProduct(work, v);
          norm = vectorNorm(next);
          if (norm < 1e-10) break;
          next = next.map((x) => x / norm);
          const delta = vectorNorm(next.map((x, i) => x - v[i]));
          v = next;
          if (delta < 1e-8) break;
        }
        const mv = matrixVectorProduct(work, v);
        const eigenvalue = dot(v, mv);
        if (!Number.isFinite(eigenvalue) || eigenvalue <= 1e-8) break;
        components.push({ eigenvalue, loadings: v.slice() });
        for (let i = 0; i < p; i++) {
          for (let j = 0; j < p; j++) {
            work[i][j] -= eigenvalue * v[i] * v[j];
          }
        }
      }
      return components;
    }

    function runComponentRegression(rows, responseFeature, predictorFeatures) {
      if (rows.length < 5 || predictorFeatures.length < 2) return null;
      const standardized = standardizePredictorMatrix(rows, predictorFeatures);
      if (!standardized) return null;
      const cov = covarianceMatrixFromStandardized(standardized.matrix);
      const maxComponents = Math.min(8, predictorFeatures.length, rows.length - 2);
      const components = topEigenComponentsSymmetric(cov, maxComponents);
      if (!components.length) return null;
      const totalVariance = predictorFeatures.length;
      const componentRows = rows.map((row, rowIdx) => {
        const out = { response: parseMaybeNumber(row[responseFeature]) };
        for (let i = 0; i < components.length; i++) {
          out[`PC${i + 1}`] = dot(standardized.matrix[rowIdx], components[i].loadings);
        }
        return out;
      });
      const componentFeatures = components.map((_, i) => `PC${i + 1}`);
      const full = fitOls(componentRows, "response", componentFeatures);
      if (!full || full.dfError <= 0) return null;
      const mseFull = full.sse / full.dfError;
      const rowsOut = componentFeatures.map((feature, idx) => {
        const reducedFeatures = componentFeatures.filter((f) => f !== feature);
        const reduced = fitOls(componentRows, "response", reducedFeatures);
        const partialSS = reduced ? Math.max(0, reduced.sse - full.sse) : null;
        const fStat = partialSS === null || mseFull <= 0 ? null : partialSS / mseFull;
        const pValue = fStat === null ? null : fSurvival(fStat, 1, full.dfError);
        const loadings = components[idx].loadings
          .map((loading, featureIdx) => ({ feature: predictorFeatures[featureIdx], loading }))
          .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading));
        return {
          component: feature,
          coefficient: full.beta[idx + 1],
          eigenvalue: components[idx].eigenvalue,
          predictorVariancePct: totalVariance > 0 ? (100 * components[idx].eigenvalue) / totalVariance : null,
          partialSS,
          responseModelSSPct: full.ssr > 0 && partialSS !== null ? (100 * partialSS) / full.ssr : null,
          fStat,
          pValue,
          topLoadings: loadings.slice(0, 6),
        };
      }).sort((a, b) => (b.partialSS || 0) - (a.partialSS || 0));
      return { full, rows: rowsOut, componentCount: components.length };
    }

    function renderComponentAnalysis(selection, responseFeature) {
      const result = runComponentRegression(selection.rows, responseFeature, selection.predictors);
      if (!result) {
        componentSummaryEl.textContent = "Component analysis not available: need at least two usable predictors and enough response-available rows.";
        componentHead.innerHTML = "";
        componentBody.innerHTML = "";
        safeNewPlot("componentPlot", [], { title: "No component model" }, { responsive: true });
        return;
      }
      componentSummaryEl.textContent =
        `Component model: ${result.componentCount} predictor components | rows=${result.full.n} | R²=${result.full.r2 === null ? "-" : result.full.r2.toFixed(4)} | adjusted R²=${result.full.adjR2 === null ? "-" : result.full.adjR2.toFixed(4)}. Components are PCA aggregates of standardized predictors.`;
      const cols = ["component", "coefficient", "predictor_variance_%", "response_SS_%", "F", "p_value", "top feature loadings"];
      componentHead.innerHTML = cols.map((c) => `<th>${c}</th>`).join("");
      componentBody.innerHTML = result.rows.map((row) => `
        <tr>
          <td>${row.component}</td>
          <td>${Number.isFinite(row.coefficient) ? row.coefficient.toFixed(6) : "-"}</td>
          <td>${row.predictorVariancePct === null ? "-" : row.predictorVariancePct.toFixed(2)}</td>
          <td>${row.responseModelSSPct === null ? "-" : row.responseModelSSPct.toFixed(2)}</td>
          <td>${row.fStat === null ? "-" : row.fStat.toFixed(4)}</td>
          <td>${formatPValue(row.pValue)}</td>
          <td>${row.topLoadings.map((item) => `${item.feature} (${item.loading.toFixed(3)})`).join("<br>")}</td>
        </tr>
      `).join("");

      const plotRows = result.rows.slice().reverse();
      safeNewPlot("componentPlot", [
        {
          type: "bar",
          orientation: "h",
          name: "Response contribution",
          y: plotRows.map((row) => row.component),
          x: plotRows.map((row) => row.responseModelSSPct || 0),
          text: plotRows.map((row) => `${row.component}<br>response SS%=${row.responseModelSSPct === null ? "-" : row.responseModelSSPct.toFixed(2)}<br>top loading: ${row.topLoadings[0]?.feature || "-"}`),
          hovertemplate: "%{text}<extra></extra>",
          marker: { color: "rgba(180,83,9,0.72)" },
        },
        {
          type: "bar",
          orientation: "h",
          name: "Predictor variance",
          y: plotRows.map((row) => row.component),
          x: plotRows.map((row) => row.predictorVariancePct || 0),
          text: plotRows.map((row) => `${row.component}<br>predictor variance%=${row.predictorVariancePct === null ? "-" : row.predictorVariancePct.toFixed(2)}`),
          hovertemplate: "%{text}<extra></extra>",
          marker: { color: "rgba(8,145,178,0.45)" },
        }
      ], {
        title: "Component aggregates: response contribution vs predictor variance",
        barmode: "group",
        xaxis: { title: "%" },
        yaxis: { automargin: true },
        margin: { t: 48, r: 20, b: 56, l: 90 },
      }, { responsive: true });
    }

    function renderMlrAnalysis() {
      lastMlrExportRows = [];
      const responseFeature = mlrResponseFeature.value;
      if (!responseFeature) {
        mlrStatusEl.textContent = "Select a response variable.";
        return;
      }
      const allCandidates = numericFeatures.filter((feature) => feature !== responseFeature);
      const candidates = allCandidates.filter((feature) => mlrIncludedFeatures.has(feature));
      if (candidates.length < 1) {
        mlrStatusEl.textContent = "No independent variables are selected. Check at least one predictor feature.";
        return;
      }
      const analysisRows = plottedRows.filter((_, i) => rowVisibility[i]);
      if (analysisRows.length < 4) {
        mlrStatusEl.textContent = "Not enough selected rows to fit a multivariable model. Use the plotted rows panel to include more rows.";
        return;
      }
      const selection = selectMlrPredictors(analysisRows, responseFeature, candidates);
      const baselineSelection = selectMlrPredictors(analysisRows, responseFeature, allCandidates);
      const baselineFull = baselineSelection.predictors.length >= 1 && baselineSelection.rows.length >= 4
        ? fitOls(baselineSelection.rows, responseFeature, baselineSelection.predictors)
        : null;
      if (selection.predictors.length < 1 || selection.rows.length < 4) {
        mlrStatusEl.textContent = "Not enough response-available rows or usable predictors to fit a multivariable model.";
        mlrSummaryEl.textContent = "Model: not available";
        mlrHead.innerHTML = "";
        mlrBody.innerHTML = "";
        safeNewPlot("mlrPlot", [], { title: "No multivariable model" }, { responsive: true });
        componentSummaryEl.textContent = "Component analysis not available.";
        componentHead.innerHTML = "";
        componentBody.innerHTML = "";
        safeNewPlot("componentPlot", [], { title: "No component model" }, { responsive: true });
        return;
      }
      const full = fitOls(selection.rows, responseFeature, selection.predictors);
      if (!full || full.dfError <= 0) {
        mlrStatusEl.textContent = "Model could not be fit; predictors may be collinear or too numerous for the response-available rows.";
        componentSummaryEl.textContent = "Component analysis not available because the main model could not be fit.";
        componentHead.innerHTML = "";
        componentBody.innerHTML = "";
        safeNewPlot("componentPlot", [], { title: "No component model" }, { responsive: true });
        return;
      }
      const mseFull = full.sse / full.dfError;
      const rows = selection.predictors.map((feature, idx) => {
        const reducedPredictors = selection.predictors.filter((f) => f !== feature);
        const reduced = fitOls(selection.rows, responseFeature, reducedPredictors);
        const partialSS = reduced ? Math.max(0, reduced.sse - full.sse) : null;
        const fStat = partialSS === null || mseFull <= 0 ? null : partialSS / mseFull;
        const pValue = fStat === null ? null : fSurvival(fStat, 1, full.dfError);
        return {
          feature,
          coefficient: full.beta[idx + 1],
          partialSS,
          percentModelSS: full.ssr > 0 && partialSS !== null ? (100 * partialSS) / full.ssr : null,
          fStat,
          pValue,
        };
      }).sort((a, b) => (b.partialSS || 0) - (a.partialSS || 0));
      lastMlrExportRows = rows.map((row, idx) => ({
        visual_component: "multilinear_regression_anova",
        contribution_rank: idx + 1,
        response_feature: responseFeature,
        predictor_feature: row.feature,
        coefficient: row.coefficient,
        partial_SS: row.partialSS,
        model_SS_percent: row.percentModelSS,
        F: row.fStat,
        p_value: row.pValue,
        model_r2: full.r2,
        model_adjusted_r2: full.adjR2,
        model_sse: full.sse,
        model_ssr: full.ssr,
        df_model: full.dfModel,
        df_error: full.dfError,
        rows_used: selection.rows.length,
        predictors_used: selection.predictors.length,
        candidate_predictors_checked: candidates.length,
        all_candidate_predictors: allCandidates.length,
        selection_mode: selection.mode,
        dropped_unused_predictors: selection.droppedCount,
        rejected_for_fit: selection.rejectedForFit || 0,
        complete_case_rows_used: selection.completeCaseMeta?.usedRowCount || selection.rows.length,
        rows_skipped_missing_selected_values: selection.completeCaseMeta?.droppedRows?.length || 0,
        baseline_r2: baselineFull ? baselineFull.r2 : null,
        baseline_adjusted_r2: baselineFull ? baselineFull.adjR2 : null,
        baseline_sse: baselineFull ? baselineFull.sse : null,
      }));

      mlrStatusEl.textContent =
        `Response: ${responseFeature} | Mode: ${selection.mode} | Candidate predictors checked: ${candidates.length} / ${allCandidates.length} | Complete observed rows used: ${selection.rows.length} | Predictors used: ${selection.predictors.length} | Dropped/unused predictors: ${selection.droppedCount} | Rejected for collinearity/row limits: ${selection.rejectedForFit || 0}. ${summarizeCompleteCase(selection.completeCaseMeta)}`;
      mlrSummaryEl.textContent =
        `Model: R²=${full.r2 === null ? "-" : full.r2.toFixed(4)} | adjusted R²=${full.adjR2 === null ? "-" : full.adjR2.toFixed(4)} | df_model=${full.dfModel} | df_error=${full.dfError} | SSE=${full.sse.toFixed(4)} | SSR=${full.ssr.toFixed(4)}. ${compareModelFits(full, baselineFull)}`;

      const cols = ["feature", "coefficient", "partial_SS", "model_SS_%", "F", "p_value"];
      mlrHead.innerHTML = cols.map((c) => `<th>${c}</th>`).join("");
      mlrBody.innerHTML = rows.map((row) => `
        <tr data-mlr-contribution-feature="${htmlEscape(row.feature)}">
          <td><button type="button" class="feature-link" data-mlr-jump-feature="${htmlEscape(row.feature)}">${htmlEscape(row.feature)}</button></td>
          <td>${Number.isFinite(row.coefficient) ? row.coefficient.toFixed(6) : "-"}</td>
          <td>${row.partialSS === null ? "-" : row.partialSS.toFixed(6)}</td>
          <td>${row.percentModelSS === null ? "-" : row.percentModelSS.toFixed(2)}</td>
          <td>${row.fStat === null ? "-" : row.fStat.toFixed(4)}</td>
          <td>${formatPValue(row.pValue)}</td>
        </tr>
      `).join("");
      mlrBody.querySelectorAll("[data-mlr-jump-feature]").forEach((el) => {
        el.addEventListener("click", () => focusMlrFeature(el.getAttribute("data-mlr-jump-feature")));
      });

      const top = rows.slice(0, 20).reverse();
      safeNewPlot("mlrPlot", [{
        type: "bar",
        orientation: "h",
        y: top.map((row) => featureShortLabel(row.feature, featureCategory(row.feature))),
        x: top.map((row) => row.percentModelSS || 0),
        customdata: top.map((row) => row.feature),
        text: top.map((row) => `${row.feature}<br>partial SS=${row.partialSS === null ? "-" : row.partialSS.toFixed(4)}<br>F=${row.fStat === null ? "-" : row.fStat.toFixed(3)}<br>p=${formatPValue(row.pValue)}`),
        hovertemplate: "%{text}<extra></extra>",
        marker: { color: "rgba(14,116,144,0.75)" },
      }], {
        title: "Top feature contributions by partial sum of squares",
        xaxis: { title: "% of model regression SS" },
        yaxis: { automargin: true },
        margin: { t: 48, r: 20, b: 56, l: 220 },
      }, { responsive: true });
      const mlrPlotEl = document.getElementById("mlrPlot");
      if (mlrPlotEl && typeof mlrPlotEl.on === "function") {
        if (mlrPlotClickHandler && typeof mlrPlotEl.removeListener === "function") {
          mlrPlotEl.removeListener("plotly_click", mlrPlotClickHandler);
        }
        mlrPlotClickHandler = (eventData) => {
          const feature = eventData?.points?.[0]?.customdata;
          if (feature) focusMlrFeature(feature);
        };
        mlrPlotEl.on("plotly_click", mlrPlotClickHandler);
      }
      renderComponentAnalysis(selection, responseFeature);
    }

    function scheduleDeferredContextPlots() {
      if (deferredContextPlotTimer) window.clearTimeout(deferredContextPlotTimer);
      cohortStatsEl.textContent = "Cohort plot queued...";
      distStatsEl.textContent = "Feature distribution queued...";
      deferredContextPlotTimer = window.setTimeout(() => {
        deferredContextPlotTimer = null;
        rerenderCohortHistogram();
        window.setTimeout(() => {
          rerenderFeatureDistribution();
        }, 80);
      }, 120);
    }

    function refreshAnalysis() {
      const xKey = xFeature.value;
      const yKey = yFeature.value;

      if (!xKey || !yKey) {
        setStatus("No plottable numeric features found.");
        return;
      }
      if (xKey === yKey) {
        setStatus("Choose different features for X and Y axis.");
        return;
      }

      if (!allRows.length) {
        plottedRows = [];
        setStatus("No rows loaded.");
        r2El.textContent = "R²: not available";
        regressionEl.textContent = "Regression: not available";
        updateRowStats([]);
        renderDroppedRows([], 0);
        safeNewPlot("plot", [], { title: "No data" }, { responsive: true });
        tsneStatusEl.textContent = "No data for t-SNE.";
        rowManualVisibility = [];
        rowFilterVisibility = [];
        rowVisibility = [];
        updateRowFilterStatus();
        renderTable([], xKey, yKey);
        return;
      }

      const rowUniverseChanged = plottedRows.length !== allRows.length || plottedRows.some((row, idx) => row !== allRows[idx]);
      plottedRows = allRows.slice();
      if (rowUniverseChanged || rowManualVisibility.length !== plottedRows.length) {
        rowManualVisibility = plottedRows.map(() => true);
      }
      currentXKey = xKey;
      currentYKey = yKey;
      recomputeRowVisibility(false);
      const selectedRows = plottedRows.filter((_, i) => rowVisibility[i]);
      const plotReadyCount = selectedRows.filter((row) => parseMaybeNumber(row[xKey]) !== null && parseMaybeNumber(row[yKey]) !== null).length;
      setStatus(`Selected row universe: ${selectedRows.length} / ${plottedRows.length}. Scatter has ${plotReadyCount} rows with both selected axis values.`);
      renderTable(plottedRows, currentXKey, currentYKey);
      rerenderFromVisibility();
    }

    function rerenderFromVisibility() {
      const visibleRows = plottedRows.filter((_, i) => rowVisibility[i]);
      updateRowStats(visibleRows);
      if (visibleRows.length === 0) {
        r2El.textContent = "R²: not available";
        regressionEl.textContent = "Regression: not available";
        setStatus("No visible points. Check at least one row.");
        renderDroppedRows([], 0);
        safeNewPlot("plot", [], { title: "No data" }, { responsive: true });
        tsneStatusEl.textContent = "No visible rows for t-SNE.";
        return;
      }

      const plotReadyCount = visibleRows.filter((row) => parseMaybeNumber(row[currentXKey]) !== null && parseMaybeNumber(row[currentYKey]) !== null).length;
      const r2 = computeR2(visibleRows, currentXKey, currentYKey);
      r2El.textContent = r2 === null ? "R²: not available" : `R²: ${r2.toFixed(4)}`;
      const reg = fitLinearRegression(visibleRows, currentXKey, currentYKey);
      if (reg === null) {
        regressionEl.textContent = "Regression: not available";
      } else {
        const sign = reg.intercept >= 0 ? "+" : "-";
        regressionEl.textContent = `Regression: y = ${reg.slope.toFixed(6)}x ${sign} ${Math.abs(reg.intercept).toFixed(6)}`;
      }
      setStatus(`Selected rows: ${visibleRows.length} / ${plottedRows.length}. Scatter plot-ready rows: ${plotReadyCount}.`);
      renderPlot(plottedRows, currentXKey, currentYKey);
      tsneStatusEl.textContent = "t-SNE not run automatically. Click Run t-SNE to compute.";
      scheduleDeferredContextPlots();
    }

    function highlightRow(rowIdx) {
      rowsBody.querySelectorAll("tr").forEach((tr) => tr.classList.remove("selected-row"));
      const rowEl = rowsBody.querySelector(`tr[data-row-index="${rowIdx}"]`);
      if (!rowEl) return;
      rowEl.classList.add("selected-row");
      rowEl.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function updateRowStats(rows) {
      const uniqueSubjects = new Set();
      const uniqueCases = new Set();
      for (const row of rows) {
        if (row.subject_id !== undefined && row.subject_id !== null && String(row.subject_id).trim() !== "") {
          uniqueSubjects.add(String(row.subject_id));
        } else if (row.case_path) {
          const sid = extractSubjectIdFromCasePath(row.case_path);
          if (sid) uniqueSubjects.add(sid);
        }
        if (row.case_path) uniqueCases.add(String(row.case_path));
      }
      rowStatsEl.textContent =
        `Rows: ${rows.length} | Unique subjects: ${uniqueSubjects.size} | Unique subject-date cases: ${uniqueCases.size}`;
    }

    async function rerenderTsne() {
      lastTsneExportRows = [];
      const visibleRows = plottedRows.filter((_, i) => rowVisibility[i]);
      const selected = Array.from(tsneSelectedFeatures).filter((f) => numericFeatures.includes(f));
      dropRowsHead.innerHTML = "";
      dropRowsBody.innerHTML = "";
      tsneDropSummaryEl.textContent = "";
      if (visibleRows.length < 3) {
        tsneStatusEl.textContent = "t-SNE needs at least 3 visible rows.";
        safeNewPlot("tsnePlot", [], { title: "t-SNE: not enough rows" }, { responsive: true });
        return;
      }
      if (selected.length < 1) {
        tsneStatusEl.textContent = "Select at least one feature for t-SNE.";
        safeNewPlot("tsnePlot", [], { title: "t-SNE: no features selected" }, { responsive: true });
        return;
      }

      const visibleSourceIndices = [];
      for (let i = 0; i < plottedRows.length; i++) {
        if (rowVisibility[i]) visibleSourceIndices.push(i);
      }
      const complete = buildCompleteCaseNumericMatrix(visibleRows, selected, {
        sourceIndices: visibleSourceIndices,
        standardize: true,
      });
      const dataRows = complete.matrix;

      renderDroppedRows(complete.droppedRows, visibleRows.length, complete);

      if (dataRows.length < 3 || complete.features.length < 1) {
        tsneStatusEl.textContent = "Not enough complete observed rows for selected t-SNE features. Narrow feature selection or include more rows.";
        safeNewPlot("tsnePlot", [], { title: "t-SNE: insufficient usable data" }, { responsive: true });
        return;
      }

      const perplexity = Math.max(5, Math.min(30, Math.floor(dataRows.length / 3)));
      let Y = [];
      let embeddingMethod = "t-SNE";
      try {
        const resp = await fetch("/api/tsne", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ matrix: dataRows, perplexity: perplexity }),
        });
        const payload = await resp.json();
        if (!resp.ok || !payload.ok) {
          throw new Error(payload.error || `HTTP ${resp.status}`);
        }
        Y = payload.embedding || [];
        embeddingMethod = payload.method || embeddingMethod;
      } catch (err) {
        tsneStatusEl.textContent = `t-SNE failed to compute: ${err.message || err}`;
        safeNewPlot("tsnePlot", [], { title: "t-SNE computation failed" }, { responsive: true });
        return;
      }

      lastTsneRender = {
        Y,
        rowIndices: complete.rowIndices,
        dataRowsLength: dataRows.length,
        selectedCount: selected.length,
        selectedFeatures: complete.features,
        originalSelectedFeatures: selected,
        perplexity,
        dataRows,
        completeCaseMeta: complete,
        embeddingMethod,
      };
      renderTsnePlotFromEmbedding(lastTsneRender);
      tsneStatusEl.textContent = `${embeddingMethod} computed on ${dataRows.length} complete observed rows using ${complete.features.length} usable features from ${selected.length} selected features. ${summarizeCompleteCase(complete)}`;
    }

    function renderDroppedRows(droppedRows, totalVisibleRows, completeCaseMeta = null) {
      const kept = Math.max(0, totalVisibleRows - droppedRows.length);
      tsneDropSummaryEl.textContent = completeCaseMeta
        ? `t-SNE row check: visible=${totalVisibleRows}, used=${completeCaseMeta.usedRowCount || kept}, dropped=${droppedRows.length}. ${summarizeCompleteCase(completeCaseMeta)}`
        : `t-SNE row check: visible=${totalVisibleRows}, used=${kept}, dropped=${droppedRows.length}.`;
      const cols = ["row_index", "case_path", "instance_id", "pair_name", "category", "panel", "missing_features"];
      dropRowsHead.innerHTML = cols.map((c) => `<th>${c}</th>`).join("");
      if (droppedRows.length === 0) {
        dropRowsBody.innerHTML = `<tr><td colspan="${cols.length}">No dropped rows for current t-SNE feature selection.</td></tr>`;
        return;
      }
      dropRowsBody.innerHTML = droppedRows.map((r) =>
        "<tr>" + cols.map((c) => `<td>${r[c] ?? ""}</td>`).join("") + "</tr>"
      ).join("");
    }

    function ingestPayload(payload, sourceLabel) {
      setStatus("Extracting rows from annotation JSON...");
      const caseLabFeaturesMap = buildCaseLabFeaturesMap(payload);
      subjectLabMap = buildSubjectLabMap(payload);
      allRows = extractRowsFromPayload(payload, caseLabFeaturesMap);
      setStatus("Finding numeric features...");
      numericFeatures = getNumericFeatures(allRows);
      exportFeatures = getExportableFeatures(allRows);
      rowFilterFeatures = getFilterableFeatures(allRows);
      rowFilterRules = [];
      rowSelPresenceMode = "all";
      if (rowSelPresenceFilter) rowSelPresenceFilter.value = "all";
      cohortLabFeatures = Array.from(
        new Set(
          Object.values(subjectLabMap)
            .flatMap((m) => Object.keys(m || {}))
            .filter((k) => k.startsWith("lab_features."))
        )
      ).sort();
      const mergedDistFeatures = Array.from(new Set([...numericFeatures, ...cohortLabFeatures])).sort();
      setStatus("Populating feature selectors...");
      populateDistFeatureSelector(mergedDistFeatures);
      populateCohortFeatureSelector(cohortLabFeatures);
      populateRowFilterFeatureSelector();
      renderRowFilterRules();

      summaryEl.textContent = `${sourceLabel || "JSON"} | Rows extracted: ${allRows.length}. Plot-ready numeric features: ${numericFeatures.length}. Exportable feature columns: ${exportFeatures.length}.`;
      lastCorrelationExportRows = [];
      corrStatusEl.textContent = "Click Run Correlation Filter to screen numeric feature pairs.";
      corrHead.innerHTML = "";
      corrBody.innerHTML = "";

      if (allRows.length === 0) {
        setStatus("No rows extracted from JSON. Expected peak_pair_features.cases[*].paired_features[*].");
        r2El.textContent = "R²: not available";
        renderTable([], "x", "y");
        xFeature.innerHTML = "";
        yFeature.innerHTML = "";
        exportSelectedFeatures = new Set();
        renderExportFeatureSelector([]);
        corrSelectedFeatures = new Set();
        renderCorrelationFeatureSelector([]);
        return;
      }

      if (numericFeatures.length < 2) {
        setStatus("Not enough numeric features to plot. Need at least 2 numeric columns.");
        r2El.textContent = "R²: not available";
        renderTable([], "x", "y");
        xFeature.innerHTML = "";
        yFeature.innerHTML = "";
        exportSelectedFeatures = new Set(exportFeatures);
        renderExportFeatureSelector(exportFeatures);
        corrSelectedFeatures = new Set(numericFeatures);
        renderCorrelationFeatureSelector(numericFeatures);
        return;
      }

      populateFeatureSelectors(numericFeatures);
      plottedRows = allRows.slice();
      rowVisibility = plottedRows.map(() => true);
      rowManualVisibility = plottedRows.map(() => true);
      rowFilterVisibility = plottedRows.map(() => true);
      currentXKey = xFeature.value;
      currentYKey = yFeature.value;
      renderTable(plottedRows, currentXKey, currentYKey);
      updateRowStats(plottedRows);
      refreshAnalysis();
    }

    async function loadCurrentServerAnnotations(silent = false) {
      if (loadCurrentStateBtn) {
        loadCurrentStateBtn.disabled = true;
        loadCurrentStateBtn.textContent = "Loading...";
      }
      try {
        if (!silent) setStatus("Loading current alignment annotations...");
        const resp = await fetch("/api/analysis/current");
        const payload = await resp.json().catch(() => ({}));
        if (!resp.ok || payload.error) throw new Error(payload.error || `HTTP ${resp.status}`);
        ingestPayload(payload, payload?.meta?.source || "Current alignment annotations");
      } catch (err) {
        setStatus("Failed to load current annotations: " + (err.message || err));
        summaryEl.textContent = "";
      } finally {
        if (loadCurrentStateBtn) {
          loadCurrentStateBtn.disabled = false;
          loadCurrentStateBtn.textContent = "Load Current Annotations";
        }
      }
    }

    async function onFileUpload(file) {
      if (!file) return;
      try {
        setStatus("Reading JSON...");
        const text = await file.text();
        ingestPayload(JSON.parse(text), file.name || "Uploaded JSON");
      } catch (err) {
        setStatus("Failed to parse JSON: " + err.message);
        summaryEl.textContent = "";
      }
    }

    fileInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0] ? e.target.files[0] : null;
      onFileUpload(file);
    });
    rowFilterFeature.addEventListener("change", renderRowFilterValueControl);
    rowSelPresenceFilter.addEventListener("change", () => {
      rowSelPresenceMode = rowSelPresenceFilter.value || "all";
      recomputeRowVisibility(true);
    });
    addRowFilterRuleBtn.addEventListener("click", addRowFilterRule);
    clearRowFilterRulesBtn.addEventListener("click", () => {
      rowFilterRules = [];
      rowSelPresenceMode = "all";
      rowSelPresenceFilter.value = "all";
      renderRowFilterRules();
      recomputeRowVisibility(true);
    });
    loadCurrentStateBtn.addEventListener("click", () => loadCurrentServerAnnotations(false));
    if (window.location.pathname === "/analysis") {
      loadCurrentServerAnnotations(true);
    }
    exportScatterCsvBtn.addEventListener("click", () => {
      const csv = buildRowsCsv(lastScatterExportRows, [
        "visual_component", "row_index", "case_path", "instance_id", "pair_name", "category", "panel", "subject_id",
        "x_feature", "y_feature", "x_value", "y_value", "regression_slope", "regression_intercept",
        "regression_predicted_y", "r2"
      ]);
      if (!downloadCsv(csv, "analysis_scatter_plot_data")) {
        setStatus("No scatter plot data available to export. Load annotations and render the scatter plot first.");
      }
    });
    swapScatterAxesBtn.addEventListener("click", swapScatterAxes);
    runCorrelationFilterBtn.addEventListener("click", renderCorrelationFilterPanel);
    corrThresholdInput.addEventListener("change", renderCorrelationFilterPanel);
    corrR2ThresholdInput.addEventListener("change", renderCorrelationFilterPanel);
    corrUseAbsInput.addEventListener("change", renderCorrelationFilterPanel);
    corrSelectAllBtn.addEventListener("click", () => {
      corrSelectedFeatures = new Set(numericFeatures);
      renderCorrelationFeatureSelector(numericFeatures);
      corrStatusEl.textContent = `Selected all ${numericFeatures.length} correlation features. Click Run Correlation Filter to rescreen.`;
    });
    corrClearAllBtn.addEventListener("click", () => {
      corrSelectedFeatures = new Set();
      renderCorrelationFeatureSelector(numericFeatures);
      corrStatusEl.textContent = "Cleared all correlation features. Select at least two features before screening.";
    });
    downloadCorrFeatureSelectionBtn.addEventListener("click", downloadCorrelationFeatureSelection);
    reloadCorrFeatureSelectionBtn.addEventListener("click", () => {
      corrFeatureSelectionFile.click();
    });
    corrFeatureSelectionFile.addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
      if (!file) return;
      try {
        const text = await file.text();
        applyCorrelationFeatureSelectionPayload(JSON.parse(text), file.name || "JSON");
      } catch (err) {
        corrStatusEl.textContent = "Failed to reload correlation feature selection: " + (err.message || err);
      } finally {
        corrFeatureSelectionFile.value = "";
      }
    });
    exportCorrelationCsvBtn.addEventListener("click", () => {
      const selectedRows = selectedCorrelationExportRows();
      const csv = buildRowsCsv(selectedRows, [
        "feature_a", "feature_b", "correlation_r", "abs_correlation", "r_squared", "complete_rows",
        "visible_rows", "screened_feature_count", "screened_features", "threshold", "r_squared_threshold", "threshold_mode"
      ]);
      if (!downloadCsv(csv, "analysis_feature_correlations")) {
        corrStatusEl.textContent = lastCorrelationExportRows.length
          ? "No correlation rows are checked for export."
          : "No correlation results available to export. Run Correlation Filter first.";
      }
    });
    runTsneBtn.addEventListener("click", () => {
      rerenderTsne();
    });
    exportTsneCsvBtn.addEventListener("click", () => {
      const csv = buildRowsCsv(lastTsneExportRows, [
        "visual_component", "row_index", "case_path", "instance_id", "pair_name", "category", "panel", "subject_id",
        "tsne_1", "tsne_2", "color_feature", "color_value", "color_label", "color_key", "color_hex",
        "perplexity", "selected_feature_count", "selected_features"
      ]);
      if (!downloadCsv(csv, "analysis_tsne_data")) {
        tsneStatusEl.textContent = "No t-SNE data available to export. Run t-SNE first.";
      }
    });
    runTsneSvmBtn.addEventListener("click", () => {
      renderClusterSeparationAnalysis();
    });
    classifierSelectAllBtn.addEventListener("click", () => {
      classifierSelectedFeatures = new Set(numericFeatures);
      renderClassifierFeatureSelector(numericFeatures);
      classifierStatusEl.textContent = `Selected all ${numericFeatures.length} classifier features.`;
    });
    classifierClearAllBtn.addEventListener("click", () => {
      classifierSelectedFeatures = new Set();
      renderClassifierFeatureSelector(numericFeatures);
      classifierStatusEl.textContent = "Cleared all classifier features. Select at least one feature before running classification.";
    });
    downloadClassifierFeatureSelectionBtn.addEventListener("click", downloadClassifierFeatureSelection);
    reloadClassifierFeatureSelectionBtn.addEventListener("click", () => {
      classifierFeatureSelectionFile.click();
    });
    bindFeatureSelectionFileInput(
      classifierFeatureSelectionFile,
      (payload, sourceLabel) => applyFeatureSelectionPayload(
        payload,
        numericFeatures,
        (selection) => { classifierSelectedFeatures = selection; },
        () => renderClassifierFeatureSelector(numericFeatures),
        classifierStatusEl,
        sourceLabel,
        "classifier"
      ),
      classifierStatusEl,
      "classifier"
    );
    addClassifierClassRuleBtn.addEventListener("click", () => {
      const palette = ["#0f766e", "#b45309", "#be123c", "#2563eb", "#7c3aed", "#15803d"];
      const idx = classifierClassRules.length + 1;
      classifierClassRules.push({
        id: `classifier_rule_${Date.now()}_${idx}`,
        enabled: true,
        label: `Class ${idx}`,
        lowerOperator: "",
        lowerValue: "",
        upperOperator: "",
        upperValue: "",
        color: palette[(idx - 1) % palette.length],
      });
      renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
    });
    runClassifierBtn.addEventListener("click", () => {
      renderClassificationPanel();
    });
    exportClassifierCsvBtn.addEventListener("click", () => {
      const csv = buildRowsCsv(lastClassifierExportRows, [
        "section", "model_type", "model_label", "seed", "class_feature", "feature_space",
        "evaluation_unit", "evaluation_label", "class_count", "classes", "labeled_rows", "visible_rows",
        "evaluated_units", "subject_count", "predictor_count", "predictors_used",
        "training_accuracy", "row_training_accuracy", "loo_accuracy", "loo_correct", "loo_total",
        "macro_auroc", "macro_sensitivity", "macro_specificity", "macro_precision", "macro_f1",
        "complete_case_rows_used", "rows_skipped_missing_selected_values",
        "class_label", "auroc", "sensitivity", "specificity", "precision", "f1",
        "tp", "fp", "tn", "fn", "subject_id", "case_count", "actual", "predicted", "correct",
        "truth_vote_counts", "prediction_vote_counts", "mean_probabilities", "count",
        "row_normalized_percent", "point_index", "threshold", "fpr", "tpr",
        "predictor", "rank", "mean_abs_contribution", "mean_signed_contribution", "details",
        "enabled", "rule", "lower_operator", "lower_value", "upper_operator", "upper_value", "color"
      ]);
      if (!downloadCsv(csv, "analysis_classification_performance")) {
        classifierStatusEl.textContent = "No classification performance data available to export. Run Classification first.";
      }
    });
    classifierClassFeature.addEventListener("change", () => {
      renderClassifierClassRuleTable(buildClassifierClassGroups(plottedRows.filter((_, i) => rowVisibility[i]), classifierClassFeature.value || "").groups);
      classifierStatusEl.textContent = "Classification class feature changed. Review class rules and click Run Classification.";
    });
    classifierEvaluationUnit.addEventListener("change", () => {
      classifierStatusEl.textContent = "Classification evaluation unit changed. Click Run Classification to recompute metrics and plots.";
    });
    tsneSelectAllBtn.addEventListener("click", () => {
      tsneSelectedFeatures = new Set(numericFeatures);
      renderTsneFeatureSelector(numericFeatures);
      tsneStatusEl.textContent = `Selected all ${numericFeatures.length} t-SNE features. Click Run t-SNE to recompute.`;
    });
    tsneClearAllBtn.addEventListener("click", () => {
      tsneSelectedFeatures = new Set();
      renderTsneFeatureSelector(numericFeatures);
      tsneStatusEl.textContent = "Cleared all t-SNE features. Select at least one feature before running t-SNE.";
    });
    downloadTsneFeatureSelectionBtn.addEventListener("click", downloadTsneFeatureSelection);
    reloadTsneFeatureSelectionBtn.addEventListener("click", () => {
      tsneFeatureSelectionFile.click();
    });
    bindFeatureSelectionFileInput(
      tsneFeatureSelectionFile,
      (payload, sourceLabel) => applyFeatureSelectionPayload(
        payload,
        numericFeatures,
        (selection) => { tsneSelectedFeatures = selection; },
        () => renderTsneFeatureSelector(numericFeatures),
        tsneStatusEl,
        sourceLabel,
        "t-SNE"
      ),
      tsneStatusEl,
      "t-SNE"
    );
    runPcaBtn.addEventListener("click", renderPcaPanel);
    refreshAllCasesPcaBtn.addEventListener("click", () => {
      const hadPca = !!lastPcaRender;
      refreshRowsToAllCases("All cases restored for PCA. Complete-case PCA may still skip rows missing selected feature values.");
      if (hadPca) renderPcaPanel();
    });
    runPcaSupervisedBtn.addEventListener("click", renderPcaSupervisedFeatureRanking);
    pcaTopNFeaturesInput.addEventListener("change", () => {
      if (lastPcaSupervisedRows.length) renderPcaSupervisedFeatureRanking();
    });
    pcaMinCompleteCasesInput.addEventListener("change", () => {
      if (lastPcaSupervisedRows.length) renderPcaSupervisedFeatureRanking();
    });
    applyPcaTopFeaturesBtn.addEventListener("click", applyPcaSupervisedTopFeatures);
    runPcaAutoClusterBtn.addEventListener("click", () => {
      pcaAutoClusterMode = true;
      renderPcaPanel();
    });
    clearPcaAutoClusterBtn.addEventListener("click", () => {
      pcaAutoClusterMode = false;
      pcaAutoClusterInfo = null;
      if (lastPcaRender) renderPcaPanel();
      else {
        renderPcaColorRuleTable();
        pcaStatusEl.textContent = "Manual PCA color rules active. Click Run PCA to compute.";
      }
    });
    pcaClusterCountInput.addEventListener("change", () => {
      if (pcaAutoClusterMode && lastPcaRender) renderPcaPanel();
    });
    pcaSelectAllBtn.addEventListener("click", () => {
      pcaSelectedFeatures = new Set(numericFeatures);
      renderPcaFeatureSelector(numericFeatures);
      pcaStatusEl.textContent = `Selected all ${numericFeatures.length} PCA features. Click Run PCA to recompute.`;
    });
    pcaClearAllBtn.addEventListener("click", () => {
      pcaSelectedFeatures = new Set();
      renderPcaFeatureSelector(numericFeatures);
      pcaStatusEl.textContent = "Cleared all PCA features. Select at least two features before running PCA.";
    });
    downloadPcaFeatureSelectionBtn.addEventListener("click", downloadPcaFeatureSelection);
    reloadPcaFeatureSelectionBtn.addEventListener("click", () => {
      pcaFeatureSelectionFile.click();
    });
    bindFeatureSelectionFileInput(
      pcaFeatureSelectionFile,
      (payload, sourceLabel) => applyFeatureSelectionPayload(
        payload,
        numericFeatures,
        (selection) => { pcaSelectedFeatures = selection; },
        () => renderPcaFeatureSelector(numericFeatures),
        pcaStatusEl,
        sourceLabel,
        "PCA"
      ),
      pcaStatusEl,
      "PCA"
    );
    addPcaColorRuleBtn.addEventListener("click", () => {
      const palette = ["#0f766e", "#b45309", "#be123c", "#2563eb", "#7c3aed", "#15803d"];
      const idx = pcaColorRules.length + 1;
      pcaAutoClusterMode = false;
      pcaAutoClusterInfo = null;
      pcaColorRules.push({
        id: `pca_rule_${Date.now()}_${idx}`,
        enabled: true,
        label: `PCA Class ${idx}`,
        lowerOperator: "",
        lowerValue: "",
        upperOperator: "",
        upperValue: "",
        color: palette[(idx - 1) % palette.length],
      });
      if (lastPcaRender) renderPcaPanel();
      else renderPcaColorRuleTable();
    });
    pcaColorFeature.addEventListener("change", () => {
      if (lastPcaRender) renderPcaPanel();
      else {
        renderPcaColorRuleTable();
        pcaStatusEl.textContent = "PCA color feature changed. Click Run PCA to compute.";
      }
    });
    pcaColorFeatureGroup.addEventListener("change", () => {
      window.setTimeout(() => {
        if (lastPcaRender) renderPcaPanel();
      }, 0);
    });
    exportSelectAllBtn.addEventListener("click", () => {
      exportSelectedFeatures = new Set(exportFeatures);
      renderExportFeatureSelector(exportFeatures);
      updateExportStatus(`Selected all ${exportFeatures.length} export feature columns.`);
    });
    exportClearAllBtn.addEventListener("click", () => {
      exportSelectedFeatures = new Set();
      renderExportFeatureSelector(exportFeatures);
      updateExportStatus("Cleared all export feature columns.");
    });
    exportCsvBtn.addEventListener("click", () => {
      exportSelectedFeatures = new Set(
        Array.from(exportFeatureListEl.querySelectorAll('input[data-export-feature]:checked'))
          .map((el) => el.getAttribute("data-export-feature"))
          .filter(Boolean)
      );
      exportSelectedFeaturesCsv();
    });
    addTsneColorRuleBtn.addEventListener("click", () => {
      const palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"];
      const idx = tsneColorRules.length + 1;
      tsneColorRules.push({
        id: `rule_${Date.now()}_${idx}`,
        enabled: true,
        label: `Group ${idx}`,
        lowerOperator: "",
        lowerValue: "",
        upperOperator: "",
        upperValue: "",
        color: palette[(idx - 1) % palette.length],
      });
      if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
      else renderTsneColorRuleTable();
    });
    tsneColorFeature.addEventListener("change", () => {
      tsneColorLabelStatusEl.textContent = "Color feature changed. Existing t-SNE embedding will be recolored if rules are defined.";
      if (lastTsneRender) renderTsnePlotFromEmbedding(lastTsneRender);
    });
    runMlrBtn.addEventListener("click", () => {
      renderMlrAnalysis();
    });
    exportMlrCsvBtn.addEventListener("click", () => {
      const csv = buildRowsCsv(lastMlrExportRows, [
        "visual_component", "contribution_rank", "response_feature", "predictor_feature",
        "coefficient", "partial_SS", "model_SS_percent", "F", "p_value",
        "model_r2", "model_adjusted_r2", "model_sse", "model_ssr", "df_model", "df_error",
        "rows_used", "predictors_used", "candidate_predictors_checked", "all_candidate_predictors",
        "selection_mode", "dropped_unused_predictors", "rejected_for_fit",
        "complete_case_rows_used", "rows_skipped_missing_selected_values",
        "baseline_r2", "baseline_adjusted_r2", "baseline_sse"
      ]);
      if (!downloadCsv(csv, "analysis_mlr_anova_data")) {
        mlrStatusEl.textContent = "No MLR/ANOVA data available to export. Run Multilinear Regression first.";
      }
    });
    mlrResponseFeature.addEventListener("change", () => {
      renderMlrFeatureSelector();
      mlrStatusEl.textContent = "Response changed. Review selected predictors and click Run Multilinear Regression.";
    });
    mlrSelectAllBtn.addEventListener("click", () => {
      mlrIncludedFeatures = new Set(numericFeatures.filter((feature) => feature !== mlrResponseFeature.value));
      renderMlrFeatureSelector();
      mlrStatusEl.textContent = `All ${mlrIncludedFeatures.size} candidate predictors selected. Click Run Multilinear Regression to refit.`;
    });
    mlrClearAllBtn.addEventListener("click", () => {
      mlrIncludedFeatures = new Set();
      renderMlrFeatureSelector();
      mlrStatusEl.textContent = "All candidate predictors cleared. Select features to include before fitting.";
    });
    downloadMlrFeatureSelectionBtn.addEventListener("click", downloadMlrFeatureSelection);
    reloadMlrFeatureSelectionBtn.addEventListener("click", () => {
      mlrFeatureSelectionFile.click();
    });
    bindFeatureSelectionFileInput(
      mlrFeatureSelectionFile,
      (payload, sourceLabel) => applyFeatureSelectionPayload(
        payload,
        numericFeatures.filter((feature) => feature !== mlrResponseFeature.value),
        (selection) => { mlrIncludedFeatures = selection; },
        renderMlrFeatureSelector,
        mlrStatusEl,
        sourceLabel,
        "MLR predictor"
      ),
      mlrStatusEl,
      "MLR"
    );
    xFeature.addEventListener("change", () => {
      refreshAnalysis();
    });
    yFeature.addEventListener("change", () => {
      refreshAnalysis();
    });
    cohortFeature.addEventListener("change", rerenderCohortHistogram);
    distFeature.addEventListener("change", rerenderFeatureDistribution);
    distLevel.addEventListener("change", rerenderFeatureDistribution);
  </script>
</body>
</html>
"""


def _split_inline_dashboard_script():
    page = _inline_html_page()
    start_marker = "  <script>\n"
    end_marker = "  </script>"
    start = page.index(start_marker)
    script_start = start + len(start_marker)
    end = page.index(end_marker, script_start)
    return page, start, script_start, end, end + len(end_marker)


def analysis_dashboard_script():
    page, _, script_start, end, _ = _split_inline_dashboard_script()
    return page[script_start:end].strip() + "\n"


def html_page():
    page, start, _, _, end_close = _split_inline_dashboard_script()
    external_script = '  <script defer src="/analysis_dashboard.js"></script>'
    return page[:start] + external_script + page[end_close:]


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text, code=200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_js(self, text, code=200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(html_page())
            return
        if parsed.path == "/analysis_dashboard.js":
            self._send_js(analysis_dashboard_script())
            return
        if parsed.path == "/api/analysis/current":
            try:
                if not ANNOTATION_STATE_JSON.exists():
                    self._send_json({"error": f"Annotation file not found: {ANNOTATION_STATE_JSON}"}, code=404)
                    return
                payload = json.loads(ANNOTATION_STATE_JSON.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Annotation JSON root is not an object.")
                peak = payload.get("peak_pair_features", payload.get("paired_features"))
                if isinstance(peak, dict):
                    payload["peak_pair_features"] = peak
                    payload["paired_features"] = peak
                meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                meta["source"] = "Current alignment annotations"
                meta["source_path"] = str(ANNOTATION_STATE_JSON)
                payload["meta"] = meta
                self._send_json(payload)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, code=500)
                return
        self._send_json({"ok": False, "error": "Not found"}, code=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid JSON payload: {exc}"}, code=400)
            return

        if parsed.path == "/api/tsne":
            try:
                matrix = payload.get("matrix", [])
                perplexity = float(payload.get("perplexity", 10))

                if not isinstance(matrix, list) or len(matrix) < 3:
                    self._send_json({"ok": False, "error": "Need at least 3 rows for t-SNE."}, code=400)
                    return

                cleaned = []
                dims = None
                for row in matrix:
                    if not isinstance(row, list):
                        continue
                    vals = [float(x) for x in row]
                    if dims is None:
                        dims = len(vals)
                    if not dims or len(vals) != dims:
                        continue
                    cleaned.append(vals)

                if len(cleaned) < 3:
                    self._send_json({"ok": False, "error": "Insufficient valid rows for t-SNE."}, code=400)
                    return

                per = max(2.0, min(30.0, min(perplexity, (len(cleaned) - 1) / 3.0)))
                method = "scikit-learn t-SNE"
                if SklearnTSNE is not None:
                    try:
                        model = SklearnTSNE(
                            n_components=2,
                            perplexity=per,
                            init="pca",
                            learning_rate="auto",
                            random_state=42,
                            max_iter=1000,
                        )
                    except TypeError:
                        model = SklearnTSNE(
                            n_components=2,
                            perplexity=per,
                            init="pca",
                            learning_rate="auto",
                            random_state=42,
                            n_iter=1000,
                        )
                    emb = model.fit_transform(cleaned)
                else:
                    emb = fallback_tsne_embedding(cleaned, perplexity=per, max_iter=500, random_state=42)
                    method = "NumPy t-SNE fallback" if NUMPY_AVAILABLE else "dependency-free PCA fallback"
                out = [[float(v[0]), float(v[1])] for v in emb]
                self._send_json({"ok": True, "embedding": out, "method": method}, code=200)
                return
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, code=500)
                return

        self._send_json({"ok": False, "error": "Not found"}, code=404)

    def log_message(self, fmt, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Analysis dashboard running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
