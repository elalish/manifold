"use strict";

const DEFAULT_DATA_ROOT =
  "https://raw.githubusercontent.com/elalish/manifold/benchmark-data";

const CHARTS = [
  {
    element: "perf-time-chart",
    metric: "perf_time",
    title: "perfTest size sweep time",
    unit: "Milliseconds",
    valueSuffix: " ms",
  },
  {
    element: "perf-rss-chart",
    metric: "perf_rss",
    title: "perfTest peak RSS",
    unit: "MB",
    valueSuffix: " MB",
  },
  {
    element: "ember-full-chart",
    metric: "ember_full",
    title: "Ember full phase time",
    unit: "Milliseconds",
    valueSuffix: " ms",
  },
  {
    element: "ember-phase-chart",
    metric: "ember_phase",
    title: "Ember individual phase time",
    unit: "Milliseconds",
    valueSuffix: " ms",
    maxSeries: 16,
  },
  {
    element: "gtest-time-chart",
    metric: "gtest_time",
    title: "Existing gtest time",
    unit: "Milliseconds",
    valueSuffix: " ms",
  },
];

const state = {
  dataRoot: "",
  branchRoot: "",
  weeklyRoot: "",
  index: null,
  records: [],
  scale: "linear",
};

function qs(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  const element = qs(id);
  if (element) {
    element.textContent = text;
  }
}

function setStatus(text) {
  setText("status-text", text);
}

function setError(message) {
  const box = qs("error-box");
  if (!box) {
    return;
  }
  if (!message) {
    box.hidden = true;
    box.textContent = "";
    return;
  }
  box.hidden = false;
  box.textContent = message;
}

function normalizeRoot(root) {
  return (root || DEFAULT_DATA_ROOT).replace(/\/+$/, "");
}

function withNoCache(url) {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}t=${Date.now()}`;
}

async function fetchJson(url) {
  const response = await fetch(withNoCache(url), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${url}`);
  }
  return response.json();
}

async function loadIndex(dataRoot) {
  const root = normalizeRoot(dataRoot);
  const rootIsWeeklyDir = root.endsWith("/weekly");
  const candidate = rootIsWeeklyDir
    ? {
        indexUrl: `${root}/index.json`,
        branchRoot: root.slice(0, -"/weekly".length),
        weeklyRoot: root,
      }
    : {
        indexUrl: `${root}/weekly/index.json`,
        branchRoot: root,
        weeklyRoot: `${root}/weekly`,
      };

  try {
    const index = await fetchJson(candidate.indexUrl);
    return { ...candidate, index };
  } catch (error) {
    throw new Error(`Unable to load benchmark index.\n${error.message}`);
  }
}

function resultUrl(entry) {
  const path = entry.result_path || "";
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  if (path.startsWith("weekly/")) {
    return `${state.branchRoot}/${path}`;
  }
  return `${state.weeklyRoot}/${path}`;
}

function shortSha(sha) {
  return sha ? sha.slice(0, 8) : "unknown";
}

function formatDate(raw) {
  if (!raw) {
    return "unknown";
  }
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) {
    return raw;
  }
  return date.toISOString().slice(0, 10);
}

function runLabel(entry) {
  const label = formatDate(entry.timestamp);
  return entry.release_tag ? `${label} (${entry.release_tag})` : label;
}

function sanitizerSummary(entry) {
  const sanitizer = entry?.sanitizer;
  if (!sanitizer) {
    return "-";
  }
  const subset = sanitizer.subset || "sanitizer";
  const build = sanitizer.build_result || "unknown";
  const test = sanitizer.test_result || "unknown";
  return build === "success" ? `${subset}: ${test}` : `${subset}: build ${build}`;
}

function statKey(stat, suffix = "") {
  return `${stat}${suffix}`;
}

function currentStat() {
  return qs("stat-select")?.value || "median";
}

function metricItems(record, metricId, stat) {
  const suites = record.result.suites || {};

  if (metricId === "perf_time") {
    return (suites.perf_size_sweep?.workloads || []).map((workload) => ({
      name: workload.benchmark || `nTri=${workload.n_tri}`,
      value: workload.time_ms?.[statKey(stat, "_ms")] ?? null,
    }));
  }

  if (metricId === "perf_rss") {
    return (suites.perf_size_sweep?.workloads || []).map((workload) => ({
      name: workload.benchmark || `nTri=${workload.n_tri}`,
      value: workload.peak_rss_mb?.[stat] ?? null,
    }));
  }

  if (metricId === "ember_full") {
    return (suites.weekly_ember_phase?.cases || []).map((caseItem) => ({
      name: `case ${caseItem.case_index}`,
      value: caseItem.full_phase_sum_ms?.[statKey(stat, "_ms")] ?? null,
    }));
  }

  if (metricId === "ember_phase") {
    const items = [];
    for (const caseItem of suites.weekly_ember_phase?.cases || []) {
      for (const [phase, stats] of Object.entries(caseItem.phases || {})) {
        items.push({
          name: `case ${caseItem.case_index} / ${phase}`,
          value: stats?.[statKey(stat, "_ms")] ?? null,
        });
      }
    }
    return items;
  }

  if (metricId === "gtest_time") {
    return (suites.existing_gtests?.tests || []).map((test) => ({
      name: test.name,
      value: test.time_ms?.[statKey(stat, "_ms")] ?? null,
    }));
  }

  return [];
}

function buildSeries(metricId, stat, maxSeries) {
  const categories = state.records.map((record) => runLabel(record.entry));
  const names = new Set();
  const byRun = state.records.map((record) => {
    const map = new Map();
    for (const item of metricItems(record, metricId, stat)) {
      names.add(item.name);
      map.set(item.name, item.value);
    }
    return map;
  });

  const latestMap = byRun[byRun.length - 1] || new Map();
  const sortedNames = [...names].sort((a, b) => {
    const bValue = latestMap.get(b) ?? -Infinity;
    const aValue = latestMap.get(a) ?? -Infinity;
    return bValue - aValue || a.localeCompare(b);
  });

  const visibleNames = sortedNames.slice(0, maxSeries || sortedNames.length);
  const series = visibleNames.map((name) => ({
    type: "line",
    name,
    animation: false,
    data: byRun.map((run) => {
      const value = run.get(name) ?? null;
      if (state.scale === "log" && value !== null && value <= 0) {
        return null;
      }
      return value;
    }),
  }));

  return { categories, series, totalSeries: sortedNames.length };
}

function renderHighchartsChart(config) {
  const stat = currentStat();
  const { categories, series, totalSeries } = buildSeries(
    config.metric,
    stat,
    config.maxSeries,
  );

  Highcharts.chart({
    chart: {
      renderTo: config.element,
      zooming: {
        type: "xy",
      },
      type: "line",
    },
    title: {
      text: config.title,
    },
    subtitle: {
      text:
        totalSeries > series.length
          ? `Showing top ${series.length} of ${totalSeries} series by latest ${stat}`
          : `${stat} across ${categories.length} loaded runs`,
    },
    yAxis: {
      title: { text: config.unit },
      min: state.scale === "linear" ? 0 : undefined,
      type: state.scale === "log" ? "logarithmic" : undefined,
    },
    xAxis: {
      categories,
      title: { text: "Run date" },
    },
    tooltip: {
      shared: false,
      valueSuffix: config.valueSuffix,
    },
    series,
  });
}

function renderCharts() {
  const chartsOnPage = CHARTS.filter((chart) => qs(chart.element));
  if (!chartsOnPage.length) {
    return;
  }
  if (!window.Highcharts) {
    throw new Error("Highcharts was not loaded.");
  }
  for (const chart of chartsOnPage) {
    renderHighchartsChart(chart);
  }
}

function renderLatestMetadata() {
  const latest = state.records[state.records.length - 1];
  if (!latest) {
    setText("latest-run", "-");
    setText("latest-cpu", "-");
    setText("latest-compiler", "-");
    return;
  }

  const metadata = latest.result.metadata || {};
  const entry = latest.entry;
  setText("latest-run", `${formatDate(entry.timestamp)} / ${shortSha(entry.commit_sha)}`);
  setText("latest-cpu", metadata.cpu_model || entry.cpu_model || "unknown");
  setText("latest-compiler", metadata.compiler || entry.compiler || "unknown");
  renderLatestTable(latest);
}

function renderLatestTable(latest) {
  const container = qs("latest-metadata-table");
  if (!container) {
    return;
  }
  container.replaceChildren();

  const metadata = latest.result.metadata || {};
  const storage = latest.result.storage || latest.entry || {};
  const sanitizer = storage.sanitizer || latest.entry.sanitizer || {};
  const rows = [
    ["Run ID", storage.run_id || latest.entry.run_id || "unknown"],
    ["Trigger", storage.trigger || latest.entry.trigger || "weekly"],
    ["Release tag", storage.release_tag || latest.entry.release_tag || "-"],
    ["Timestamp", storage.timestamp || metadata.timestamp || latest.entry.timestamp || "unknown"],
    ["Commit", storage.commit_sha || metadata.commit_sha || latest.entry.commit_sha || "unknown"],
    ["Workflow", storage.workflow || metadata.workflow || latest.entry.workflow || "unknown"],
    ["Runner", storage.runner || metadata.runner || latest.entry.runner || "unknown"],
    ["OS", storage.os || metadata.os || latest.entry.os || "unknown"],
    ["Compiler", metadata.compiler || storage.compiler || latest.entry.compiler || "unknown"],
    ["CPU", metadata.cpu_model || storage.cpu_model || latest.entry.cpu_model || "unknown"],
    ["CPU count", metadata.cpu_count || storage.cpu_count || latest.entry.cpu_count || "unknown"],
    ["Sanitizer subset", sanitizer.subset || "-"],
    ["Sanitizer build", sanitizer.build_result || "-"],
    ["Sanitizer test", sanitizer.test_result || "-"],
  ];

  const table = document.createElement("table");
  const tbody = document.createElement("tbody");
  for (const [label, value] of rows) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = label;
    const td = document.createElement("td");
    td.textContent = value;
    tr.append(th, td);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
}

function renderRunTable(records) {
  const container = qs("run-table");
  if (!container) {
    return;
  }
  container.replaceChildren();

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const text of ["date", "release", "commit", "runner", "cpu", "sanitizer", "run", "json"]) {
    const th = document.createElement("th");
    th.textContent = text;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const record of [...records].reverse()) {
    const entry = record.entry;
    const tr = document.createElement("tr");
    for (const text of [
      formatDate(entry.timestamp),
      entry.release_tag || "-",
      shortSha(entry.commit_sha),
      entry.runner || "unknown",
      entry.cpu_model || "unknown",
      sanitizerSummary(entry),
    ]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }

    const runCell = document.createElement("td");
    if (entry.github_run_url) {
      const runLink = document.createElement("a");
      runLink.href = entry.github_run_url;
      runLink.textContent = entry.run_id || "run";
      runCell.appendChild(runLink);
    } else {
      runCell.textContent = entry.run_id || "-";
    }
    tr.appendChild(runCell);

    const jsonCell = document.createElement("td");
    const jsonLink = document.createElement("a");
    jsonLink.href = resultUrl(entry);
    jsonLink.textContent = "result.json";
    jsonCell.appendChild(jsonLink);
    tr.appendChild(jsonCell);

    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
}

function renderDashboard() {
  renderLatestMetadata();
  renderCharts();
  renderRunTable(state.records);
}

function selectedRunCount(total) {
  const raw = qs("run-count")?.value || "10";
  return raw === "all" ? total : Math.min(Number(raw), total);
}

async function loadDashboard() {
  setError("");
  setStatus("Loading data");
  setText("data-summary", "Fetching benchmark index...");
  const loadButton = qs("load-button");
  if (loadButton) {
    loadButton.disabled = true;
  }

  try {
    const root = state.dataRoot || DEFAULT_DATA_ROOT;
    const loaded = await loadIndex(root);
    state.dataRoot = root;
    state.branchRoot = loaded.branchRoot;
    state.weeklyRoot = loaded.weeklyRoot;
    state.index = loaded.index;

    const runs = [...(loaded.index.runs || [])].sort((a, b) =>
      (a.timestamp || "").localeCompare(b.timestamp || ""),
    );
    const count = selectedRunCount(runs.length);
    const selected = runs.slice(-count);

    setText("data-summary", `Fetching ${selected.length} result file${selected.length === 1 ? "" : "s"}...`);
    state.records = await Promise.all(
      selected.map(async (entry) => ({
        entry,
        result: await fetchJson(resultUrl(entry)),
      })),
    );

    setStatus("Loaded");
    setText(
      "data-summary",
      `${runs.length} run${runs.length === 1 ? "" : "s"} in index, ${state.records.length} loaded. Latest: ${
        loaded.index.latest_run_id || "unknown"
      }.` ,
    );
    renderDashboard();
  } catch (error) {
    setStatus("Load failed");
    setText("data-summary", "The dashboard could not load benchmark data.");
    setError(error.stack || error.message || String(error));
  } finally {
    if (loadButton) {
      loadButton.disabled = false;
    }
  }
}

function handleScaleChange(event) {
  const value = event.target.value;
  if (value !== "linear" && value !== "log") {
    console.error("Invalid scale");
    return;
  }
  if (state.scale !== value) {
    state.scale = value;
    renderDashboard();
  }
}

function updateNavLinks() {
  const data = new URLSearchParams(window.location.search).get("data");
  if (!data) {
    return;
  }
  for (const link of document.querySelectorAll('a[href$=".html"]')) {
    const url = new URL(link.getAttribute("href"), window.location.href);
    url.searchParams.set("data", data);
    link.href = url.pathname.split("/").pop() + url.search;
  }
}

function init() {
  const params = new URLSearchParams(window.location.search);
  state.dataRoot = params.get("data") || DEFAULT_DATA_ROOT;
  setText("data-root-label", state.dataRoot);
  updateNavLinks();

  qs("load-button")?.addEventListener("click", loadDashboard);
  qs("stat-select")?.addEventListener("change", renderDashboard);
  qs("run-count")?.addEventListener("change", loadDashboard);
  qs("linear-scale-input")?.addEventListener("input", handleScaleChange);
  qs("log-scale-input")?.addEventListener("input", handleScaleChange);

  if (document.body.dataset.page !== "help") {
    loadDashboard();
  }
}

document.addEventListener("DOMContentLoaded", init);
