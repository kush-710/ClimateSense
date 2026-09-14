// Set this to your deployed Render backend URL (e.g. "https://climatesense-api.onrender.com").
// Defaults to localhost for local development against `uvicorn api.main:app --port 8000`.
// Backend URL. Paste your deployed Render URL here once (no trailing slash):
//   const PROD_API = "https://climatesense-api.onrender.com";
// Local development keeps using localhost automatically, so the same commit
// works in both places and there's nothing to edit per deploy.
const PROD_API = "";   // <-- set this after deploying the backend

const IS_LOCAL = ["localhost", "127.0.0.1", ""].includes(location.hostname);
const API_BASE = IS_LOCAL ? "http://localhost:8000" : PROD_API;

if (!IS_LOCAL && !PROD_API) {
  // No visible warning by design: visitors should never see configuration
  // text. The console note keeps it debuggable without putting anything on
  // the page.
  console.warn("PROD_API is not set in docs/app.js; API calls will not resolve.");
}

const SEVERITY_COLOR = {
  SAFE: "var(--safe)", MODERATE: "var(--moderate)",
  HIGH: "var(--high)", CRITICAL: "var(--critical)",
};

let CITY_NAMES = {};       // populated from /cities at startup
let currentCity = 1;
let currentSport = "general";
let currentForecastTarget = "temp_c";

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function fmt(v, digits = 1) {
  return v === null || v === undefined ? "-" : (Math.round(v * 10 ** digits) / 10 ** digits);
}

function renderError(el, err) {
  el.innerHTML = `<p class="error">Couldn't load this: ${err.message}</p>`;
}

// ---------------- clock ----------------
function tickClock() {
  const now = new Date(Date.now() + 5.5 * 3600 * 1000); // IST = UTC+5:30
  const hh = String(now.getUTCHours()).padStart(2, "0");
  const mm = String(now.getUTCMinutes()).padStart(2, "0");
  const ss = String(now.getUTCSeconds()).padStart(2, "0");
  document.getElementById("clock").textContent = `${hh}:${mm}:${ss} IST`;
}
setInterval(tickClock, 1000);
tickClock();

// ---------------- city filter ----------------
async function loadCityDropdown() {
  const select = document.getElementById("city-select");
  try {
    CITY_NAMES = await apiGet("/cities");
    const ids = Object.keys(CITY_NAMES).sort((a, b) => a - b);
    select.innerHTML = ids.map(id => `<option value="${id}">${CITY_NAMES[id].name}</option>`).join("");
    currentCity = parseInt(ids[0], 10);
    select.value = currentCity;
  } catch (err) {
    select.innerHTML = `<option>Unavailable</option>`;
  }
}

document.getElementById("city-select").addEventListener("change", (e) => {
  currentCity = parseInt(e.target.value, 10);
  loadDashboard();
});

// ---------------- risk panel ----------------
const GAUGE_CIRC = 2 * Math.PI * 58;

async function loadRiskPanel() {
  const body = document.getElementById("risk-body");
  try {
    const data = await apiGet(`/climate/${currentCity}?sport=${currentSport}`);
    const obs = data.observed || {};

    const arc = document.getElementById("gauge-arc");
    const offset = GAUGE_CIRC * (1 - (data.risk_score || 0) / 100);
    arc.style.strokeDasharray = GAUGE_CIRC;
    arc.style.strokeDashoffset = offset;
    arc.style.stroke = SEVERITY_COLOR[data.severity] || "var(--safe)";
    document.getElementById("gauge-score").textContent = data.risk_score ?? "-";
    document.getElementById("gauge-severity").textContent = data.severity;

    // --- Current Conditions: identical for every activity, shown once ---
    // AQI is asterisked with its own timestamp when it's not from the same
    // hour as the weather reading: Open-Meteo's AQ feed can lag the weather
    // feed by up to a day or two, so this is often the most recent reading
    // actually available, not a live "right now" number.
    let aqStale = false;
    if (obs.aq_observed_at && data.observed_at) {
      const wxTime = new Date(data.observed_at.replace(" ", "T")).getTime();
      const aqTime = new Date(obs.aq_observed_at).getTime();
      aqStale = Math.abs(wxTime - aqTime) > 3600 * 1000; // different hour
    }
    const sourceLabel = obs.aq_source === "aqicn" ? "station" : obs.aq_source === "open-meteo" ? "model" : "";
    const aqLabel = aqStale
      ? `AQI (CPCB, ${sourceLabel}), as of ${new Date(obs.aq_observed_at).toLocaleString("en-IN", {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"})}`
      : `AQI (CPCB${sourceLabel ? ", " + sourceLabel : ""})`;
    // UV is 0 all night, which looks like missing data, so show today's peak
    // alongside it for context.
    const p = data.uv_peak_today;
    const uvLabel = (p && (obs.uv_index === 0 || obs.uv_index == null))
      ? `UV Index now (peak today ${p.uv_index} at ${String(p.hour).padStart(2, "0")}:00)`
      : "UV Index";

    const tiles = [
      ["temp_c", "°C", "Temp"], ["wbgt_c", "°C", "WBGT (heat stress)"],
      ["humidity_pct", "%", "Humidity"], ["wind_kph", "km/h", "Wind"],
      ["uv_index", "", uvLabel], ["aqi_cpcb", "", aqLabel],
    ];
    document.getElementById("metric-tiles").innerHTML = tiles.map(([key, unit, label]) => `
      <div class="metric-tile">
        <div class="metric-value">${fmt(obs[key])}${obs[key] != null ? unit : ""}</div>
        <div class="metric-label">${label}</div>
      </div>
    `).join("");

    // Risk 24h out, scored from the ingested weather forecast for that hour.
    const t = data.tomorrow;
    const tomorrowLine = t
      ? `<p><b>Forecast ${t.horizon_h}h ahead</b><br>
           <span class="severity-tag severity-${t.tier_name}">${t.tier_name}</span>
           <span class="model-note">${t.risk_score}/100, ${t.basis}</span></p>`
      : "";

    document.getElementById("general-notes").innerHTML = `
      <span class="severity-tag severity-${data.severity}">${data.severity}</span>
      <p>${data.summary}</p>
      <p><b>Safe window</b><br>${data.safe_window.message}</p>
      ${tomorrowLine}
      <p><b>Hydration</b><br>${data.hydration}</p>
      <p><b>Mask</b><br>${data.mask}</p>
    `;

    // --- Sport Fitness Check: only this block changes with the dropdown ---
    const sv = data.sport_verdict;
    const limitRows = [
      ["WBGT", obs.wbgt_c, sv.limits.max_wbgt_c, "°C"],
      ["Temp", obs.temp_c, sv.limits.max_temp_c, "°C"],
      ["AQI", obs.aqi_cpcb, sv.limits.max_aqi, ""],
      ["UV", obs.uv_index, sv.limits.max_uv, ""],
    ];
    document.getElementById("sport-limits").innerHTML = limitRows.map(([label, cur, lim, unit]) => {
      const breached = cur != null && cur > lim;
      const cls = cur == null ? "" : breached ? "breach" : "ok";
      return `
        <div class="limit-cell ${cls}">
          <div class="limit-label">${label} / ${sv.sport.replace(/_/g, " ")}</div>
          <div class="limit-compare"><span class="cur">${fmt(cur)}${unit}</span> <span class="lim">/ ${lim}${unit} limit</span></div>
        </div>
      `;
    }).join("");

    document.getElementById("sport-notes").innerHTML = `
      <p><b>Verdict</b><br>${sv.playable ? "Cleared to play" : "Not cleared"} -
         ${sv.breaches.length ? sv.breaches.join("; ") : "no threshold breached for " + sv.sport.replace(/_/g, " ")}</p>
    `;
  } catch (err) {
    renderError(body, err);
  }
}

// ---------------- outlook line (ENSO regime + real CPC forecast) ----------------
async function loadOutlook() {
  const el = document.getElementById("outlook-line");
  el.textContent = "Loading outlook...";
  try {
    const data = await apiGet(`/enso?city_id=${currentCity}`);
    const parts = [data.latest.narrative];
    if (data.outlook_headline) parts.push(data.outlook_headline + ".");
    el.innerHTML = `<b>Outlook -</b> ${parts.join(" ")}`;
  } catch (err) {
    el.textContent = `Outlook unavailable: ${err.message}`;
  }
}

// ---------------- forecast chart (hand-rolled SVG line chart) ----------------
async function loadForecastChart() {
  const el = document.getElementById("forecast-chart");
  el.textContent = "Loading...";
  try {
    const resp = await apiGet(`/forecast/${currentCity}?target=${currentForecastTarget}&days=7`);
    const data = resp.points || [];
    const acc = resp.accuracy;
    if (!data.length) { el.innerHTML = "<p class='error'>No forecast data.</p>"; return; }

    const W = 560, H = 200, PAD_L = 46, PAD_R = 12, PAD_T = 14, PAD_B = 26;
    const unit = currentForecastTarget === "temp_c" ? "°C" : "";          // compact, for chart axis
    const stripUnit = currentForecastTarget === "temp_c" ? "°C" : " µg/m³"; // fuller, for the day-strip
    const lowers = data.map(d => d.lower);
    const uppers = data.map(d => d.upper);
    const min = Math.min(...lowers), max = Math.max(...uppers);
    const range = (max - min) || 1;
    const plotW = W - PAD_L - PAD_R, plotH = H - PAD_T - PAD_B;
    const x = i => PAD_L + (i / (data.length - 1)) * plotW;
    const y = v => PAD_T + plotH - ((v - min) / range) * plotH;

    const linePath = data.map((d, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(d.value)}`).join(" ");
    const areaPath = `M${x(0)},${y(lowers[0])} ` +
      data.map((d, i) => `L${x(i)},${y(d.upper)}`).join(" ") +
      data.slice().reverse().map((d, i) => `L${x(data.length - 1 - i)},${y(d.lower)}`).join(" ") + " Z";

    const yTicks = [0, 0.5, 1].map(f => min + f * range);
    const gridLines = yTicks.map(v => `
      <line class="chart-grid" x1="${PAD_L}" x2="${W - PAD_R}" y1="${y(v)}" y2="${y(v)}"/>
      <text class="chart-axis-label" x="${PAD_L - 6}" y="${y(v) + 3}" text-anchor="end">${fmt(v)}${unit}</text>
    `).join("");

    const points = data.map((d, i) => `
      <g class="chart-point">
        <circle cx="${x(i)}" cy="${y(d.value)}" r="3"/>
        <text class="chart-axis-label" x="${x(i)}" y="${H - 8}" text-anchor="middle">${d.date.slice(5)}</text>
      </g>
    `).join("");

    // Show how far off this model typically is, so the numbers aren't read
    // as more precise than they are.
    const accNote = acc
      ? `<div class="accuracy-note">Prophet walk-forward accuracy: MAE +/-${acc.mae.toFixed(1)}${unit || " µg/m³"},
         RMSE ${acc.rmse.toFixed(1)}. Shaded band = 80% interval.</div>`
      : "";

    el.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}">
        ${gridLines}
        <path class="chart-area" d="${areaPath}"/>
        <path class="chart-line" d="${linePath}"/>
        ${points}
      </svg>
      ${accNote}
    `;

    // Explicit ML-forecasted numbers per day (Prophet yhat + 80% interval),
    // since a line chart alone doesn't let you read exact values.
    document.getElementById("forecast-strip").innerHTML = data.map(d => `
      <div class="forecast-day">
        <div class="fd-date">${d.date.slice(5)}</div>
        <div class="fd-value">${d.value}${stripUnit}</div>
        <div class="fd-range">${d.lower}-${d.upper}</div>
      </div>
    `).join("");
  } catch (err) {
    renderError(el, err);
    document.getElementById("forecast-strip").innerHTML = "";
  }
}

// ---------------- policy panel ----------------
async function loadPolicyPanel() {
  const grapEl = document.getElementById("grap-steps");
  const statusEl = document.getElementById("grap-status");
  const barsEl = document.getElementById("source-bars");
  const noteEl = document.getElementById("policy-note");
  barsEl.textContent = "Loading...";
  try {
    const data = await apiGet(`/policy/${currentCity}`);

    if (data.grap_applicable) {
      const activeStage = data.grap_stage ? data.grap_stage.stage : null;
      grapEl.innerHTML = data.grap_reference.map(ref => `
        <div class="grap-step ${activeStage === ref.stage ? "current" : ""}">
          <span class="stage-num">${ref.stage}</span>
          <span class="stage-label">${ref.label}</span>
          <span class="stage-range">AQI ${ref.aqi_range[0]}-${ref.aqi_range[1]}</span>
        </div>
      `).join("");
      statusEl.innerHTML = activeStage
        ? `<span class="dot" style="color:var(--accent)"></span>Stage ${activeStage} in effect at AQI ${fmt(data.aqi_cpcb, 0)} - ${data.grap_stage.actions.join("; ")}`
        : `<span class="dot"></span>AQI${data.aqi_cpcb != null ? " " + fmt(data.aqi_cpcb, 0) : ""} is below the GRAP Stage I threshold (201) - no mandatory stage in effect.`;
    } else {
      // GRAP is Delhi-NCR-specific policy machinery - show the national AQI
      // category instead of implying a framework that doesn't apply here.
      grapEl.innerHTML = "";
      statusEl.innerHTML = `<span class="dot"></span>GRAP (CAQM) applies to Delhi-NCR only - not shown for ${data.city}. `
        + (data.aqi_category
           ? `National AQI category: <b>${data.aqi_category}</b>${data.aqi_cpcb != null ? ` (${fmt(data.aqi_cpcb, 0)})` : ""}.`
           : "AQI reading unavailable right now.");
    }

    barsEl.innerHTML = data.likely_dominant_sources.map(s => `
      <div class="source-row">
        <div class="source-top">
          <span class="source-name">${s.source.replace(/_/g, " ")}</span>
          <span class="source-pct">${s.likelihood_pct}%</span>
        </div>
        <div class="source-bar-track"><div class="source-bar-fill" style="width:${s.likelihood_pct}%"></div></div>
        <div class="source-detail"><b>Action:</b> ${s.short_term_actions[0] || "-"}</div>
      </div>
    `).join("");

    noteEl.textContent = data.used_generic_source_prior
      ? data.methodology_note + " No published city-specific source-apportionment study is encoded for "
        + data.city + " - the shares above are generic indicative estimates, not a local study."
      : data.methodology_note;
  } catch (err) {
    renderError(barsEl, err);
    grapEl.innerHTML = "";
    statusEl.innerHTML = "";
    noteEl.textContent = "";
  }
}

function loadDashboard() {
  loadRiskPanel();
  loadOutlook();
  loadForecastChart();
  loadPolicyPanel();
}

// ---------------- event wiring ----------------
document.getElementById("sport-select").addEventListener("change", (e) => {
  currentSport = e.target.value;
  loadRiskPanel();
});

document.querySelector(".forecast-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".ftab");
  if (!btn) return;
  document.querySelectorAll(".ftab").forEach(t => t.classList.remove("active"));
  btn.classList.add("active");
  currentForecastTarget = btn.dataset.target;
  loadForecastChart();
});

document.getElementById("chat-toggle").addEventListener("click", () => {
  document.getElementById("chat-panel").classList.toggle("collapsed");
});

document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-text");
  const question = input.value.trim();
  if (!question) return;
  const messages = document.getElementById("chat-messages");

  messages.insertAdjacentHTML("beforeend", `<div class="msg user">${escapeHtml(question)}</div>`);
  input.value = "";
  messages.scrollTop = messages.scrollHeight;

  try {
    // currentCity is sent as the fallback; the backend still detects a
    // named city in the question text and overrides it if one is found.
    const data = await apiPost("/assistant/ask", { question, city_id: currentCity });
    const note = data.city_detected ? "" : ` <i>(assuming ${data.city})</i>`;
    messages.insertAdjacentHTML("beforeend", `<div class="msg bot">${escapeHtml(data.answer)}${note}</div>`);
  } catch (err) {
    messages.insertAdjacentHTML("beforeend", `<div class="msg error">${escapeHtml(err.message)}</div>`);
  }
  messages.scrollTop = messages.scrollHeight;
});

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

(async function init() {
  await loadCityDropdown();
  loadDashboard();
})();
