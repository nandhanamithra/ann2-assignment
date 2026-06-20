(() => {
  "use strict";

  // Canonical ordinal order for the dataset's 7 classes (LabelEncoder sorts
  // alphabetically, which is NOT the same as clinical severity order, so we
  // keep our own mapping for positioning things on the spectrum).
  const SEVERITY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
  ];

  const SEVERITY_SHORT = {
    Insufficient_Weight: "Insuff.",
    Normal_Weight: "Normal",
    Overweight_Level_I: "Over I",
    Overweight_Level_II: "Over II",
    Obesity_Type_I: "Obese I",
    Obesity_Type_II: "Obese II",
    Obesity_Type_III: "Obese III",
  };

  const SPECTRUM_COLORS = [
    "var(--spec-1)", "var(--spec-2)", "var(--spec-3)", "var(--spec-4)",
    "var(--spec-5)", "var(--spec-6)", "var(--spec-7)",
  ];

  // Friendly labels + grouping for known dataset columns. Anything not
  // listed here still renders, just with a humanized fallback label.
  const FIELD_META = {
    Gender: { group: "Body metrics", label: "Gender" },
    Age: { group: "Body metrics", label: "Age", unit: "yrs" },
    Height: { group: "Body metrics", label: "Height", unit: "m" },
    Weight: { group: "Body metrics", label: "Weight", unit: "kg" },
    family_history_with_overweight: { group: "Habits & diet", label: "Family history of overweight" },
    FAVC: { group: "Habits & diet", label: "Frequent high-caloric food" },
    FCVC: { group: "Habits & diet", label: "Vegetable consumption", unit: "freq/day" },
    NCP: { group: "Habits & diet", label: "Main meals per day" },
    CAEC: { group: "Habits & diet", label: "Snacking between meals" },
    CALC: { group: "Habits & diet", label: "Alcohol consumption" },
    SMOKE: { group: "Activity & lifestyle", label: "Smokes" },
    CH2O: { group: "Activity & lifestyle", label: "Water intake", unit: "L/day" },
    SCC: { group: "Activity & lifestyle", label: "Monitors calorie intake" },
    FAF: { group: "Activity & lifestyle", label: "Physical activity frequency", unit: "days/wk" },
    TUE: { group: "Activity & lifestyle", label: "Screen time", unit: "hrs/day" },
    MTRANS: { group: "Activity & lifestyle", label: "Main transportation" },
  };

  const GROUP_ORDER = ["Body metrics", "Habits & diet", "Activity & lifestyle", "Other"];

  const humanize = (name) =>
    name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  const form = document.getElementById("intake-form");
  const fieldsContainer = document.getElementById("form-fields");
  const submitBtn = document.getElementById("submit-btn");
  const statusLine = document.getElementById("status-line");

  const idleState = document.getElementById("readout-idle");
  const errorState = document.getElementById("readout-error");
  const resultState = document.getElementById("readout-result");
  const errorMessage = document.getElementById("error-message");

  const resultClassEl = document.getElementById("result-class");
  const resultConfidenceEl = document.getElementById("result-confidence");
  const spectrumTrack = document.getElementById("spectrum-track");
  const spectrumMarker = document.getElementById("spectrum-marker");
  const spectrumTicks = document.getElementById("spectrum-ticks");
  const probRows = document.getElementById("prob-rows");

  let meta = null;

  function showState(state) {
    [idleState, errorState, resultState].forEach((el) => el.classList.add("hidden"));
    state.classList.remove("hidden");
  }

  function buildSpectrum() {
    spectrumTrack.innerHTML = "";
    spectrumTicks.innerHTML = "";
    SEVERITY_ORDER.forEach((cls, i) => {
      const seg = document.createElement("span");
      seg.style.background = SPECTRUM_COLORS[i];
      spectrumTrack.appendChild(seg);

      const tick = document.createElement("span");
      tick.textContent = SEVERITY_SHORT[cls];
      spectrumTicks.appendChild(tick);
    });
  }

  function buildField(name) {
    const info = FIELD_META[name] || { group: "Other", label: humanize(name) };
    const row = document.createElement("div");
    row.className = "field-row";

    if (meta.categorical_fields[name]) {
      const options = meta.categorical_fields[name];
      row.innerHTML = `
        <label for="f-${name}">${info.label}</label>
        <select id="f-${name}" name="${name}" data-field="${name}">
          ${options.map((opt) => `<option value="${opt}">${humanize(opt)}</option>`).join("")}
        </select>
      `;
    } else {
      const range = meta.numeric_fields[name] || { min: 0, max: 10, step: 1, default: 5 };
      const unit = info.unit ? ` ${info.unit}` : "";
      row.innerHTML = `
        <label for="f-${name}">${info.label}
          <span class="field-value"><span id="v-${name}">${range.default}</span>${unit}</span>
        </label>
        <input type="range" id="f-${name}" name="${name}" data-field="${name}"
               min="${range.min}" max="${range.max}" step="${range.step}" value="${range.default}">
      `;
    }
    return row;
  }

  function renderForm() {
    fieldsContainer.innerHTML = "";

    const byGroup = {};
    meta.feature_order.forEach((name) => {
      const group = (FIELD_META[name] || {}).group || "Other";
      (byGroup[group] = byGroup[group] || []).push(name);
    });

    let delay = 0;
    GROUP_ORDER.forEach((group) => {
      const names = byGroup[group];
      if (!names || !names.length) return;

      const fieldset = document.createElement("fieldset");
      fieldset.style.animationDelay = `${delay}s`;
      delay += 0.08;

      const legend = document.createElement("legend");
      legend.textContent = group;
      fieldset.appendChild(legend);

      names.forEach((name) => fieldset.appendChild(buildField(name)));
      fieldsContainer.appendChild(fieldset);
    });

    fieldsContainer.addEventListener("input", (e) => {
      const field = e.target.dataset.field;
      if (field && e.target.type === "range") {
        const out = document.getElementById(`v-${field}`);
        if (out) out.textContent = e.target.value;
      }
    });
  }

  function collectPayload() {
    const payload = {};
    fieldsContainer.querySelectorAll("[data-field]").forEach((el) => {
      payload[el.dataset.field] = el.value;
    });
    return payload;
  }

  function setLoading(isLoading) {
    submitBtn.classList.toggle("loading", isLoading);
    submitBtn.disabled = isLoading;
  }

  function renderResult(data) {
    showState(resultState);

    resultClassEl.textContent = humanize(data.prediction);
    resultConfidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;

    const idx = SEVERITY_ORDER.indexOf(data.prediction);
    const position = idx >= 0 ? (idx + 0.5) / SEVERITY_ORDER.length : 0.5;
    // Animate from center, then settle into place on the next frame.
    spectrumMarker.style.left = "50%";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        spectrumMarker.style.left = `${position * 100}%`;
      });
    });

    const rows = Object.entries(data.probabilities)
      .sort((a, b) => SEVERITY_ORDER.indexOf(a[0]) - SEVERITY_ORDER.indexOf(b[0]));

    probRows.innerHTML = "";
    rows.forEach(([cls, prob], i) => {
      const sevIdx = SEVERITY_ORDER.indexOf(cls);
      const color = sevIdx >= 0 ? SPECTRUM_COLORS[sevIdx] : "var(--ink-soft)";
      const row = document.createElement("div");
      row.className = "prob-row" + (cls === data.prediction ? " is-winner" : "");
      row.innerHTML = `
        <span class="prob-name">${SEVERITY_SHORT[cls] || humanize(cls)}</span>
        <span class="prob-bar-track"><span class="prob-bar-fill" style="background:${color}"></span></span>
        <span class="prob-pct mono">${(prob * 100).toFixed(1)}%</span>
      `;
      probRows.appendChild(row);
      const fill = row.querySelector(".prob-bar-fill");
      setTimeout(() => { fill.style.width = `${prob * 100}%`; }, 30 + i * 60);
    });
  }

  function renderError(errBody) {
    showState(errorState);
    const parts = [];
    if (errBody.missing_fields && errBody.missing_fields.length) {
      parts.push(`Missing: ${errBody.missing_fields.map(humanize).join(", ")}`);
    }
    if (errBody.invalid_fields && errBody.invalid_fields.length) {
      errBody.invalid_fields.forEach((f) => parts.push(`${humanize(f.field)}: ${f.reason}`));
    }
    if (!parts.length) parts.push(errBody.error || "Unknown error.");
    errorMessage.textContent = parts.join(" — ");
  }

  async function init() {
    buildSpectrum();
    try {
      const res = await fetch("/api/meta");
      meta = await res.json();
      renderForm();

      if (meta.ready) {
        statusLine.textContent = "Model loaded — ready for assessment";
        statusLine.classList.add("ready");
        submitBtn.disabled = false;
      } else {
        statusLine.textContent = "Model artifacts not found on server — run train_model.py first";
        statusLine.classList.add("error");
        submitBtn.disabled = true;
      }
    } catch (err) {
      statusLine.textContent = "Could not reach the API";
      statusLine.classList.add("error");
      fieldsContainer.innerHTML = `<p class="loading-note">Could not load the intake form. Is the Flask server running?</p>`;
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectPayload()),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        renderResult(data);
      } else {
        renderError(data);
      }
    } catch (err) {
      renderError({ error: "Network error — could not reach the API." });
    } finally {
      setLoading(false);
    }
  });

  init();
})();
