"use strict";
(() => {
  const DATA = window.ACE_DATA || {};
  const R = DATA.results;
  const CASES = (DATA.cases && DATA.cases.cases) || [];
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hasIO = "IntersectionObserver" in window;
  function whenVisible(node, fn) {
    if (!hasIO || reduced) { fn(); return; }
    const io = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { io.disconnect(); fn(); }
    }, { threshold: 0.15 });
    io.observe(node);
  }

  // Colour theme: follows the system unless the visitor chooses one.
  const root = document.documentElement;
  $("#theme-toggle").addEventListener("click", () => {
    const current = root.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("ace-theme", next); } catch (error) { /* storage unavailable */ }
  });

  // Section navigation and reading progress.
  const navLinks = $$(".section-nav a");
  const sections = navLinks.map(link => $(link.getAttribute("href")));
  const menu = $("#menu-toggle");
  function closeMenu() {
    $("#section-nav").classList.remove("open");
    menu.setAttribute("aria-expanded", "false");
    menu.setAttribute("aria-label", "Open section navigation");
  }
  menu.addEventListener("click", () => {
    const open = $("#section-nav").classList.toggle("open");
    menu.setAttribute("aria-expanded", String(open));
    menu.setAttribute("aria-label", open ? "Close section navigation" : "Open section navigation");
  });
  navLinks.forEach(link => {
    link.setAttribute("aria-label", link.textContent.trim());
    link.addEventListener("click", closeMenu);
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && menu.getAttribute("aria-expanded") === "true") { closeMenu(); menu.focus(); }
  });
  let scheduled = false;
  function updateProgress() {
    const max = document.documentElement.scrollHeight - innerHeight;
    $("#reading-progress").style.width = `${max > 0 ? scrollY / max * 100 : 0}%`;
    let active = sections[0];
    sections.forEach(section => { if (section.getBoundingClientRect().top < innerHeight * .35) active = section; });
    navLinks.forEach(link => {
      const selected = link.hash === `#${active.id}`;
      link.classList.toggle("active", selected);
      if (selected) link.setAttribute("aria-current", "location"); else link.removeAttribute("aria-current");
    });
    scheduled = false;
  }
  addEventListener("scroll", () => { if (!scheduled) { scheduled = true; requestAnimationFrame(updateProgress); } }, { passive: true });
  addEventListener("resize", updateProgress);
  updateProgress();

  // Overview schematic: five queries on one execution.
  (() => {
    const overview = (DATA.figures && DATA.figures.overview) || { frames: [], queries: [] };
    const QA = overview.queries;
    if (!QA.length) return;
    const frames = $$("#tl-frames .tl-frame");
    const buttons = $$(".tl-queries button"), steps = $$(".tl-steps li"), pause = $("#tl-pause"), icon = $("#tl-pause-icon");
    let index = 0, timer = null, paused = reduced, visible = true;
    function show(i) {
      index = i;
      buttons.forEach((button, k) => button.setAttribute("aria-pressed", String(k === i)));
      steps.forEach((step, k) => step.classList.toggle("hot", QA[i].hot.includes(k)));
      frames.forEach((cell, k) => cell.classList.toggle("unseen", !QA[i].frames.includes(k)));
      $("#tl-sees").textContent = QA[i].sees;
      $("#tl-question").textContent = QA[i].q;
      $("#tl-answer").textContent = QA[i].a;
    }
    function sync() {
      clearInterval(timer); timer = null;
      if (!paused && visible) timer = setInterval(() => show((index + 1) % QA.length), 3200);
      pause.setAttribute("aria-pressed", String(paused));
      pause.setAttribute("aria-label", paused ? "Play the animation" : "Pause the animation");
      icon.setAttribute("d", paused ? "M8 5v14l11-7z" : "M7 5h4v14H7zM13 5h4v14h-4z");
    }
    buttons.forEach((button, k) => button.addEventListener("click", () => { paused = true; show(k); sync(); }));
    pause.addEventListener("click", () => { paused = !paused; sync(); });
    if (hasIO) new IntersectionObserver(entries => { visible = entries[0].isIntersecting; sync(); }).observe($("#timeline"));
    show(0); sync();
  })();

  // Accessible tabs: click, arrows, Home, and End.
  const tabs = $$("[role=tab]");
  function selectTab(tab, focus = false) {
    tabs.forEach(item => {
      const active = item === tab;
      item.setAttribute("aria-selected", String(active));
      item.tabIndex = active ? 0 : -1;
      $(`#${item.getAttribute("aria-controls")}`).hidden = !active;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); selectTab(tabs[next], true); }
    });
  });

  // Citation.
  $("#copy-citation").addEventListener("click", async () => {
    try {
      if (!navigator.clipboard || !window.isSecureContext) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText($("#bibtex").textContent);
      $("#copy-status").textContent = "BibTeX copied.";
    } catch (error) {
      const range = document.createRange();
      range.selectNodeContents($("#bibtex"));
      const selection = window.getSelection();
      selection.removeAllRanges(); selection.addRange(range);
      $("#copy-status").textContent = "Citation selected. Press Ctrl+C (or ⌘C on Mac) to copy.";
    }
  });

  /* ---------- Lightbox: one native modal dialog for every enlargeable figure ---------- */
  const dialog = $("#figure-dialog"), dialogImg = $("#dialog-img"), dialogScroll = $("#dialog-scroll"), dialogZoom = $("#dialog-zoom");
  let dialogTrigger = null;
  function setZoom(on) {
    dialogScroll.classList.toggle("zoomed", on);
    dialogZoom.setAttribute("aria-pressed", String(on));
    dialogZoom.textContent = on ? "Fit to window" : "Actual size";
  }
  let dialogSet = null, dialogIndex = 0;
  function showDialogImage(src, alt, title) {
    setZoom(false); dialogZoom.hidden = true;
    dialogImg.onload = () => { dialogZoom.hidden = dialogImg.naturalWidth <= dialogImg.clientWidth * 1.15; };
    dialogImg.alt = alt || ""; dialogImg.src = src;
    $("#dialog-title").textContent = title || "Figure";
  }
  function stepDialog(delta) {
    if (!dialogSet) return;
    dialogIndex = (dialogIndex + delta + dialogSet.length) % dialogSet.length;
    const item = dialogSet[dialogIndex];
    showDialogImage(item.src, item.alt, item.title);
  }
  // set: optional [{src, alt, title}] to step through with the arrow keys or the dialog buttons.
  function openFigure(src, alt, title, trigger, set, index) {
    dialogTrigger = trigger || null;
    dialogSet = set && set.length > 1 ? set : null; dialogIndex = index || 0;
    $("#dialog-step").hidden = !dialogSet;
    showDialogImage(src, alt, title);
    dialog.showModal();
    document.body.style.overflow = "hidden";
  }
  $("#dialog-prev").addEventListener("click", () => stepDialog(-1));
  $("#dialog-next").addEventListener("click", () => stepDialog(1));
  dialog.addEventListener("keydown", event => {
    if (event.key === "ArrowLeft") stepDialog(-1);
    if (event.key === "ArrowRight") stepDialog(1);
  });
  dialogZoom.addEventListener("click", () => setZoom(dialogZoom.getAttribute("aria-pressed") !== "true"));
  dialogImg.addEventListener("click", () => { if (!dialogZoom.hidden) setZoom(dialogZoom.getAttribute("aria-pressed") !== "true"); });
  $("#dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  dialog.addEventListener("close", () => {
    document.body.style.overflow = "";
    if (dialogTrigger && dialogTrigger.isConnected) dialogTrigger.focus({ preventScroll: true });
  });
  document.addEventListener("click", event => {
    const button = event.target.closest && event.target.closest("[data-zoom]");
    if (!button) return;
    const img = $("img", button);
    openFigure(button.dataset.zoom, img ? img.alt : "", button.dataset.title, button);
  });

  const FIG = DATA.figures || {};

  /* ---------- Pipeline figure: select a step to highlight its row ---------- */
  (() => {
    const figure = $("#pipeline-figure"), bands = FIG.pipeline && FIG.pipeline.bands;
    if (!figure || !bands) return;
    const buttons = $$("#pipeline-steps button"), top = $(".pipe-shade.top", figure), bottom = $(".pipe-shade.bottom", figure);
    let pinned = null, touched = false;
    function show(i) {
      figure.classList.toggle("focus", i !== null);
      if (i !== null) { top.style.height = `${bands[i][0] * 100}%`; bottom.style.top = `${bands[i][1] * 100}%`; }
      buttons.forEach((button, k) => { button.classList.toggle("active", k === i); button.setAttribute("aria-pressed", String(k === pinned)); });
    }
    buttons.forEach((button, k) => {
      button.addEventListener("click", () => { touched = true; pinned = pinned === k ? null : k; show(pinned); });
      button.addEventListener("pointerenter", event => { if (event.pointerType === "mouse") { touched = true; show(k); } });
      button.addEventListener("pointerleave", event => { if (event.pointerType === "mouse") show(pinned); });
    });
    show(null);
    // Walk through the four rows once when the figure is first seen.
    if (!reduced && hasIO) whenVisible(figure, () => {
      let k = 0;
      const next = () => { if (touched) return; if (k >= bands.length) { show(null); return; } show(k++); setTimeout(next, 1150); };
      setTimeout(next, 500);
    });
  })();

  /* ---------- Case studies of the paper: step through the four cards ---------- */
  (() => {
    const cases = FIG.cases || [], host = $("#paper-cases");
    if (!host) return;
    if (!cases.length) { host.hidden = true; return; }
    const NAMES = { ours: "Ours", astra: "Astra", opus: "Opus" };
    let index = 0;
    const card = $("#case-card"), button = element("button", "fig-zoom"), img = element("img");
    button.type = "button"; img.loading = "lazy"; img.decoding = "async";
    const hint = element("span", "zoom-hint", "⤢ Enlarge"); hint.setAttribute("aria-hidden", "true");
    button.append(img, hint); card.append(button);
    const tabs = cases.map((item, k) => {
      const tab = element("button", "", `(${item.key}) ${item.capability}`);
      tab.type = "button"; tab.addEventListener("click", () => show(k));
      $("#case-tabs").append(tab);
      return tab;
    });
    function show(k) {
      index = (k + cases.length) % cases.length;
      const item = cases[index], title = `(${item.key}) ${item.capability} · ${item.benchmark}`;
      img.src = item.src; img.width = item.w; img.height = item.h; img.alt = item.alt;
      button.dataset.zoom = item.src; button.dataset.title = title; button.setAttribute("aria-label", `Enlarge: case study ${item.key}`);
      $("#case-kicker").textContent = `CASE STUDY (${item.key}) · ${item.capability.toUpperCase()}`;
      $("#case-title").textContent = `${item.type} · ${item.benchmark}`;
      const verdicts = $("#case-verdicts"); verdicts.replaceChildren();
      Object.keys(NAMES).forEach(key => verdicts.append(element("span", `badge ${item.verdict[key] ? "good" : "bad"}`, `${item.verdict[key] ? "✓" : "✗"} ${NAMES[key]}`)));
      $("#case-why").textContent = item.why;
      $("#case-count").textContent = `${index + 1} / ${cases.length}`;
      tabs.forEach((tab, t) => tab.setAttribute("aria-pressed", String(t === index)));
    }
    $("#case-prev").addEventListener("click", () => show(index - 1));
    $("#case-next").addEventListener("click", () => show(index + 1));
    show(0);
  })();

  if (!R) return;

  /* ---------- Wordmark: the model's name as its logotype, with the name kept as text for assistive technology ---------- */
  function wordmark(animate) {
    const NS = "http://www.w3.org/2000/svg", wrap = element("span", "wm-wrap"), svg = document.createElementNS(NS, "svg"), use = document.createElementNS(NS, "use");
    svg.setAttribute("class", "wm"); svg.setAttribute("viewBox", "0 0 1691.4 143.9"); svg.setAttribute("aria-hidden", "true"); svg.setAttribute("focusable", "false");
    use.setAttribute("href", "#wm"); svg.append(use);
    wrap.append(svg, element("span", "visually-hidden", R.series.ours));
    if (animate && !reduced && hasIO) {
      wrap.classList.add("pending");
      whenVisible(wrap, () => { wrap.classList.remove("pending"); wrap.classList.add("sweep"); });
    }
    return wrap;
  }

  /* ---------- Emphasis: gains and key numbers in text that comes from the data ---------- */
  // Wraps each standalone number of a string in <strong class="num-em">; signed numbers get the "delta" variant.
  // Digits inside model names (Qwen3.5-9B, Claude Opus 5.5, ...) are left alone. The text itself is not changed.
  const MODEL_NAMES = [...new Set([...Object.values(R.series), ...R.task.models.map(model => model.name), ...R.grounding.models.map(model => model.name)])]
    .sort((a, b) => b.length - a.length);
  const NUMBER = /(^|[\s(])([+\u2212-]?\d[\d,]*(?:\.\d+)?(?:[Mk%])?)(?=$|[\s.,;:)])/g;
  function emphasize(node, text, quiet) {
    node.replaceChildren();
    const plain = part => {
      let last = 0;
      part.replace(NUMBER, (match, lead, token, offset) => {
        const start = offset + lead.length;
        node.append(part.slice(last, start));
        const strong = element("strong", "num-em", token);
        if (/^[+\u2212-]/.test(token)) strong.classList.add("delta");
        if (quiet) strong.classList.add("quiet");
        node.append(strong);
        last = start + token.length;
        return match;
      });
      node.append(part.slice(last));
    };
    let rest = text;
    while (rest) {
      let at = -1, hit = "";
      MODEL_NAMES.forEach(name => { const i = rest.indexOf(name); if (i >= 0 && (at < 0 || i < at)) { at = i; hit = name; } });
      if (at < 0) { plain(rest); break; }
      plain(rest.slice(0, at)); node.append(hit);
      rest = rest.slice(at + hit.length);
    }
    return node;
  }

  /* ---------- Numbers ---------- */
  const fmt = (value, digits) => (value === null || value === undefined) ? "–" : value.toFixed(digits);
  const digitsFor = (suite, column) => column === "Progress MAE" ? 3 : suite === "task" ? 1 : 2;
  const lowerIsBetter = (suite, column) => R[suite].lower_is_better.includes(column);
  function score(suite, model, column) {
    const table = R[suite], row = table.models.find(item => item.name === model);
    return row ? row.scores[table.columns.indexOf(column)] : null;
  }
  $$("[data-bind]").forEach(node => {
    const [suite, model, column] = node.dataset.bind.split("|");
    node.textContent = fmt(score(suite, model, column), digitsFor(suite, column));
    if (model === R.series.ours) node.classList.add("num-em");
  });

  // Headline metrics, counted up once when first seen.
  R.headline.forEach(item => {
    const box = element("div"), strong = element("strong"), number = document.createTextNode(fmt(item.value, item.decimals));
    strong.append(number);
    if (item.suffix) strong.append(element("span", "", item.suffix));
    if (/Avg\d/.test(item.label)) box.append(wordmark(true));
    box.append(strong, element("span", "", item.label), emphasize(element("small"), item.small));
    $("#metrics").append(box);
    if (reduced || !hasIO) return;
    whenVisible(box, () => {
      const start = performance.now(), duration = 900;
      const tick = now => {
        const p = Math.min(1, (now - start) / duration), eased = 1 - Math.pow(1 - p, 3);
        number.nodeValue = fmt(item.value * eased, item.decimals);
        if (p < 1) requestAnimationFrame(tick); else number.nodeValue = fmt(item.value, item.decimals);
      };
      requestAnimationFrame(tick);
    });
  });

  /* ---------- Bar charts ---------- */
  function rowClass(name) {
    if (name === R.series.ours) return "ours";
    if (name === R.series.astra) return "astra";
    if (name === R.series.opus) return "opus";
    return "";
  }
  // rows: [{name, label?, value, text}]; widths animate the first time the chart is seen.
  function drawBars(host, rows, max) {
    const first = !host.dataset.drawn;
    host.dataset.drawn = "1";
    host.replaceChildren();
    const fills = [];
    rows.forEach(item => {
      const row = element("div", `bar-row ${rowClass(item.name)}`);
      const track = element("span", "bar-track"), fill = element("span", "bar-fill");
      track.setAttribute("aria-hidden", "true");
      track.append(fill);
      const name = element("span", "bar-name");
      if (item.name === R.series.ours && (item.label || item.name) === R.series.ours) name.append(wordmark(first)); else name.textContent = item.label || item.name;
      row.append(name, track, /^[+\u2212-]/.test(item.text) ? emphasize(element("span", "bar-value"), item.text) : element("span", "bar-value", item.text));
      host.append(row);
      fills.push([fill, Math.max(0, Math.min(100, item.value / max * 100))]);
    });
    const grow = () => fills.forEach(([fill, width]) => { fill.style.width = `${width}%`; });
    if (first) whenVisible(host, () => requestAnimationFrame(() => requestAnimationFrame(grow)));
    else requestAnimationFrame(grow);
  }
  function setupChart(suite, averageColumn, averageTitle) {
    const table = R[suite], select = $(`#${suite}-select`);
    const leads = leadColumns(suite).map(item => item.column).filter(column => column !== averageColumn);
    [averageColumn, ...leads, ...table.columns.filter(column => column !== averageColumn && !leads.includes(column))].forEach(column => {
      const option = element("option", "", column); option.value = column; select.append(option);
    });
    function render() {
      const column = select.value, index = table.columns.indexOf(column), low = lowerIsBetter(suite, column), digits = digitsFor(suite, column);
      const rows = R.charts[suite].map(name => table.models.find(model => model.name === name))
        .filter(model => model && model.scores[index] !== null)
        .map(model => ({ name: model.name, label: model.base ? `${model.name} (base)` : model.name, value: model.scores[index], text: fmt(model.scores[index], digits) }))
        .sort((a, b) => low ? a.value - b.value : b.value - a.value);
      const max = low ? Math.max(...rows.map(row => row.value)) * 1.05 : 100;
      $(`#${suite}-chart-title`).textContent = column === averageColumn ? averageTitle : column;
      drawBars($(`#${suite}-chart`), rows, max);
      $(`#${suite}-chart-note`).textContent = column === averageColumn
        ? `${averageColumn} is the unweighted mean of the ${suite === "task" ? "six task-reasoning" : "seven public spatial-grounding"} benchmark scores. Higher is better.`
        : low ? `${column} is a mean absolute error on a 200-question set: shorter bars are better. The complete comparison is in the table below.`
          : `${column} uses its benchmark-specific metric. All bars share a 0–100 scale. The complete comparison is in the table below.`;
    }
    select.addEventListener("change", render);
    render();
  }
  // Columns on which ACE-Brain-0.7 has the best score of all compared models, computed from the table itself.
  function leadColumns(suite) {
    const table = R[suite], ours = table.models.find(model => model.name === R.series.ours);
    return table.columns.map((column, i) => {
      const low = lowerIsBetter(suite, column);
      const others = table.models.filter(model => model !== ours && model.scores[i] !== null)
        .sort((a, b) => low ? a.scores[i] - b.scores[i] : b.scores[i] - a.scores[i]);
      const first = low ? ours.scores[i] < others[0].scores[i] : ours.scores[i] > others[0].scores[i];
      return first ? { column, value: ours.scores[i], next: others[0], nextValue: others[0].scores[i], low, reported: others.length + 1 } : null;
    }).filter(Boolean);
  }
  function drawLeads(suite, averageColumn, extras, host, sentence) {
    const table = R[suite], leads = leadColumns(suite);
    const benchmarks = table.columns.filter(column => column !== averageColumn && !extras.includes(column));
    const won = leads.filter(item => benchmarks.includes(item.column));
    const order = [...leads.filter(item => item.column === averageColumn), ...leads.filter(item => extras.includes(item.column)), ...won];
    order.forEach(item => {
      const box = element("div"), digits = digitsFor(suite, item.column);
      box.append(wordmark(true), element("span", "lead-label", item.column), element("strong", "", fmt(item.value, digits)),
        emphasize(element("small"), `${item.low ? "Lowest" : "Highest"} · next: ${fmt(item.nextValue, digits)} · ${item.next.name}`, true));
      host.append(box);
    });
    const parts = [];
    if (leads.some(item => item.column === averageColumn)) parts.push(`the highest ${averageColumn} of the ${table.models.length} compared models`);
    if (won.length) parts.push(`the best score on ${won.length} of ${benchmarks.length} benchmarks (${won.map(item => item.column).join(", ")})`);
    // A column that some models do not report is claimed only among the models that do.
    leads.filter(item => extras.includes(item.column)).forEach(item => parts.push(`the ${item.low ? "lowest" : "best"} ${item.column}` +
      (item.reported < table.models.length ? ` of the ${item.reported} models that report it` : "")));
    sentence.textContent = `${sentence.textContent} ${R.series.ours} has ${parts.join(", ").replace(/, ([^,]*)$/, " and $1")}. Every score of every model is in the full table below.`.trim();
  }
  drawLeads("task", "Avg6", ["Progress MAE"], $("#task-leads"), $("#task-lead-text"));
  drawLeads("grounding", "Avg7", ["RefCOCOg"], $("#grounding-leads"), $("#grounding-lead-text"));
  setupChart("task", "Avg6", "Six-benchmark average");
  setupChart("grounding", "Avg7", "Seven-benchmark average");

  /* ---------- Sortable tables ---------- */
  function setupTable(suite, averageColumn) {
    const table = R[suite], host = $(`#${suite}-table`);
    const state = { filter: "all", column: table.columns.indexOf(averageColumn), ascending: false };
    const best = table.columns.map((column, i) => {
      const values = table.models.map(model => model.scores[i]).filter(value => value !== null);
      return lowerIsBetter(suite, column) ? Math.min(...values) : Math.max(...values);
    });
    const head = element("tr"), nameHeader = element("th", "", "Model");
    nameHeader.scope = "col"; head.append(nameHeader);
    const buttons = table.columns.map((column, i) => {
      const header = element("th"); header.scope = "col";
      const button = element("button", "", column); button.type = "button";
      button.addEventListener("click", () => {
        state.ascending = state.column === i ? !state.ascending : lowerIsBetter(suite, column);
        state.column = i; render();
      });
      header.append(button); head.append(header);
      return button;
    });
    $("thead", host).append(head);
    function render() {
      const models = table.models.filter(model => state.filter === "all" ||
        (state.filter === "closed" ? model.group === "Closed" : model.group !== "Closed"));
      models.sort((a, b) => {
        const x = a.scores[state.column], y = b.scores[state.column];
        if (x === null) return 1;
        if (y === null) return -1;
        return (state.ascending ? 1 : -1) * (x - y);
      });
      const body = $("tbody", host); body.replaceChildren();
      models.forEach(model => {
        const row = element("tr", model.group === "Ours" ? "ours" : "");
        const name = element("th"); name.scope = "row";
        if (model.group === "Ours") name.append(wordmark(!host.dataset.drawn)); else name.textContent = model.name;
        name.append(element("span", "model-category", model.group === "Ours" ? `Ours · ${R.model.params}` : model.base ? `${model.group} · base` : model.group));
        row.append(name);
        model.scores.forEach((value, i) => {
          const cell = element("td", table.columns[i] === averageColumn ? "avg" : "", fmt(value, digitsFor(suite, table.columns[i])));
          if (value !== null && value === best[i]) cell.classList.add("best");
          row.append(cell);
        });
        body.append(row);
      });
      host.dataset.drawn = "1";
      buttons.forEach((button, i) => {
        const active = i === state.column, column = table.columns[i];
        button.textContent = column + (active ? state.ascending ? " ↑" : " ↓" : "");
        button.parentElement.setAttribute("aria-sort", active ? state.ascending ? "ascending" : "descending" : "none");
        button.setAttribute("aria-label", `Sort by ${column}`);
      });
    }
    $$(`.filter-group[data-table="${suite}"] button`).forEach(button => button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      $$(`.filter-group[data-table="${suite}"] button`).forEach(item => item.setAttribute("aria-pressed", String(item === button)));
      render();
    }));
    render();
  }
  setupTable("task", "Avg6");
  setupTable("grounding", "Avg7");
  if ($("#footnotes")) R.footnotes.forEach(text => $("#footnotes").append(element("li", "", text)));

  /* ---------- Capabilities and analysis ---------- */
  const gain = R.capability_gain;
  drawBars($("#gain-chart"), gain.values.map(([name, value]) => ({ name: R.series.ours, label: name, value, text: `+${value}` })), 40);
  $("#gain-note").textContent = gain.note;
  emphasize($("#gain-sentence"), "Averaged over the sub-items of a capability, ACE-Brain-0.7 gains " +
    gain.values.map(([name, value]) => `${value} points on ${name.toLowerCase()}`).join(", ").replace(/, ([^,]*)$/, " and $1") + " over its base model.");
  function subitemChart(host, prefix) {
    const item = R.subitems.find(entry => entry.item.indexOf(prefix) === 0);
    const rows = ["ours", "astra", "opus", "base"].map(key => ({
      name: R.series[key], label: key === "base" ? `${R.series[key]} (base)` : R.series[key], value: item[key], text: fmt(item[key], 1)
    })).sort((a, b) => b.value - a.value);
    drawBars(host, rows, 100);
  }
  subitemChart($("#analysis-chart-a"), "RoboBench Q2");
  subitemChart($("#analysis-chart-b"), "Cosmos-Reason1: AgiBot");

  /* ---------- Data study ---------- */
  (() => {
    if (!$("#comp")) return;
    const study = R.data_study, composition = R.composition;
    $("#ego-share").textContent = study.ego_share;
    $("#ego-gain").textContent = `+${(study.avg6_with - study.avg6_without).toFixed(1)}`;
    $("#ego-detail").textContent = `Avg6 rises from ${study.avg6_without.toFixed(1)} to ${study.avg6_with.toFixed(1)}. The gain is concentrated on EgoPlan-Bench, whose questions are posed on egocentric video: ${study.egoplan_with.toFixed(1)} against ${study.egoplan_without.toFixed(1)}.`;
    const totals = composition.capabilities.map((_, i) => composition.sources.reduce((sum, source) => sum + source.values[i], 0));
    const max = Math.max(...totals), bars = [];
    composition.capabilities.forEach((name, i) => {
      const row = element("div", "stack-row"), cell = element("span"), bar = element("span", "stack-bar");
      composition.sources.forEach((source, k) => {
        if (!source.values[i]) return;
        const part = element("i", `c${k + 1}`);
        part.style.width = `${source.values[i] / totals[i] * 100}%`;
        part.title = `${source.name}: ${source.values[i]}k`;
        bar.append(part);
      });
      cell.append(bar);
      row.append(element("span", "", name), cell, element("span", "stack-value", `${Math.round(totals[i]).toLocaleString("en-US")}k`));
      $("#comp").append(row);
      bars.push([bar, totals[i] / max * 100]);
    });
    whenVisible($("#comp"), () => requestAnimationFrame(() => requestAnimationFrame(() => bars.forEach(([bar, width]) => { bar.style.width = `${width}%`; }))));
    composition.sources.forEach((source, k) => {
      const item = element("span"); item.append(element("i", `sw c${k + 1}`), source.name); $("#comp-legend").append(item);
    });
    const caption = $("#comp-caption");
    caption.append(element("span", "", "THE COLLECTION"), ` ${composition.total.toLocaleString("en-US")} ${composition.unit}, by capability and source.`);
  })();

  /* ---------- Paper figures as native charts ---------- */
  const SVGNS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs, parent, text) {
    const node = document.createElementNS(SVGNS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  }
  const SERIES = {
    ours: R.series.ours, astra: R.series.astra, opus: R.series.opus, base: `${R.series.base} (base)`,
    baseline: "Uniform pick among the wrong options"
  };
  function legend(host, keys) {
    keys.forEach(key => { const item = element("span"); item.append(element("i", `sw s-${key}`), key === "ours" ? wordmark(true) : SERIES[key]); host.append(item); });
  }
  // One tooltip for every chart. It repeats values that are also in the labels, the readout or the tables.
  const tip = element("div", "chart-tip"); tip.hidden = true; tip.setAttribute("aria-hidden", "true"); document.body.append(tip);
  let tipClear = null;
  function hideTip() { tip.hidden = true; if (tipClear) { tipClear(); tipClear = null; } }
  function showTip(target, title, rows, point) {
    tip.replaceChildren(element("strong", "tip-title", title));
    rows.forEach(row => {
      const line = element("span", `tip-row${row.best ? " best" : ""}`);
      line.append(element("i", `key s-${row.key}`), element("b", "", row.text), element("span", "", row.name));
      tip.append(line);
    });
    tip.hidden = false;
    const box = target.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    const x = point ? point.x : box.left + box.width / 2, y = point ? point.y : box.top;
    let top = y - h - 14;
    if (top < 8) top = Math.min(innerHeight - h - 8, (point ? point.y : box.bottom) + 18);
    tip.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, x - w / 2))}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }
  function hover(node, title, rows, onEnter, onLeave) {
    const enter = event => {
      if (tipClear && tipClear !== onLeave) tipClear();
      tipClear = onLeave || null;
      if (onEnter) onEnter();
      showTip(node, title, rows, event.type === "focus" ? null : { x: event.clientX, y: event.clientY });
    };
    const leave = event => { if (event.pointerType === "touch") return; hideTip(); };
    node.addEventListener("pointerenter", enter); node.addEventListener("pointermove", enter);
    node.addEventListener("pointerleave", leave); node.addEventListener("focus", enter); node.addEventListener("blur", hideTip);
  }
  addEventListener("scroll", hideTip, { passive: true });
  document.addEventListener("pointerdown", event => { if (!event.target.closest || !event.target.closest(".hit")) hideTip(); });
  document.addEventListener("keydown", event => { if (event.key === "Escape") hideTip(); });

  // Charts are drawn at the pixel size of their container, so text keeps its size on a phone.
  function responsive(host, draw) {
    let width = 0, shown = false;
    const run = () => {
      const w = Math.round(host.clientWidth);
      if (!w || w === width) return;
      width = w;
      const animate = !shown && !reduced && hasIO;
      shown = true;
      draw(w, animate);
    };
    if ("ResizeObserver" in window) new ResizeObserver(run).observe(host); else addEventListener("resize", run);
    run();
  }
  const rowsFor = (keys, values, digits) => {
    const best = Math.max(...keys.filter(key => key !== "baseline").map(key => values[key]));
    return keys.map(key => ({ key, name: SERIES[key], text: values[key].toFixed(digits), best: key !== "baseline" && values[key] === best }));
  };

  // Vertical grouped bars (error structure).
  function groupedBars(host, spec, keys) {
    responsive(host, (W, animate) => {
      host.replaceChildren();
      const H = W < 420 ? 286 : 316, m = { l: 36, r: 4, t: 40, b: spec.axis ? 68 : 48 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b, floor = m.t + ih, y = value => floor - value / spec.max * ih;
      const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": `${spec.title}. ${spec.unit}.` }, host);
      const pattern = svgEl("pattern", { id: `${host.id}-hatch`, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, svgEl("defs", {}, svg));
      svgEl("rect", { width: 6, height: 6, class: "hatch-bg" }, pattern);
      svgEl("rect", { width: 2.2, height: 6, class: "hatch-line" }, pattern);
      spec.ticks.forEach(tick => {
        svgEl("line", { x1: m.l, x2: W - m.r, y1: y(tick), y2: y(tick), class: tick ? "grid" : "axis" }, svg);
        svgEl("text", { x: m.l - 7, y: y(tick) + 4.5, "text-anchor": "end", class: "tick" }, svg, String(tick));
      });
      const band = iw / spec.groups.length, gap = 2, gw = Math.min(band * 0.84, keys.length * 46), bw = (gw - gap * (keys.length - 1)) / keys.length;
      const flat = bw >= 31, groups = [];
      spec.groups.forEach((group, gi) => {
        const g = svgEl("g", { class: "bar-group hit", tabindex: 0, role: "img",
          "aria-label": `${group.name}, ${group.sub}: ${keys.map(key => `${SERIES[key]} ${group[key].toFixed(1)}`).join(", ")}` }, svg);
        const x0 = m.l + gi * band + (band - gw) / 2;
        keys.forEach((key, ki) => {
          const x = x0 + ki * (bw + gap), top = y(group[key]), r = Math.min(3, bw / 2, floor - top);
          const bar = svgEl("path", { class: `mark s-${key}${animate ? " grow" : ""}`,
            d: `M${x},${floor}V${top + r}Q${x},${top} ${x + r},${top}H${x + bw - r}Q${x + bw},${top} ${x + bw},${top + r}V${floor}Z` }, g);
          if (key === "baseline") bar.style.fill = `url(#${host.id}-hatch)`;
          const cx = x + bw / 2, cls = `bar-label${key === "ours" ? " ours" : ""}${animate ? " fade" : ""}`;
          if (flat) svgEl("text", { x: cx, y: top - 6, "text-anchor": "middle", class: cls }, g, group[key].toFixed(1));
          else svgEl("text", { x: cx + 4.5, y: top - 5, transform: `rotate(-90 ${cx + 4.5} ${top - 5})`, class: cls }, g, group[key].toFixed(1));
        });
        const mid = m.l + (gi + 0.5) * band;
        svgEl("text", { x: mid, y: floor + 19, "text-anchor": "middle", class: "group-name" }, g, group.name);
        svgEl("text", { x: mid, y: floor + 37, "text-anchor": "middle" }, g, group.sub);
        svgEl("rect", { x: m.l + gi * band, y: 2, width: band, height: floor + 42, fill: "transparent" }, g);
        hover(g, `${group.name} · ${group.sub}`, rowsFor(keys, group, 1),
          () => groups.forEach(other => other.classList.toggle("dim", other !== g)),
          () => groups.forEach(other => other.classList.remove("dim")));
        groups.push(g);
      });
      if (spec.axis) svgEl("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, svg, spec.axis);
      if (animate) whenVisible(host, () => requestAnimationFrame(() => requestAnimationFrame(() => svg.classList.add("in"))));
    });
  }

  // Radial bars (capability profile): one group of four bars per sub-item, sectors by capability.
  function radialProfile(host, readout, labels) {
    const items = R.subitems, keys = ["ours", "astra", "opus", "base"], caps = [];
    items.forEach((item, i) => {
      const last = caps[caps.length - 1];
      if (last && last.name === item.capability) last.n += 1; else caps.push({ name: item.capability, start: i, n: 1 });
    });
    const hint = readout.textContent;
    responsive(host, (W, animate) => {
      host.replaceChildren();
      const size = Math.min(W, 620), wide = size >= 540, c = size / 2;
      const rArc = c - 22, r1 = wide ? c - 154 : c - 34, r0 = r1 * 0.42, rOut = wide ? rArc - 8 : r1 + 6, step = Math.PI * 2 / items.length;
      const n = value => value.toFixed(2), pt = (r, a) => [c + r * Math.sin(a), c - r * Math.cos(a)];
      const arc = (r, a0, a1) => { const [x0, y0] = pt(r, a0), [x1, y1] = pt(r, a1); return `M${n(x0)},${n(y0)}A${n(r)},${n(r)} 0 0 ${a1 > a0 ? 1 : 0} ${n(x1)},${n(y1)}`; };
      const wedge = (ra, rb, a0, a1) => {
        const [x0, y0] = pt(ra, a0), [x1, y1] = pt(rb, a0), [x2, y2] = pt(rb, a1), [x3, y3] = pt(ra, a1);
        return `M${n(x0)},${n(y0)}L${n(x1)},${n(y1)}A${n(rb)},${n(rb)} 0 0 1 ${n(x2)},${n(y2)}L${n(x3)},${n(y3)}A${n(ra)},${n(ra)} 0 0 0 ${n(x0)},${n(y0)}Z`;
      };
      const svg = svgEl("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}`, role: "group",
        "aria-label": "Scores of four models on seventeen sub-items, grouped by capability" }, host);
      const defs = svgEl("defs", {}, svg);
      caps.forEach((cap, k) => {
        const a0 = cap.start * step, a1 = (cap.start + cap.n) * step, mid = (a0 + a1) / 2, pad = 0.03;
        svgEl("path", { d: wedge(r0, rOut, a0, a1), class: `sector sector-${k % 2}` }, svg);
        svgEl("path", { d: arc(rArc, a0 + pad, a1 - pad), class: "cap-arc" }, svg);
        const flip = mid > Math.PI / 2 && mid < Math.PI * 1.5, id = `${host.id}-cap-${k}`;
        svgEl("path", { id, d: flip ? arc(rArc + 17, a1, a0) : arc(rArc + 6, a0, a1) }, defs);
        const text = svgEl("text", { class: "cap-name" }, svg);
        svgEl("textPath", { href: `#${id}`, startOffset: "50%", "text-anchor": "middle" }, text, cap.name);
      });
      [[0, "ring"], [50, "ring half"], [100, "ring"]].forEach(([value, cls]) => svgEl("circle", { cx: c, cy: c, r: r0 + (r1 - r0) * value / 100, class: cls }, svg));
      const bars = [], groups = [];
      items.forEach((item, i) => {
        const a = (i + 0.5) * step, label = labels[i] || [item.item];
        const g = svgEl("g", { class: "bar-group hit", tabindex: 0, role: "img",
          "aria-label": `${item.item} (${item.capability}): ${keys.map(key => `${SERIES[key]} ${item[key].toFixed(1)}`).join(", ")}` }, svg);
        const usable = step * 0.8, each = usable / keys.length, gapA = 1.1 / r0;
        keys.forEach((key, k) => {
          const a0 = i * step + step * 0.1 + k * each + gapA / 2, a1 = a0 + each - gapA;
          const bar = svgEl("path", { class: `mark s-${key}` }, g);
          bars.push(p => bar.setAttribute("d", wedge(r0, r0 + Math.max(0.01, (r1 - r0) * item[key] / 100 * p), a0, a1)));
        });
        if (wide) {
          const right = a < Math.PI, deg = a * 180 / Math.PI + (right ? -90 : 90), x = right ? r1 + 10 : -(r1 + 10);
          const text = svgEl("text", { transform: `translate(${c} ${c}) rotate(${n(deg)})`, "text-anchor": right ? "start" : "end", class: "item-label" }, g);
          label.forEach((line, k) => svgEl("tspan", { x, y: label.length === 1 ? 4.5 : k ? 12.5 : -3 }, text, line));
        }
        svgEl("path", { d: wedge(r0, rOut, i * step, (i + 1) * step), fill: "transparent" }, g);
        const rows = rowsFor(keys, item, 1);
        hover(g, item.item, rows, () => {
          groups.forEach(other => other.classList.toggle("dim", other !== g));
          readout.textContent = `${item.item} (${item.capability}) · ${rows.map(row => `${row.name} ${row.text}`).join(" · ")}`;
        }, () => { groups.forEach(other => other.classList.remove("dim")); readout.textContent = hint; });
        groups.push(g);
      });
      [[50, 4.5], [100, -5]].forEach(([value, dy]) => svgEl("text", { x: c, y: c - (r0 + (r1 - r0) * value / 100) + dy, "text-anchor": "middle", class: "ring-label" }, svg, String(value)));
      const wmW = r0 * 1.62, wmH = wmW * 143.9 / 1691.4;
      svgEl("use", { href: "#wm", x: n(c - wmW / 2), y: n(c - wmH / 2), width: n(wmW), height: n(wmH), class: "center-wm" }, svg);
      const paint = p => bars.forEach(fn => fn(p));
      if (!animate) { paint(1); return; }
      paint(0);
      whenVisible(host, () => {
        const start = performance.now(), duration = 850;
        const tick = now => { const p = Math.min(1, (now - start) / duration); paint(1 - Math.pow(1 - p, 3)); if (p < 1) requestAnimationFrame(tick); };
        requestAnimationFrame(tick);
      });
    });
  }

  // Horizontal grouped bars (capability averages).
  function averageBars(host, groups, keys) {
    const fills = [];
    groups.forEach(group => {
      const rows = rowsFor(keys, group, 1), box = element("div", `gb-group hit${group.summary ? " summary" : ""}`), bars = element("div", "gb-bars");
      box.tabIndex = 0; box.setAttribute("role", "img");
      box.setAttribute("aria-label", `${group.name}: ${rows.map(row => `${row.name} ${row.text}`).join(", ")}`);
      rows.forEach(row => {
        const line = element("div", `gb-row s-${row.key}`), track = element("span", "gb-track"), fill = element("span", "gb-fill");
        track.append(fill);
        line.append(track, element("span", `bar-value${row.best ? " best-value" : ""}`, row.text));
        bars.append(line);
        fills.push([fill, group[row.key]]);
      });
      box.append(element("span", "gb-name", group.name), bars);
      hover(box, group.name, rows);
      host.append(box);
    });
    whenVisible(host, () => requestAnimationFrame(() => requestAnimationFrame(() => fills.forEach(([fill, value]) => { fill.style.width = `${value}%`; }))));
  }

  if (FIG.profile && $("#profile-chart")) {
    const keys = ["ours", "astra", "opus", "base"];
    legend($("#profile-legend"), keys);
    radialProfile($("#profile-chart"), $("#profile-readout"), FIG.profile.labels || []);
    averageBars($("#profile-averages"), FIG.profile.averages, keys);
  }
  if (FIG.errors && $("#errors-chart-a")) {
    const a = FIG.errors.wrong_picks, b = FIG.errors.plan_length;
    $("#errors-title-a").textContent = `(a) ${a.title}`;
    legend($("#errors-legend-a"), ["baseline", "ours", "astra", "opus"]);
    groupedBars($("#errors-chart-a"), a, ["baseline", "ours", "astra", "opus"]);
    $("#errors-note-a").textContent = `${a.unit}, by position of the wrong pick relative to the correct step on the timeline of the recording. ${a.takeaway}`;
    if (b) {
      $("#errors-title-b").textContent = `(b) ${b.title}`;
      legend($("#errors-legend-b"), ["ours", "astra", "opus"]);
      groupedBars($("#errors-chart-b"), b, ["ours", "astra", "opus"]);
      $("#errors-note-b").textContent = `${b.unit} on task decomposition, by the number of steps in the reference plan.`;
    } else $("#errors-panel-b").hidden = true;
  }

  /* ---------- Example browser ---------- */
  (() => {
    if (!CASES.length) return;
    let currentIndex = 0;
    const groups = [...new Set(CASES.map(example => example.capability))];
    const benchmarks = [...new Set(CASES.map(example => example.benchmark))];
    $("#example-facts").textContent = `${CASES.length} items · ${groups.length} capabilities · ${benchmarks.length} public benchmarks · saved answers of 3 models`;
    if ($("#examples-rule")) $("#examples-rule").textContent = DATA.cases.note || "";
    $("#example-disclaimer").textContent = "✓ / ✗: the answer matches the reference or not. A number is the judge score out of 100. Answers are saved evaluation outputs, shown as written.";
    groups.forEach(group => {
      const button = element("button", "", group);
      button.type = "button"; button.dataset.group = group;
      button.addEventListener("click", () => { currentIndex = CASES.findIndex(example => example.capability === group); renderExample(); });
      $(".gallery-groups").append(button);
    });
    CASES.forEach((example, index) => {
      const option = element("option", "", `${String(index + 1).padStart(2, "0")} · ${example.capability} · ${example.type}`);
      option.value = index; $("#example-select").append(option);
    });
    function badge(answer) {
      const judged = !("correct" in answer), good = judged ? answer.score >= 100 : answer.correct;
      const mark = element("span", `badge ${good ? "good" : judged ? "plain" : "bad"}`, judged ? String(Math.round(answer.score)) : good ? "✓" : "✗");
      mark.setAttribute("role", "img");
      mark.setAttribute("aria-label", judged ? `Judge score ${Math.round(answer.score)} of 100` : good ? "Correct" : "Incorrect");
      return mark;
    }
    // Long text is clamped; a button appears only where something is cut off.
    function clamp(node, label) {
      node.classList.add("clamp");
      if (node.scrollHeight <= node.clientHeight + 3) return;
      const button = element("button", "more", `Show full ${label}`); button.type = "button";
      button.setAttribute("aria-expanded", "false");
      button.addEventListener("click", () => {
        const open = node.classList.toggle("open");
        button.textContent = open ? "Show less" : `Show full ${label}`;
        button.setAttribute("aria-expanded", String(open));
      });
      node.after(button);
    }
    function renderExample() {
      const example = CASES[currentIndex], count = example.frames.length;
      $("#example-position").textContent = `${currentIndex + 1} / ${CASES.length}`;
      $("#example-select").value = currentIndex;
      $("#example-kicker").textContent = `${example.capability} · ${example.benchmark}`.toUpperCase();
      $("#example-title").textContent = example.type;
      $$(".ex-copy .more").forEach(node => node.remove());
      const question = $("#example-question"), reference = $("#example-answer");
      question.classList.remove("open"); reference.classList.remove("open");
      question.textContent = example.question;
      reference.textContent = example.reference;
      $$("[data-group]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.group === example.capability)));
      const answers = $("#model-answers"); answers.replaceChildren();
      const texts = [];
      [["ours", R.series.ours], ["astra", R.series.astra], ["opus", R.series.opus]].forEach(([key, name]) => {
        const answer = example.answers[key], row = element("div", `ex-row${key === "ours" ? " ours" : ""}`), label = element("span", "ex-label");
        const text = element("p", "ex-text", answer.text);
        label.append(badge(answer), key === "ours" ? wordmark(false) : name);
        row.append(label, text);
        answers.append(row);
        texts.push(text);
      });
      // Frames: a tight strip, each frame opens the lightbox; the last frame is the current view.
      const visual = $("#example-visual"); visual.replaceChildren();
      const strip = element("div", `filmstrip n${Math.min(count, 5)}`);
      const set = example.frames.map((frame, k) => ({ src: frame.src, title: `${example.benchmark} · frame ${k + 1} of ${count}`,
        alt: `${example.benchmark} example, frame ${k + 1} of ${count} given to the models` }));
      example.frames.forEach((frame, k) => {
        const button = element("button", `film-frame${k === count - 1 ? " current" : ""}`), img = element("img");
        button.type = "button"; button.setAttribute("aria-label", `Enlarge frame ${k + 1} of ${count}`);
        img.src = frame.src; img.alt = set[k].alt; img.width = frame.w; img.height = frame.h; img.decoding = "async";
        button.append(img);
        if (count > 1) button.append(element("span", "film-index", k === count - 1 ? "Now" : String(k + 1)));
        button.addEventListener("click", () => openFigure(set[k].src, set[k].alt, set[k].title, button, set, k));
        strip.append(button);
      });
      visual.append(strip, element("p", "example-disclaimer",
        `${count > 1 ? `${count} frames in order; the last is the current view` : "The frame given to the models"} · select a frame to enlarge · Source: ${example.credit}.`));
      clamp(question, "question"); clamp(reference, "reference"); texts.forEach(text => clamp(text, "answer"));
    }
    const step = delta => { currentIndex = (currentIndex + delta + CASES.length) % CASES.length; renderExample(); };
    $("#example-select").addEventListener("change", event => { currentIndex = Number(event.target.value); renderExample(); });
    $("#example-next").addEventListener("click", () => step(1));
    $("#example-prev").addEventListener("click", () => step(-1));

    const match = /^#example=(\d+)$/.exec(location.hash);
    if (match && CASES[Number(match[1]) - 1]) { currentIndex = Number(match[1]) - 1; }
    renderExample();
    if (match) $("#examples").scrollIntoView();
  })();
})();

/* pending project links: do nothing until a URL is filled in */
document.querySelectorAll(".link-chip.is-pending").forEach(a => a.addEventListener("click", e => e.preventDefault()));
