/* turducken frontend — vanilla JS, no build step, no dependencies.
   The form is generated entirely from the API's annotated schema, so adding a
   knob in Python makes it appear here, fully documented, with no change to
   this file. */

const api = (path, opts) =>
  fetch(`/api${path}`, opts).then(async (r) => {
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(body.detail || r.statusText), { body });
    return body;
  });

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const state = { recipe: null, docs: {}, schema: null, lessonLinks: {}, tier: 'essential', job: null, cursor: 0, poll: null, losses: [] };

const TIER_RANK = { essential: 0, standard: 1, advanced: 2, experimental: 2 };

/* ── Tabs ─────────────────────────────────────────────────── */
document.querySelectorAll('.tab').forEach((tab) =>
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $(`#panel-${tab.dataset.panel}`).classList.add('active');
  })
);

document.querySelectorAll('.tier-btn').forEach((btn) =>
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tier-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.tier = btn.dataset.tier;
    renderForm();
  })
);

/* ── Recipe form ──────────────────────────────────────────── */
const GROUPS = [
  { key: '', title: 'Run' },
  { key: 'lora', title: 'Adapter — how much capacity you are adding' },
  { key: 'data', title: 'Data — what it learns from' },
  { key: 'training', title: 'Training — how it learns' },
];

function getPath(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
}
function setPath(obj, path, value) {
  const keys = path.split('.');
  const last = keys.pop();
  keys.reduce((o, k) => (o[k] ??= {}), obj)[last] = value;
}

function schemaFor(path) {
  // Resolve a dotted path against the JSON schema, following $refs.
  const defs = state.schema.$defs || {};
  let node = state.schema;
  for (const key of path.split('.')) {
    let prop = (node.properties || {})[key];
    if (!prop) return {};
    const ref = prop.$ref || (prop.allOf || []).find((a) => a.$ref)?.$ref;
    node = ref ? defs[ref.split('/').pop()] : prop;
    if (!ref) return prop;
  }
  return node;
}

function renderForm() {
  const form = $('#recipe-form');
  form.innerHTML = '';
  const limit = TIER_RANK[state.tier];

  for (const group of GROUPS) {
    const paths = Object.keys(state.docs).filter((p) => {
      const parent = p.includes('.') ? p.slice(0, p.lastIndexOf('.')) : '';
      return parent === group.key && TIER_RANK[state.docs[p].tier] <= limit;
    });
    if (!paths.length) continue;

    const section = el('div', 'knob-group');
    section.appendChild(el('h3', null, esc(group.title)));
    paths.forEach((p) => section.appendChild(knobCard(p)));
    form.appendChild(section);
  }
}

function knobCard(path) {
  const doc = state.docs[path];
  const spec = schemaFor(path);
  const value = getPath(state.recipe, path);
  const label = path.split('.').pop().replace(/_/g, ' ');

  const card = el('div', 'knob');
  const top = el('div', 'knob-top');

  const left = el('div');
  left.appendChild(
    el('div', 'knob-name', `${esc(label)}<span class="path">${esc(path)}</span>`)
  );
  left.appendChild(el('p', 'knob-plain', esc(doc.plain)));
  top.appendChild(left);

  const right = el('div', 'knob-control');
  right.appendChild(el('span', 'tier-badge', esc(doc.tier)));
  right.appendChild(inputFor(path, spec, value));
  if (doc.unit) right.appendChild(el('span', 'unit', esc(doc.unit)));
  top.appendChild(right);
  card.appendChild(top);

  // The detail drawer: closed by default so the form stays scannable, but one
  // click from every knob rather than buried in separate documentation.
  const details = el('details');
  details.appendChild(el('summary', null, 'Why this matters'));
  const body = el('div', 'knob-detail');
  body.appendChild(el('p', null, esc(doc.why)));
  if (doc.tradeoff) body.appendChild(el('p', null, `<strong>Trade-off.</strong> ${esc(doc.tradeoff)}`));
  if (doc.rule_of_thumb) body.appendChild(el('p', null, `<strong>Rule of thumb.</strong> ${esc(doc.rule_of_thumb)}`));
  for (const [when, what] of Object.entries(doc.symptoms || {})) {
    body.appendChild(el('div', 'symptom', `<strong>Set ${esc(when)}:</strong> ${esc(what)}`));
  }
  const slug = doc.lesson || state.lessonLinks[path];
  if (slug) {
    const link = el('a', 'lesson-link', `Read the lesson →`);
    link.href = '#';
    link.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelector('.tab[data-panel="learn"]').click();
      loadLesson(slug);
    });
    body.appendChild(link);
  }
  details.appendChild(body);
  card.appendChild(details);
  return card;
}

function inputFor(path, spec, value) {
  const enumValues = spec.enum || (spec.allOf || []).flatMap((a) => a.enum || []);

  if (typeof value === 'boolean') {
    const input = el('input');
    input.type = 'checkbox';
    input.checked = value;
    input.addEventListener('change', () => setPath(state.recipe, path, input.checked));
    return input;
  }
  if (enumValues.length) {
    const select = el('select');
    enumValues.forEach((v) => {
      const opt = el('option', null, esc(v));
      opt.value = v;
      if (v === value) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener('change', () => setPath(state.recipe, path, select.value));
    return select;
  }
  if (Array.isArray(value)) {
    const input = el('input');
    input.type = 'text';
    input.value = value.join(', ');
    input.addEventListener('change', () =>
      setPath(state.recipe, path, input.value.split(',').map((s) => s.trim()).filter(Boolean))
    );
    return input;
  }

  const input = el('input');
  const numeric = typeof value === 'number';
  input.type = numeric ? 'number' : 'text';
  if (numeric) {
    if (spec.minimum != null) input.min = spec.minimum;
    if (spec.maximum != null) input.max = spec.maximum;
    input.step = Number.isInteger(value) ? (spec.multipleOf || 1) : 'any';
  }
  input.value = value ?? '';
  input.addEventListener('change', () =>
    setPath(state.recipe, path, numeric ? Number(input.value) : input.value)
  );
  return input;
}

/* ── Dataset ──────────────────────────────────────────────── */
$('#analyze-btn').addEventListener('click', async () => {
  const target = $('#dataset-result');
  target.innerHTML = '<p class="empty">Scanning…</p>';
  try {
    const r = await api('/dataset/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipe: state.recipe }),
    });
    target.innerHTML = '';

    target.appendChild(
      cards([
        ['Images', r.count],
        ['Buckets', (r.buckets.buckets || []).length],
        ['Mean crop', `${((r.buckets.mean_cropped || 0) * 100).toFixed(1)}%`],
        ['Mean caption', `${r.captions.mean_words ?? 0} words`],
      ])
    );

    (r.warnings || []).forEach((w) =>
      target.appendChild(advisoryCard({ severity: 'warn', title: w, detail: '', field: 'dataset' }))
    );
    (r.rejected || []).forEach((x) =>
      target.appendChild(
        advisoryCard({ severity: 'info', title: `Skipped ${x.path}`, detail: x.reason, field: '' })
      )
    );

    if (r.buckets.buckets?.length) {
      target.appendChild(el('h3', null, 'Aspect buckets'));
      target.appendChild(
        table(['Bucket', 'Images'], r.buckets.buckets.map((b) => [b.size, b.count]))
      );
    }
    if (r.captions.frequent_tags?.length) {
      target.appendChild(el('h3', null, 'Most frequent tags'));
      target.appendChild(
        table(
          ['Tag', 'Count', 'Share of dataset'],
          r.captions.frequent_tags.map((t) => [t.tag, t.count, `${(t.share * 100).toFixed(0)}%`])
        )
      );
    }
    if (r.items?.length) {
      target.appendChild(el('h3', null, 'Images'));
      target.appendChild(
        table(
          ['File', 'Size', 'Bucket', 'Cropped', 'Issues'],
          r.items.map((i) => [
            i.image,
            `${i.width}×${i.height}`,
            i.bucket,
            `${(i.cropped_fraction * 100).toFixed(0)}%`,
            i.issues.join(' ') || '—',
          ])
        )
      );
    }
    if (!r.count) target.appendChild(el('p', 'empty', 'No usable image/caption pairs found.'));
  } catch (err) {
    target.innerHTML = `<p class="empty">${esc(err.message)}</p>`;
  }
});

/* ── Review ───────────────────────────────────────────────── */
$('#review-btn').addEventListener('click', async () => {
  const target = $('#review-result');
  target.innerHTML = '<p class="empty">Reviewing…</p>';
  try {
    const r = await api('/recipe/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipe: state.recipe }),
    });
    target.innerHTML = '';
    target.appendChild(
      cards([
        ['Est. VRAM', `${r.estimated_vram_gb} GB`],
        ['Available', r.available_vram_gb ? `${r.available_vram_gb} GB` : 'no GPU'],
        ['Effective batch', r.effective_batch_size],
        ['LoRA scale', r.lora_scale],
      ])
    );
    if (!r.advisories.length) {
      target.appendChild(el('p', null, 'Nothing to flag — this recipe looks reasonable.'));
    }
    r.advisories.forEach((a) => target.appendChild(advisoryCard(a)));
  } catch (err) {
    const detail = err.body?.detail;
    target.innerHTML = '';
    if (Array.isArray(detail)) {
      detail.forEach((d) =>
        target.appendChild(
          advisoryCard({ severity: 'blocker', title: d.message, detail: '', field: d.field })
        )
      );
    } else {
      target.innerHTML = `<p class="empty">${esc(err.message)}</p>`;
    }
  }
});

/* ── Training ─────────────────────────────────────────────── */
$('#start-btn').addEventListener('click', async () => {
  const log = $('#job-log');
  log.innerHTML = '';
  state.losses = [];
  state.cursor = 0;
  try {
    const job = await api('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recipe: state.recipe,
        backend: $('#backend-select').value,
        acknowledge_blockers: true,
      }),
    });
    state.job = job.id;
    $('#job-status').classList.remove('hidden');
    $('#cancel-btn').classList.remove('hidden');
    $('#start-btn').disabled = true;
    state.poll = setInterval(pollJob, 500);
  } catch (err) {
    log.appendChild(logLine('error', esc(err.message)));
  }
});

$('#cancel-btn').addEventListener('click', () => {
  if (state.job) api(`/jobs/${state.job}/cancel`, { method: 'POST' }).catch(() => {});
});

async function pollJob() {
  if (!state.job) return;
  try {
    const r = await api(`/jobs/${state.job}/events?cursor=${state.cursor}`);
    state.cursor = r.cursor;

    const log = $('#job-log');
    for (const e of r.events) {
      if (e.kind === 'step') {
        state.losses.push({ step: e.step, loss: e.loss });
      } else {
        log.appendChild(logLine(e.kind, esc(e.message || ''), e.seconds_elapsed));
      }
    }
    if (r.events.length) log.scrollTop = log.scrollHeight;

    const job = r.job;
    $('#progress-bar').style.width = `${(job.progress * 100).toFixed(1)}%`;
    $('#progress-label').textContent = `${(job.progress * 100).toFixed(0)}%`;
    $('#stat-step').textContent = `${job.step}/${job.total_steps}`;
    $('#stat-loss').textContent = job.last_loss?.toFixed(4) ?? '—';
    $('#stat-eta').textContent = job.eta_seconds != null ? `${Math.round(job.eta_seconds)}s` : '—';
    $('#stat-state').textContent = job.state;
    drawLoss();

    if (['done', 'failed', 'cancelled'].includes(job.state)) {
      clearInterval(state.poll);
      state.poll = null;
      $('#start-btn').disabled = false;
      $('#cancel-btn').classList.add('hidden');
    }
  } catch (err) {
    clearInterval(state.poll);
    state.poll = null;
    $('#start-btn').disabled = false;
  }
}

function logLine(kind, text, seconds) {
  const line = el('div', `log-line log-${kind}`);
  line.appendChild(el('span', 't', seconds != null ? `${seconds.toFixed(1)}s` : ''));
  line.appendChild(el('span', null, text));
  return line;
}

function drawLoss() {
  const canvas = $('#loss-chart');
  const ctx = canvas.getContext('2d');
  const w = (canvas.width = canvas.clientWidth * devicePixelRatio);
  const h = (canvas.height = 160 * devicePixelRatio);
  ctx.clearRect(0, 0, w, h);
  if (state.losses.length < 2) return;

  const pad = 24 * devicePixelRatio;
  const values = state.losses.map((p) => p.loss);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const range = hi - lo || 1;
  const x = (i) => pad + (i / (state.losses.length - 1)) * (w - pad * 2);
  const y = (v) => h - pad - ((v - lo) / range) * (h - pad * 2);

  ctx.strokeStyle = '#d98a3d';
  ctx.lineWidth = 1.5 * devicePixelRatio;
  ctx.beginPath();
  state.losses.forEach((p, i) => (i ? ctx.lineTo(x(i), y(p.loss)) : ctx.moveTo(x(i), y(p.loss))));
  ctx.stroke();

  // A trailing mean, because the raw curve's jitter hides the trend that
  // actually matters.
  const win = Math.max(2, Math.round(state.losses.length / 20));
  ctx.strokeStyle = '#4b9fd5';
  ctx.beginPath();
  state.losses.forEach((_, i) => {
    const slice = values.slice(Math.max(0, i - win), i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    i ? ctx.lineTo(x(i), y(mean)) : ctx.moveTo(x(i), y(mean));
  });
  ctx.stroke();

  ctx.fillStyle = '#8b98a8';
  ctx.font = `${11 * devicePixelRatio}px ui-monospace, monospace`;
  ctx.fillText(hi.toFixed(3), 4 * devicePixelRatio, pad);
  ctx.fillText(lo.toFixed(3), 4 * devicePixelRatio, h - pad + 10 * devicePixelRatio);
}

/* ── Learn ────────────────────────────────────────────────── */
async function loadLessons() {
  const { lessons } = await api('/lessons');
  const list = $('#lesson-list');
  list.innerHTML = '';
  lessons.forEach((l) => {
    const card = el('div', 'lesson-card');
    card.dataset.slug = l.slug;
    card.innerHTML = `<h4>${esc(l.title)}</h4><p>${esc(l.summary)}</p>
      <div class="meta">${l.minutes} min read${l.demos.length ? ` · ${l.demos.length} live demo${l.demos.length > 1 ? 's' : ''}` : ''}</div>`;
    card.addEventListener('click', () => loadLesson(l.slug));
    list.appendChild(card);
  });
}

async function loadLesson(slug) {
  document.querySelectorAll('.lesson-card').forEach((c) =>
    c.classList.toggle('active', c.dataset.slug === slug)
  );
  const body = $('#lesson-body');
  body.innerHTML = '<p class="empty">Loading…</p>';
  const lesson = await api(`/lessons/${slug}`);
  body.innerHTML = markdown(lesson.body);

  for (const name of lesson.demos || []) {
    const holder = el('div', 'demo');
    holder.innerHTML = '<p class="empty">Running…</p>';
    body.appendChild(holder);
    try {
      const d = await api(`/demos/${name}`);
      renderDemo(holder, name, d);
    } catch {
      holder.innerHTML = '<p class="empty">Demo unavailable.</p>';
    }
  }
  body.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderDemo(holder, name, d) {
  holder.innerHTML = `<h4>${esc(d.title)}</h4>`;

  if (d.structured_curve) {
    const canvas = el('canvas');
    canvas.height = 180;
    holder.appendChild(canvas);
    requestAnimationFrame(() =>
      lineChart(canvas, [
        { points: d.structured_curve.map((p) => [p.rank, p.energy_retained]), color: '#d98a3d', label: 'realistic update' },
        { points: d.noise_curve.map((p) => [p.rank, p.energy_retained]), color: '#4b9fd5', label: 'random noise' },
      ])
    );
  }
  if (d.points) {
    const canvas = el('canvas');
    canvas.height = 180;
    holder.appendChild(canvas);
    requestAnimationFrame(() =>
      lineChart(canvas, [
        { points: d.points.map((p) => [p.timestep, p.signal]), color: '#4fb477', label: 'signal remaining' },
        { points: d.points.map((p) => [p.timestep, p.noise]), color: '#d95c5c', label: 'noise added' },
      ])
    );
  }
  if (d.sweep) {
    holder.appendChild(
      table(['Rank', 'Parameters', 'Share of layer'], d.sweep.map((s) => [s.rank, s.params.toLocaleString(), `${(s.ratio * 100).toFixed(2)}%`]))
    );
  }
  if (d.rows) {
    holder.appendChild(
      table(
        ['Original', 'Bucket', 'Cropped', 'If square-only'],
        d.rows.map((r) => [r.original, r.bucket, `${(r.cropped * 100).toFixed(0)}%`, `${(r.cropped_if_square_only * 100).toFixed(0)}%`])
      )
    );
  }
  if (d.trace) {
    holder.appendChild(
      table(['Step', 'Correction magnitude'], d.trace.map((t) => [t.step, t.delta_norm.toFixed(6)]))
    );
  }
  if (d.takeaway) holder.appendChild(el('p', 'takeaway', esc(d.takeaway)));
}

function lineChart(canvas, series) {
  const ctx = canvas.getContext('2d');
  const w = (canvas.width = canvas.clientWidth * devicePixelRatio);
  const h = (canvas.height = 180 * devicePixelRatio);
  const pad = 30 * devicePixelRatio;
  ctx.clearRect(0, 0, w, h);

  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const px = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (w - pad * 2);
  const py = (v) => h - pad - ((v - y0) / (y1 - y0 || 1)) * (h - pad * 2);

  ctx.strokeStyle = '#2a323d';
  ctx.lineWidth = devicePixelRatio;
  ctx.strokeRect(pad, pad, w - pad * 2, h - pad * 2);

  ctx.font = `${11 * devicePixelRatio}px system-ui`;
  series.forEach((s, i) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath();
    s.points.forEach((p, j) => (j ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1]))));
    ctx.stroke();
    ctx.fillStyle = s.color;
    ctx.fillText(s.label, pad + 8 * devicePixelRatio, pad + (14 + i * 15) * devicePixelRatio);
  });

  // Axis extents. These charts exist to support a specific numeric claim
  // ("rank 8 retains 94%"), so an unlabelled curve would not do the job.
  const fmt = (v) => (Number.isInteger(v) ? v : v.toFixed(2));
  ctx.fillStyle = '#8b98a8';
  ctx.textAlign = 'right';
  ctx.fillText(fmt(y1), pad - 5 * devicePixelRatio, pad + 4 * devicePixelRatio);
  ctx.fillText(fmt(y0), pad - 5 * devicePixelRatio, h - pad + 4 * devicePixelRatio);
  ctx.textAlign = 'left';
  ctx.fillText(fmt(x0), pad, h - pad + 15 * devicePixelRatio);
  ctx.textAlign = 'right';
  ctx.fillText(fmt(x1), w - pad, h - pad + 15 * devicePixelRatio);
  ctx.textAlign = 'left';
}

/* ── Minimal markdown renderer ────────────────────────────── */
function markdown(src) {
  const lines = src.split('\n');
  const out = [];
  let inCode = false, inTable = false, listType = null;

  const closeList = () => { if (listType) { out.push(`</${listType}>`); listType = null; } };
  const closeTable = () => { if (inTable) { out.push('</tbody></table></div>'); inTable = false; } };

  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|\s)\*([^*]+)\*/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="#" data-lesson="$2">$1</a>');

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (line.startsWith('```')) {
      closeList(); closeTable();
      out.push(inCode ? '</code></pre>' : '<pre><code>');
      inCode = !inCode;
      continue;
    }
    if (inCode) { out.push(esc(raw)); continue; }

    if (/^\|/.test(line)) {
      const cells = line.split('|').slice(1, -1).map((c) => c.trim());
      if (/^[-: |]+$/.test(line)) continue;
      if (!inTable) {
        closeList();
        out.push('<div class="table-wrap"><table><thead><tr>' + cells.map((c) => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>');
        inTable = true;
      } else {
        out.push('<tr>' + cells.map((c) => `<td>${inline(c)}</td>`).join('') + '</tr>');
      }
      continue;
    }
    closeTable();

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) { closeList(); out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`); continue; }
    if (/^(-{3,}|\*{3,})$/.test(line)) { closeList(); out.push('<hr>'); continue; }

    const bullet = line.match(/^[-*]\s+(.*)$/);
    const numbered = line.match(/^\d+\.\s+(.*)$/);
    if (bullet || numbered) {
      const want = bullet ? 'ul' : 'ol';
      if (listType !== want) { closeList(); out.push(`<${want}>`); listType = want; }
      out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      continue;
    }
    closeList();

    if (!line.trim()) continue;
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList(); closeTable();
  if (inCode) out.push('</code></pre>');

  const html = out.join('\n');
  // Defer wiring of in-lesson links until after the HTML is in the DOM.
  queueMicrotask(() =>
    document.querySelectorAll('#lesson-body a[data-lesson]').forEach((a) =>
      a.addEventListener('click', (e) => { e.preventDefault(); loadLesson(a.dataset.lesson); })
    )
  );
  return html;
}

/* ── Helpers ──────────────────────────────────────────────── */
function cards(pairs) {
  const grid = el('div', 'summary-grid');
  pairs.forEach(([k, v]) =>
    grid.appendChild(el('div', 'summary-card', `<div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>`))
  );
  return grid;
}

function table(headers, rows) {
  const wrap = el('div', 'table-wrap');
  const t = el('table');
  t.innerHTML =
    '<thead><tr>' + headers.map((h) => `<th>${esc(h)}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map((r) => '<tr>' + r.map((c, i) => `<td class="${i ? 'num' : ''}">${esc(c)}</td>`).join('') + '</tr>').join('') +
    '</tbody>';
  wrap.appendChild(t);
  return wrap;
}

function advisoryCard(a) {
  const card = el('div', `advisory ${a.severity}`);
  card.innerHTML = `<h4>${esc(a.title)}</h4>${a.detail ? `<p>${esc(a.detail)}</p>` : ''}
    ${a.fix ? `<div class="fix">→ ${esc(a.fix)}</div>` : ''}
    ${a.field ? `<div class="field">${esc(a.field)}</div>` : ''}`;
  return card;
}

/* ── Boot ─────────────────────────────────────────────────── */
(async function init() {
  const [schemaResp, defaults, hw, backends] = await Promise.all([
    api('/recipe/schema'),
    api('/recipe/default'),
    api('/hardware'),
    api('/backends'),
  ]);

  state.schema = schemaResp.schema;
  state.docs = schemaResp.docs;
  state.lessonLinks = schemaResp.lesson_links;
  state.recipe = defaults.recipe;
  renderForm();

  if (defaults.hardware_note) {
    $('#hardware-note').textContent = `Defaults tuned to your hardware: ${defaults.hardware_note}`;
    $('#hardware-note').classList.remove('hidden');
  }

  const chip = $('#hardware-chip');
  if (hw.can_train) {
    chip.textContent = `${hw.gpus[0].name} · ${hw.gpus[0].vram_gb} GB`;
    chip.className = 'chip chip-ok';
  } else {
    chip.textContent = hw.torch_available ? 'No GPU detected' : 'PyTorch not installed';
    chip.className = 'chip chip-warn';
    chip.title = (hw.notes || []).join(' ');
  }

  const select = $('#backend-select');
  backends.backends.forEach((b) => {
    const opt = el('option', null, `${b.name}${b.is_simulation ? ' (simulation)' : ''}${b.available ? '' : ' — unavailable'}`);
    opt.value = b.name;
    opt.disabled = !b.available;
    if (b.name === backends.default) opt.selected = true;
    select.appendChild(opt);
  });
  select.addEventListener('change', showBackendNote);
  showBackendNote();

  function showBackendNote() {
    const chosen = backends.backends.find((b) => b.name === select.value);
    const note = $('#backend-note');
    if (!chosen) return;
    note.textContent = chosen.is_simulation
      ? `Simulation: ${chosen.reason} This narrates a real run's steps but produces no adapter file.`
      : chosen.reason;
    note.classList.remove('hidden');
  }

  await loadLessons();
})();
