Chart.defaults.color = '#6b7280';
Chart.defaults.borderColor = '#e5e7eb';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

const COLORS = {
  blue:    '#2563eb',
  cyan:    '#06b6d4',
  indigo:  '#4f46e5',
  green:   '#10b981',
  emerald: '#059669',
  red:     '#ef4444',
  amber:   '#f59e0b',
  purple:  '#a855f7',
  bg:      '#ffffff',
  border:  '#e5e7eb',
};

async function loadData() {
  try {
    const [forecasts, inventory, models, features, shap] = await Promise.all([
      fetch('data/future_forecasts.json').then(r => r.json()),
      fetch('data/inventory_recommendations.json').then(r => r.json()),
      fetch('data/model_results.json').then(r => r.json()),
      fetch('data/feature_importance.json').then(r => r.json()),
      fetch('data/shap_feature_importance.json').then(r => r.json()),
    ]);

    updateKPIs(forecasts, inventory);
    createForecastChart(forecasts);
    createInventoryDoughnut(inventory);
    createModelChart(models);
    buildModelTable(models);
    createFeatureChart(features);
    buildShapList(shap);
    populateInventoryTable(inventory);
    initTableFilters();
  } catch (err) {
    console.error('Data load error:', err);
  }
}

function updateKPIs(forecasts, inventory) {
  const total = forecasts.reduce((s, r) => s + r.forecast_sales, 0);
  const avg   = total / forecasts.length;
  const high  = inventory.filter(r => r.inventory_risk === 'High').length;

  animateCount('totalForecast', Math.round(total), true);
  animateCount('avgForecast',   parseFloat(avg.toFixed(2)), false);
  animateCount('highRisk',      high, false);
}

function animateCount(id, target, comma) {
  const el = document.getElementById(id);
  if (!el) return;
  const isFloat = !Number.isInteger(target);
  const duration = 1200;
  const startTime = performance.now();

  function step(now) {
    const pct = Math.min((now - startTime) / duration, 1);
    const ease = 1 - Math.pow(1 - pct, 3);
    const val  = target * ease;
    el.textContent = comma
      ? Math.round(val).toLocaleString()
      : isFloat ? val.toFixed(2) : Math.round(val).toLocaleString();
    if (pct < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function createForecastChart(forecasts) {
  const daily = {};
  forecasts.forEach(r => { daily[r.date] = (daily[r.date] || 0) + r.forecast_sales; });
  const labels = Object.keys(daily).sort();
  const data   = labels.map(d => daily[d]);

  new Chart(document.getElementById('forecastChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Forecast Demand',
        data,
        borderColor: COLORS.blue,
        borderWidth: 2.5,
        pointRadius: 3,
        pointHoverRadius: 6,
        pointBackgroundColor: COLORS.blue,
        pointBorderColor: COLORS.bg,
        pointBorderWidth: 2,
        fill: true,
        backgroundColor: ctx => {
          const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, ctx.chart.height);
          g.addColorStop(0,   'rgba(37,99,235,0.15)');
          g.addColorStop(1,   'rgba(37,99,235,0)');
          return g;
        },
        tension: 0.4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: COLORS.bg,
          borderColor: COLORS.border,
          borderWidth: 1,
          titleColor: '#111827',
          bodyColor: COLORS.blue,
          padding: 12,
          callbacks: { label: ctx => ` ${ctx.parsed.y.toLocaleString()} units` }
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            maxRotation: 45,
            callback: (_, i, ticks) => (i === 0 || i === ticks.length - 1 || i % 7 === 0) ? labels[i] : ''
          }
        },
        y: {
          grid: { color: '#f3f4f6' },
          ticks: { callback: v => v.toLocaleString(), color: '#6b7280' }
        }
      }
    }
  });
}

function createInventoryDoughnut(inventory) {
  const counts = {};
  inventory.forEach(r => { counts[r.inventory_risk] = (counts[r.inventory_risk] || 0) + 1; });
  const order  = ['High', 'Medium', 'Low'];
  const labels = order.filter(k => counts[k]);
  const data   = labels.map(k => counts[k]);
  const total  = data.reduce((s, v) => s + v, 0);
  const colorMap = { High: COLORS.red, Medium: COLORS.amber, Low: COLORS.emerald };

  new Chart(document.getElementById('inventoryChart'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: labels.map(k => colorMap[k] + '22'),
        borderColor:      labels.map(k => colorMap[k]),
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '70%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { padding: 20, usePointStyle: true, pointStyleWidth: 10, color: '#6b7280' }
        },
        tooltip: {
          backgroundColor: COLORS.bg,
          borderColor: COLORS.border,
          borderWidth: 1,
          titleColor: '#111827',
          callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed.toLocaleString()} (${(ctx.parsed/total*100).toFixed(1)}%)` }
        }
      }
    },
    plugins: [{
      id: 'centreText',
      afterDraw(chart) {
        const { ctx, chartArea: { width, height, left, top } } = chart;
        ctx.save();
        const cx = left + width / 2;
        const cy = top  + height / 2 - 10;
        ctx.textAlign = 'center';
        ctx.fillStyle = '#111827';
        ctx.font = "700 26px 'Inter', sans-serif";
        ctx.fillText(total.toLocaleString(), cx, cy);
        ctx.fillStyle = '#6b7280';
        ctx.font = "500 11px 'Inter', sans-serif";
        ctx.fillText('TOTAL', cx, cy + 18);
        ctx.restore();
      }
    }]
  });
}

function createModelChart(models) {
  new Chart(document.getElementById('modelChart'), {
    type: 'bar',
    data: {
      labels: models.map(m => m.Model),
      datasets: [
        { label: 'RMSE',  data: models.map(x => x.RMSE),  backgroundColor: COLORS.blue   + '22', borderColor: COLORS.blue,   borderWidth: 1.5, borderRadius: 4 },
        { label: 'MAE',   data: models.map(x => x.MAE),   backgroundColor: COLORS.indigo + '22', borderColor: COLORS.indigo, borderWidth: 1.5, borderRadius: 4 },
        { label: 'MAPE',  data: models.map(x => x.MAPE),  backgroundColor: COLORS.green  + '22', borderColor: COLORS.green,  borderWidth: 1.5, borderRadius: 4 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#6b7280', usePointStyle: true, padding: 14 } },
        tooltip: { backgroundColor: COLORS.bg, borderColor: COLORS.border, borderWidth: 1, titleColor: '#111827' }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#6b7280' } },
        y: { grid: { color: '#f3f4f6' }, ticks: { color: '#6b7280' } }
      }
    }
  });
}

function buildModelTable(models) {
  const maxRMSE = Math.max(...models.map(m => m.RMSE));
  const best    = models.reduce((a, b) => a.RMSE < b.RMSE ? a : b);

  const rows = models.map(m => {
    const isBest = m.Model === best.Model;
    const barW   = ((m.RMSE / maxRMSE) * 100).toFixed(1);
    return `
      <tr class="${isBest ? 'best' : ''}">
        <td><span class="model-name">${m.Model}</span>${isBest ? '<span class="best-badge">BEST</span>' : ''}</td>
        <td><div class="bar-cell"><span>${m.RMSE.toFixed(2)}</span><div class="mini-bar-bg"><div class="mini-bar" style="width:${barW}%"></div></div></div></td>
        <td>${m.MAE.toFixed(2)}</td>
        <td>${m.MAPE.toFixed(2)}%</td>
      </tr>`;
  }).join('');

  document.getElementById('modelTableWrap').innerHTML = `
    <table class="model-table">
      <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function createFeatureChart(features) {
  const top = features.slice(0, 10).reverse();

  new Chart(document.getElementById('featureChart'), {
    type: 'bar',
    data: {
      labels: top.map(f => f.feature),
      datasets: [{
        label: 'Importance',
        data:  top.map(f => f.importance),
        backgroundColor: top.map((_, i) => {
          const t = i / (top.length - 1);
          return `rgba(${Math.round(79 + t*(10-79))}, ${Math.round(70 + t*(185-70))}, ${Math.round(229 + t*(16-229))}, 0.75)`;
        }),
        borderColor: top.map((_, i) => {
          const t = i / (top.length - 1);
          return `rgb(${Math.round(79 + t*(10-79))}, ${Math.round(70 + t*(185-70))}, ${Math.round(229 + t*(16-229))})`;
        }),
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: COLORS.bg,
          borderColor: COLORS.border,
          borderWidth: 1,
          titleColor: '#111827',
          callbacks: { label: ctx => ` ${(ctx.parsed.x * 100).toFixed(2)}% importance` }
        }
      },
      scales: {
        x: { grid: { color: '#f3f4f6' }, ticks: { callback: v => (v * 100).toFixed(0) + '%', color: '#6b7280' } },
        y: { grid: { display: false }, ticks: { color: '#6b7280' } }
      }
    }
  });
}

function buildShapList(shap) {
  const top    = shap.slice(0, 10);
  const maxVal = top[0].mean_abs_shap;
  const el     = document.getElementById('shapList');
  if (!el) return;
  el.innerHTML = top.map((f, i) => `
    <div class="shap-row">
      <span class="shap-rank">${String(i + 1).padStart(2, '0')}</span>
      <span class="shap-name">${f.feature}</span>
      <div class="shap-bar-bg">
        <div class="shap-bar" style="width:${(f.mean_abs_shap / maxVal * 100).toFixed(1)}%"></div>
      </div>
      <span class="shap-val">${f.mean_abs_shap.toFixed(1)}</span>
    </div>`).join('');
}

let fullInventory = [];

function populateInventoryTable(inventory) {
  fullInventory = inventory;
  renderTableRows('all');
}

function renderTableRows(filter) {
  const body = document.getElementById('inventoryBody');
  if (!body) return;

  const filtered = filter === 'all'
    ? fullInventory
    : fullInventory.filter(r => r.inventory_risk === filter);

  const rows = filtered.slice(0, 50);

  if (rows.length === 0) {
    body.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:32px;color:#9ca3af;font-size:12px;">No records found</td></tr>`;
    return;
  }

  body.innerHTML = rows.map(r => `
    <tr>
      <td>${r.store_id}</td>
      <td>${r.dept_id}</td>
      <td>${r.date || '—'}</td>
      <td>${Math.round(r.forecast_sales).toLocaleString()}</td>
      <td>${Math.round(r.safety_stock).toLocaleString()}</td>
      <td>${Math.round(r.recommended_inventory).toLocaleString()}</td>
      <td><span class="risk-badge risk-${r.inventory_risk.toLowerCase()}">${r.inventory_risk}</span></td>
    </tr>`).join('');
}

function initTableFilters() {
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderTableRows(btn.dataset.filter);
    });
  });
}

loadData();