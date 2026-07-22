let clusters = [];
let jobsHistory = [];

document.addEventListener('DOMContentLoaded', async () => {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      showView(el.dataset.view);
    });
  });

  document.getElementById('menuToggle').addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('open');
  });

  document.querySelectorAll('.view-container').addEventListener('click', () => {
    document.getElementById('sidebar').classList.remove('open');
  });

  initClusterSelects();

  await initDashboard();
  startPolling();
});

function startPolling() {
  setInterval(async () => {
    try {
      await updateStatus();
    } catch {}
  }, 15000);
  setInterval(async () => {
    try {
      if (document.getElementById('view-dashboard').classList.contains('active')) {
        await loadDashboardStats();
      }
    } catch {}
  }, 30000);
}

function showView(viewId) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`view-${viewId}`).classList.add('active');
  document.querySelector(`.nav-item[data-view="${viewId}"]`).classList.add('active');
  const titles = { dashboard: 'Dashboard', clusters: 'Clusters', submit: 'Submit Job', jobs: 'Jobs', monitor: 'Monitor', twin: 'Digital Twin', predictions: 'Predictions', rtx: 'RTX 4090 Cluster' };
  document.getElementById('pageTitle').textContent = titles[viewId] || viewId;
  document.getElementById('sidebar').classList.remove('open');
  loadView(viewId);
}

function loadView(viewId) {
  const fns = {
    clusters: loadClusters,
    submit: loadSubmitResources,
    jobs: loadJobsMetrics,
    monitor: loadMonitorData,
    twin: loadTwinData,
    predictions: loadPredictionsData,
    rtx: loadRtxData,
  };
  if (fns[viewId]) fns[viewId]();
}

function refreshCurrentView() {
  const active = document.querySelector('.nav-item.active');
  if (active) loadView(active.dataset.view);
}

function initClusterSelects() {
  ['clusterSelect', 'twinClusterSelect', 'predClusterSelect'].forEach(id => {
    const sel = document.getElementById(id);
    if (sel) sel.addEventListener('change', () => {
      const view = sel.closest('.view');
      if (view) {
        const viewId = view.id.replace('view-', '');
        if (viewId === 'submit') loadSubmitResources();
        if (viewId === 'twin') loadTwinData();
      }
    });
  });
}

async function updateStatus() {
  try {
    const h = await api.health();
    document.getElementById('statusDot').className = 'status-dot connected';
    document.getElementById('statusText').textContent = 'Connected';
    document.getElementById('versionText').textContent = `v${h.version}`;
  } catch {
    document.getElementById('statusDot').className = 'status-dot error';
    document.getElementById('statusText').textContent = 'Disconnected';
  }
}

async function initDashboard() {
  await updateStatus();
  clusters = await api.listClusters();
  await loadDashboardStats();
}

async function loadDashboardStats() {
  try {
    const [health, metrics] = await Promise.all([
      api.getClusterHealth().catch(() => null),
      api.schedulerMetrics().catch(() => null),
    ]);

    let totalClusters = 0, healthyClusters = 0, totalGpus = 0;
    if (health && health.environments) {
      for (const env of Object.values(health.environments)) {
        totalClusters += env.clusters || 0;
        healthyClusters += env.healthy || 0;
      }
    } else if (clusters) {
      totalClusters = clusters.length;
    }

    let successRate = 0, qSize = 0, totalPlacements = 0;
    if (metrics) {
      successRate = ((metrics.placement_success_rate || 0) * 100).toFixed(1);
      qSize = metrics.q_table_size || 0;
      totalPlacements = metrics.total_placements || 0;
    }

    const states = await Promise.all(
      (clusters || []).map(c => api.getClusterState(c.id).catch(() => null))
    );
    totalGpus = states.reduce((s, st) => s + (st ? st.gpu_count || 0 : 0), 0);
    const totalNodes = states.reduce((s, st) => s + (st ? st.node_count || 0 : 0), 0);

    document.getElementById('dashboardStats').innerHTML = `
      <div class="stat-card"><div class="stat-label">Clusters</div><div class="stat-value blue">${totalClusters}</div><div class="stat-sub">${healthyClusters} healthy</div></div>
      <div class="stat-card"><div class="stat-label">Total GPUs</div><div class="stat-value green">${totalGpus}</div><div class="stat-sub">across ${totalNodes} nodes</div></div>
      <div class="stat-card"><div class="stat-label">Placement Success</div><div class="stat-value ${successRate > 80 ? 'green' : successRate > 50 ? 'yellow' : 'red'}">${successRate}%</div><div class="stat-sub">${totalPlacements} total placements</div></div>
      <div class="stat-card"><div class="stat-label">RL Model</div><div class="stat-value blue">${qSize}</div><div class="stat-sub">Q-table states</div></div>
    `;

    const healthHtml = health && health.environments
      ? Object.entries(health.environments).map(([env, data]) => `
        <div class="stat-card" style="margin-bottom:8px">
          <div class="stat-label" style="text-transform:capitalize">${env}</div>
          <div style="display:flex;gap:16px;margin-top:6px">
            <span style="color:var(--text-muted)">${data.clusters || 0} clusters</span>
            <span style="color:var(--success)">${data.healthy || 0} healthy</span>
            <span style="color:var(--warning)">${data.warning || 0} warning</span>
            <span style="color:var(--danger)">${data.failing || 0} failing</span>
          </div>
        </div>
      `).join('') : '<p class="placeholder">No health data available</p>';
    document.getElementById('clusterHealthChart').innerHTML = healthHtml;

    const statesHtml = states.filter(Boolean).map(st => {
      const gpuUtil = st.gpu_count && st.total_gpu_memory_bytes
        ? Math.round((st.nodes.reduce((s, n) =>
            s + n.gpu_devices.filter(g => g.memory_used_bytes > 0).length, 0
          ) / st.gpu_count) * 100)
        : 0;
      return `
        <div style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px">
            <span>${st.cluster_name}</span>
            <span>${st.gpu_count} GPUs</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill ${gpuUtil > 80 ? 'red' : gpuUtil > 50 ? 'yellow' : 'green'}" style="width:${gpuUtil}%"></div>
          </div>
          <div style="font-size:0.72rem;color:var(--text-muted);margin-top:2px">${gpuUtil}% utilized</div>
        </div>
      `;
    }).join('');
    document.getElementById('gpuUtilChart').innerHTML = statesHtml || '<p class="placeholder">No GPU data</p>';

    const recent = [];
    if (metrics && metrics.placement_history) {
      recent.push(...metrics.placement_history.slice(-5).reverse());
    }
    document.getElementById('recentActivity').innerHTML = recent.length
      ? recent.map(r => `
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(42,52,86,0.3)">
          <span>Job <code>${r.job_id || ''}</code> → ${r.node_id || '?'}</span>
          <span><span class="badge ${r.success ? 'badge-green' : 'badge-red'}">${r.success ? '✓ Success' : '✗ Failed'}</span></span>
        </div>
      `).join('') : '<p class="placeholder">No recent activity</p>';
  } catch (err) {
    console.error('Dashboard load error:', err);
  }
}

async function loadClusters() {
  try {
    clusters = await api.listClusters();
  } catch {}
  const container = document.getElementById('clustersList');
  if (!clusters || !clusters.length) {
    container.innerHTML = '<p class="placeholder">No clusters registered. Create one via the API.</p>';
    return;
  }

  container.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Environment</th><th>Region</th><th></th></tr></thead>
        <tbody>
          ${clusters.map(c => `
            <tr>
              <td><strong>${c.name}</strong></td>
              <td><span class="badge badge-blue">${c.connector_type}</span></td>
              <td><span class="badge ${c.status === 'healthy' ? 'badge-green' : c.status === 'warning' ? 'badge-yellow' : 'badge-red'}">${c.status}</span></td>
              <td>${c.region || '—'}</td>
              <td>${c.environment || '—'}</td>
              <td><button class="btn btn-sm" onclick="showClusterDetail('${c.id}')">View</button></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

async function showClusterDetail(clusterId) {
  try {
    const [state, plan] = await Promise.all([
      api.getClusterState(clusterId).catch(() => null),
      api.getSchedulingPlan(clusterId).catch(() => null),
    ]);

    const container = document.getElementById('clusterDetail');
    if (!state) {
      container.innerHTML = '<p class="placeholder">No state data available for this cluster</p>';
      return;
    }

    const nodes = state.nodes || [];
    const gpuModels = {};
    nodes.forEach(n => (n.gpu_devices || []).forEach(g => {
      const model = g.model || 'unknown';
      gpuModels[model] = (gpuModels[model] || 0) + 1;
    }));

    container.innerHTML = `
      <div style="margin-bottom:12px">
        <strong style="font-size:1rem">${state.cluster_name}</strong>
        <span class="badge badge-blue" style="margin-left:8px">${state.environment}</span>
      </div>
      <div class="stats-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:12px">
        <div class="stat-card"><div class="stat-label">Nodes</div><div class="stat-value blue">${state.node_count}</div></div>
        <div class="stat-card"><div class="stat-label">GPUs</div><div class="stat-value green">${state.gpu_count}</div></div>
        <div class="stat-card"><div class="stat-label">Total Memory</div><div class="stat-value">${(state.total_gpu_memory_bytes / 1e9).toFixed(1)} GB</div></div>
        ${plan ? `<div class="stat-card"><div class="stat-label">Free GPUs</div><div class="stat-value">${plan.free_gpus}</div></div>` : ''}
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Node</th><th>Status</th><th>GPUs</th><th>GPU Models</th><th>Memory</th></tr></thead>
          <tbody>
            ${nodes.map(n => {
              const totalMem = n.gpu_devices.reduce((s, g) => s + g.memory_total_bytes, 0);
              const usedMem = n.gpu_devices.reduce((s, g) => s + g.memory_used_bytes, 0);
              const pct = totalMem ? Math.round((usedMem / totalMem) * 100) : 0;
              return `<tr>
                <td>${n.name}</td>
                <td><span class="badge ${n.status === 'Ready' || n.status === 'ready' ? 'badge-green' : 'badge-red'}">${n.status}</span></td>
                <td>${n.gpu_devices.length}</td>
                <td>${[...new Set(n.gpu_devices.map(g => g.model).filter(Boolean))].join(', ') || '—'}</td>
                <td>
                  <div style="display:flex;align-items:center;gap:8px">
                    <div class="progress-bar" style="flex:1;width:auto">
                      <div class="progress-fill ${pct > 80 ? 'red' : pct > 50 ? 'yellow' : 'green'}" style="width:${pct}%"></div>
                    </div>
                    <span style="font-size:0.78rem;color:var(--text-muted)">${pct}%</span>
                  </div>
                </td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>
    `;
  } catch (err) {
    document.getElementById('clusterDetail').innerHTML = `<p class="placeholder error">Error: ${err.message}</p>`;
  }
}

async function loadSubmitResources() {
  const sel = document.getElementById('clusterSelect');
  try {
    if (!clusters || !clusters.length) clusters = await api.listClusters();
  } catch {}
  sel.innerHTML = '<option value="">— Select Cluster —</option>' +
    (clusters || []).map(c => `<option value="${c.id}">${c.name} (${c.connector_type})</option>`).join('');

  const clusterId = sel.value;
  if (!clusterId) {
    document.getElementById('clusterResources').innerHTML = '<p class="placeholder">Select a cluster to view available resources</p>';
    return;
  }

  try {
    const [state, usage] = await Promise.all([
      api.getClusterState(clusterId).catch(() => null),
      api.getGpuUsage(clusterId).catch(() => null),
    ]);

    if (!state) {
      document.getElementById('clusterResources').innerHTML = '<p class="placeholder">No state data. Collect cluster state first.</p>';
      return;
    }

    const freeGpus = state.nodes.reduce((s, n) =>
      s + n.gpu_devices.filter(g => g.memory_used_bytes === 0).length, 0
    );
    const gpuModels = {};
    state.nodes.forEach(n => (n.gpu_devices || []).forEach(g => {
      const m = g.model || 'unknown';
      gpuModels[m] = (gpuModels[m] || 0) + 1;
    }));

    document.getElementById('clusterResources').innerHTML = `
      <div class="stats-grid" style="grid-template-columns:repeat(2,1fr)">
        <div class="stat-card"><div class="stat-label">Total GPUs</div><div class="stat-value blue">${state.gpu_count}</div></div>
        <div class="stat-card"><div class="stat-label">Free GPUs</div><div class="stat-value green">${freeGpus}</div></div>
        <div class="stat-card"><div class="stat-label">Nodes</div><div class="stat-value">${state.node_count}</div></div>
        <div class="stat-card"><div class="stat-label">Total Memory</div><div class="stat-value">${(state.total_gpu_memory_bytes / 1e9).toFixed(1)} GB</div></div>
      </div>
      <div style="margin-top:8px">
        <div class="stat-label">GPU Models:</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">
          ${Object.entries(gpuModels).map(([m, c]) =>
            `<span class="badge badge-blue">${m}: ${c}</span>`
          ).join('')}
        </div>
      </div>
    `;
  } catch (err) {
    document.getElementById('clusterResources').innerHTML = `<p class="placeholder error">Error: ${err.message}</p>`;
  }
}

async function submitJob(e) {
  e.preventDefault();
  const btn = document.getElementById('submitBtn');
  const text = document.getElementById('submitBtnText');
  const spinner = document.getElementById('submitSpinner');

  btn.disabled = true;
  text.textContent = 'Submitting...';
  spinner.style.display = 'inline-block';

  const clusterId = document.getElementById('clusterSelect').value;
  if (!clusterId) { showResult('error', 'Please select a cluster'); btn.disabled = false; text.textContent = 'Submit Job'; spinner.style.display = 'none'; return; }

  const job = {
    cluster_id: clusterId,
    job_name: document.getElementById('jobName').value || 'unnamed-job',
    required_gpus: parseInt(document.getElementById('requiredGpus').value) || 1,
    required_cpu_cores: parseInt(document.getElementById('requiredCpu').value) || 1,
    required_memory_gb: parseFloat(document.getElementById('requiredMemory').value) || 8,
    estimated_runtime_hours: parseFloat(document.getElementById('runtimeHours').value) || 1,
    priority: parseInt(document.getElementById('priority').value) || 5,
    model_name: document.getElementById('modelName').value || null,
    checkpointable: document.getElementById('checkpointable').checked,
  };

  try {
    const result = await api.submitJob(job);
    showResult(result.status === 'submitted' ? 'success' : result.status === 'rejected' ? 'warning' : 'error', formatJobResult(result));
    if (result.status === 'submitted') {
      jobsHistory.push(result);
    }
  } catch (err) {
    showResult('error', `API Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    text.textContent = 'Submit Job';
    spinner.style.display = 'none';
  }
}

function formatJobResult(result) {
  let html = `<div style="margin-bottom:8px"><strong>Status:</strong> <span class="badge ${result.status === 'submitted' ? 'badge-green' : result.status === 'rejected' ? 'badge-yellow' : 'badge-red'}">${result.status}</span></div>`;
  html += `<p>${result.message}</p>`;
  if (result.job_id) html += `<p style="margin-top:4px"><strong>Job ID:</strong> <code>${result.job_id}</code></p>`;
  if (result.simulation_results) {
    html += `<pre>${JSON.stringify(result.simulation_results, null, 2)}</pre>`;
  }
  if (result.placement) {
    html += `<div style="margin-top:8px"><strong>Placement:</strong> ${result.placement.suggested_node} (confidence: ${(result.placement.confidence * 100).toFixed(0)}%)</div>`;
  }
  return html;
}

function showResult(type, content) {
  const card = document.getElementById('resultCard');
  const container = document.getElementById('resultContent');
  card.style.display = 'block';
  container.innerHTML = `<div class="result-box ${type}">${content}</div>`;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function loadJobsMetrics() {
  try {
    const metrics = await api.schedulerMetrics();
    document.getElementById('schedulerMetrics').innerHTML = `
      <div class="stats-grid" style="grid-template-columns:repeat(4,1fr)">
        <div class="stat-card"><div class="stat-label">Q-Table Size</div><div class="stat-value blue">${metrics.q_table_size || 0}</div></div>
        <div class="stat-card"><div class="stat-label">Success Rate</div><div class="stat-value green">${((metrics.placement_success_rate || 0) * 100).toFixed(1)}%</div></div>
        <div class="stat-card"><div class="stat-label">Total Placements</div><div class="stat-value">${metrics.total_placements || 0}</div></div>
        <div class="stat-card"><div class="stat-label">Avg Reward</div><div class="stat-value">${(metrics.average_reward || 0).toFixed(2)}</div></div>
      </div>
    `;
  } catch {}

  if (jobsHistory.length) {
    document.getElementById('jobsHistory').innerHTML = `
      <div class="table-wrap" style="margin-top:12px">
        <table>
          <thead><tr><th>Job ID</th><th>Status</th><th>Message</th><th>Placement</th></tr></thead>
          <tbody>
            ${jobsHistory.slice().reverse().map(j => `
              <tr>
                <td><code>${j.job_id || '—'}</code></td>
                <td><span class="badge ${j.status === 'submitted' ? 'badge-green' : 'badge-yellow'}">${j.status}</span></td>
                <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">${j.message || ''}</td>
                <td>${j.placement ? j.placement.suggested_node : '—'}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }
}

async function loadMonitorData() {
  try {
    const clusters = await api.listClusters().catch(() => []);
    const clusterId = clusters.length ? clusters[0].id : '';

    const [snapshot, domains] = await Promise.all([
      clusterId ? api.getGpuSnapshot(clusterId).catch(() => null) : null,
      api.getDomainCounts().catch(() => null),
    ]);

    if (snapshot) {
      const gpus = snapshot.gpu_devices || [];
      document.getElementById('gpuSnapshot').innerHTML = gpus.length
        ? `<div class="gpu-grid">${gpus.map(g => `
          <div class="gpu-item">
            <div class="gpu-label">${g.node || 'node'} / GPU ${g.index || 0}</div>
            <div class="gpu-value">${((g.utilization_gpu || 0)).toFixed(0)}%</div>
            <div class="gpu-label">${g.temperature || 0}°C | ${(g.power_draw || 0).toFixed(0)}W</div>
            <div class="progress-bar" style="margin-top:4px">
              <div class="progress-fill ${(g.utilization_gpu || 0) > 80 ? 'red' : (g.utilization_gpu || 0) > 50 ? 'yellow' : 'green'}" style="width:${g.utilization_gpu || 0}%"></div>
            </div>
          </div>
        `).join('')}</div>`
        : '<p class="placeholder">No GPU devices in snapshot</p>';
    } else {
      document.getElementById('gpuSnapshot').innerHTML = '<p class="placeholder">No GPU snapshot available</p>';
    }

    if (domains) {
      document.getElementById('domainCounts').innerHTML = `
        <div class="stats-grid" style="grid-template-columns:repeat(4,1fr)">
          ${Object.entries(domains).map(([name, count]) => `
            <div class="stat-card">
              <div class="stat-label" style="text-transform:capitalize">${name.replace(/_/g, ' ')}</div>
              <div class="stat-value blue">${count}</div>
            </div>
          `).join('')}
        </div>
      `;
    }

    if (clusterId) {
      const state = await api.getClusterState(clusterId).catch(() => null);
      if (state && state.telemetry) {
        document.getElementById('nodeTelemetry').innerHTML = `
          <div class="table-wrap">
            <table>
              <thead><tr><th>Node</th><th>CPU (m)</th><th>Memory</th><th>GPUs</th><th>Avg GPU Util</th><th>Avg Temp</th></tr></thead>
              <tbody>
                ${(state.telemetry.nodes || []).map(nt => {
                  const devs = nt.gpu_devices || [];
                  const avgUtil = devs.length ? (devs.reduce((s, d) => s + (d.utilization_gpu_percent || 0), 0) / devs.length).toFixed(1) : '—';
                  const avgTemp = devs.length ? (devs.reduce((s, d) => s + (d.temperature_gpu_celsius || 0), 0) / devs.length).toFixed(0) : '—';
                  return `<tr>
                    <td>${nt.node_name}</td>
                    <td>${nt.cpu_usage_millicores || 0} / ${nt.cpu_capacity_millicores || 0}</td>
                    <td>${((nt.memory_usage_bytes || 0) / 1e9).toFixed(1)} / ${((nt.memory_capacity_bytes || 0) / 1e9).toFixed(1)} GB</td>
                    <td>${devs.length}</td>
                    <td>${avgUtil}%</td>
                    <td>${avgTemp}°C</td>
                  </tr>`;
                }).join('')}
              </tbody>
            </table>
          </div>
        `;
      }
    }
  } catch (err) {
    console.error('Monitor load error:', err);
  }
}

async function loadTwinData() {
  const sel = document.getElementById('twinClusterSelect');
  try {
    if (!clusters || !clusters.length) clusters = await api.listClusters();
  } catch {}
  sel.innerHTML = '<option value="">— Select Cluster —</option>' +
    (clusters || []).map(c => `<option value="${c.id}">${c.name}</option>`).join('');

  const clusterId = sel.value;
  if (!clusterId) return;

  try {
    const [twin, state] = await Promise.all([
      api.getTwin(clusterId).catch(() => null),
      api.getClusterState(clusterId).catch(() => null),
    ]);

    if (twin) {
      document.getElementById('twinState').innerHTML = `
        <div class="twin-metrics">
          <div class="twin-metric"><div class="twin-val">${twin.node_count || '—'}</div><div class="twin-label">Nodes</div></div>
          <div class="twin-metric"><div class="twin-val">${twin.gpu_count || '—'}</div><div class="twin-label">GPUs</div></div>
          <div class="twin-metric"><div class="twin-val">${twin.has_diverged ? 'Diverged' : 'Synced'}</div><div class="twin-label">Status</div></div>
        </div>
        ${twin.divergence_reason ? `<div class="result-box warning" style="margin-top:8px"><strong>Divergence:</strong> ${twin.divergence_reason}</div>` : ''}
        <div style="margin-top:8px;font-size:0.78rem;color:var(--text-muted)">
          Synced: ${twin.synced_at ? new Date(twin.synced_at).toLocaleString() : '—'}
        </div>
      `;
    } else {
      document.getElementById('twinState').innerHTML = '<p class="placeholder">No twin synced yet. Click "Sync Twin".</p>';
    }

    if (state) {
      document.getElementById('twinComparison').innerHTML = `
        <div style="font-size:0.85rem;color:var(--text-secondary)">
          <p>Current cluster: ${state.cluster_name}</p>
          <p style="margin-top:4px">${state.node_count} nodes, ${state.gpu_count} GPUs</p>
          <p>Last collected: ${state.collected_at ? new Date(state.collected_at).toLocaleString() : '—'}</p>
        </div>
      `;
    }
  } catch (err) {
    document.getElementById('twinState').innerHTML = `<p class="placeholder error">Error: ${err.message}</p>`;
  }
}

async function syncTwin() {
  const sel = document.getElementById('twinClusterSelect');
  const clusterId = sel.value;
  if (!clusterId) return;
  try {
    await api.syncTwin(clusterId);
    await loadTwinData();
    showToast('Digital twin synced successfully', 'success');
  } catch (err) {
    showToast(`Sync failed: ${err.message}`, 'error');
  }
}

async function compareTwin() {
  const sel = document.getElementById('twinClusterSelect');
  const clusterId = sel.value;
  if (!clusterId) return;
  try {
    const result = await api.compareTwin(clusterId);
    document.getElementById('twinComparison').innerHTML = `
      <div class="result-box ${result.overall_drift_severity === 'critical' || result.overall_drift_severity === 'high' ? 'warning' : 'info'}">
        <strong>Drift Summary:</strong> ${result.summary || 'No drift detected'}
        <div class="twin-metrics" style="margin-top:8px">
          <div class="twin-metric"><div class="twin-val">${result.drift_count || 0}</div><div class="twin-label">Total Drifts</div></div>
          <div class="twin-metric"><div class="twin-val red">${result.critical_drift_count || 0}</div><div class="twin-label">Critical</div></div>
          <div class="twin-metric"><div class="twin-val yellow">${result.high_drift_count || 0}</div><div class="twin-label">High</div></div>
        </div>
        <pre style="margin-top:8px">${JSON.stringify(result, null, 2)}</pre>
      </div>
    `;
  } catch (err) {
    showToast(`Compare failed: ${err.message}`, 'error');
  }
}

async function resetTwin() {
  const sel = document.getElementById('twinClusterSelect');
  const clusterId = sel.value;
  if (!clusterId) return;
  try {
    await api.resetTwin(clusterId);
    await loadTwinData();
    showToast('Twin reset to actual state', 'success');
  } catch (err) {
    showToast(`Reset failed: ${err.message}`, 'error');
  }
}

async function loadPredictionsData() {
  const sel = document.getElementById('predClusterSelect');
  try {
    if (!clusters || !clusters.length) clusters = await api.listClusters();
  } catch {}
  sel.innerHTML = '<option value="">— Select Cluster —</option>' +
    (clusters || []).map(c => `<option value="${c.id}">${c.name}</option>`).join('');

  document.getElementById('predictorStatus').innerHTML = `
    <div class="stats-grid" style="grid-template-columns:repeat(3,1fr)">
      <div class="stat-card"><div class="stat-label">Model</div><div class="stat-value blue">Random Forest</div><div class="stat-sub">100 estimators</div></div>
      <div class="stat-card"><div class="stat-label">Anomaly Detection</div><div class="stat-value blue">Isolation Forest</div><div class="stat-sub">contamination: 0.1</div></div>
      <div class="stat-card"><div class="stat-label">Features</div><div class="stat-value">19</div><div class="stat-sub">including derived ratios</div></div>
    </div>
  `;
}

async function analyzeCluster() {
  const sel = document.getElementById('predClusterSelect');
  const clusterId = sel.value;
  if (!clusterId) return;

  const btn = document.querySelector('#view-predictions .btn-primary');
  if (!btn) return;
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Analyzing...';

  try {
    const result = await api.analyzeCluster(clusterId, 4);
    const nodes = result.nodes || {};
    const entries = Object.entries(nodes);

    document.getElementById('predictionResults').innerHTML = `
      <div style="margin-top:12px">
        <div class="stats-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:12px">
          <div class="stat-card"><div class="stat-label">Total Nodes</div><div class="stat-value">${result.summary?.total_nodes || 0}</div></div>
          <div class="stat-card"><div class="stat-label">High Risk</div><div class="stat-value red">${result.summary?.high_risk_nodes || 0}</div></div>
          <div class="stat-card"><div class="stat-label">Medium Risk</div><div class="stat-value yellow">${result.summary?.medium_risk_nodes || 0}</div></div>
        </div>
        ${entries.map(([name, data]) => {
          const prob = data.failure_probability || 0;
          const riskClass = prob > 0.7 ? 'high' : prob > 0.4 ? 'medium' : 'low';
          return `<div class="pred-node">
            <div>
              <div class="pred-name">${name}</div>
              <div style="font-size:0.78rem;color:var(--text-muted)">${(data.risk_factors || []).join('; ') || 'No risk factors'}</div>
            </div>
            <div class="pred-risk ${riskClass}">${(prob * 100).toFixed(0)}%</div>
          </div>`;
        }).join('')}
        ${result.summary?.high_risk_node_ids?.length ? `
          <div class="result-box warning" style="margin-top:8px">
            ⚠ High risk nodes: ${result.summary.high_risk_node_ids.join(', ')}
          </div>
        ` : `
          <div class="result-box success" style="margin-top:8px">
            ✓ No high-risk nodes detected
          </div>
        `}
        <pre style="margin-top:8px;max-height:200px">${JSON.stringify(result, null, 2)}</pre>
      </div>
    `;
  } catch (err) {
    document.getElementById('predictionResults').innerHTML = `<div class="result-box error">Error: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

async function loadRtxData() {
  try {
    const [status, gpus, jobs, metrics] = await Promise.all([
      api.rtxStatus(),
      api.rtxGpus(),
      api.rtxJobs(),
      api.rtxMetrics(),
    ]);

    document.getElementById('rtxStats').innerHTML = `
      <div class="stat-card"><div class="stat-label">GPU</div><div class="stat-value blue">${status.gpus[0]?.name || 'N/A'}</div><div class="stat-sub">${status.gpus.length} device(s)</div></div>
      <div class="stat-card"><div class="stat-label">GPU Memory</div><div class="stat-value green">${status.aggregate.total_gpu_memory_gb} GB</div><div class="stat-sub">Total VRAM</div></div>
      <div class="stat-card"><div class="stat-label">GPU Utilization</div><div class="stat-value ${status.aggregate.total_gpu_usage_percent > 80 ? 'red' : status.aggregate.total_gpu_usage_percent > 50 ? 'yellow' : 'green'}">${status.aggregate.total_gpu_usage_percent}%</div><div class="stat-sub">${status.aggregate.total_power_watts}W power draw</div></div>
      <div class="stat-card"><div class="stat-label">CUDA</div><div class="stat-value blue">${status.gpus[0]?.cuda_version || 'N/A'}</div><div class="stat-sub">Driver: ${status.gpus[0]?.driver_version || 'N/A'}</div></div>
    `;

    const gpuHtml = (status.gpus || []).map(g => {
      const memPct = g.memory_total_gb ? ((g.memory_used_gb / g.memory_total_gb) * 100).toFixed(1) : 0;
      const tempClass = g.temperature_celsius > 80 ? 'red' : g.temperature_celsius > 60 ? 'yellow' : 'green';
      const utilClass = g.utilization_percent > 80 ? 'red' : g.utilization_percent > 50 ? 'yellow' : 'green';
      return `
        <div style="background:var(--bg-input);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <strong>${g.name}</strong>
            <span class="badge ${g.health_status === 'healthy' ? 'badge-green' : 'badge-yellow'}">${g.health_status}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.85rem">
            <div><span style="color:var(--text-muted)">Memory:</span> ${g.memory.used_gb} / ${g.memory.total_gb} GB (${memPct}%)</div>
            <div><span style="color:var(--text-muted)">Utilization:</span> <span class="${utilClass}">${g.utilization_percent}%</span></div>
            <div><span style="color:var(--text-muted)">Temperature:</span> <span class="${tempClass}">${g.temperature_celsius}°C</span></div>
            <div><span style="color:var(--text-muted)">Power:</span> ${g.power.current_watts} / ${g.power.limit_watts} W (${g.power.usage_percent}%)</div>
            <div><span style="color:var(--text-muted)">PCIe:</span> Gen ${g.pcie_link_gen} ×${g.pcie_link_width}</div>
            <div><span style="color:var(--text-muted)">ECC:</span> ${g.ecc_errors} errors</div>
          </div>
          <div class="progress-bar" style="margin-top:8px">
            <div class="progress-fill ${utilClass}" style="width:${g.utilization_percent}%"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:var(--text-muted);margin-top:2px">
            <span>Utilization</span>
            <span>${g.utilization_percent}%</span>
          </div>
        </div>
      `;
    }).join('');
    document.getElementById('rtxGpus').innerHTML = gpuHtml || '<p class="placeholder">No GPUs detected</p>';

    document.getElementById('rtxSystem').innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div style="background:var(--bg-input);padding:12px;border-radius:6px">
          <div style="font-size:0.78rem;color:var(--text-muted)">CPU</div>
          <div style="font-size:1.2rem;font-weight:600">${status.cpu.cores} cores</div>
          <div style="font-size:0.82rem">${status.cpu.usage_percent}% used</div>
        </div>
        <div style="background:var(--bg-input);padding:12px;border-radius:6px">
          <div style="font-size:0.78rem;color:var(--text-muted)">System Memory</div>
          <div style="font-size:1.2rem;font-weight:600">${status.memory.free_gb} GB free</div>
          <div style="font-size:0.82rem">of ${status.memory.total_gb} GB</div>
        </div>
        <div style="background:var(--bg-input);padding:12px;border-radius:6px">
          <div style="font-size:0.78rem;color:var(--text-muted)">Active Jobs</div>
          <div style="font-size:1.2rem;font-weight:600" class="blue">${metrics.jobs.running} running</div>
          <div style="font-size:0.82rem">${metrics.jobs.queued} queued / ${metrics.jobs.total} total</div>
        </div>
        <div style="background:var(--bg-input);padding:12px;border-radius:6px">
          <div style="font-size:0.78rem;color:var(--text-muted)">GPU Driver</div>
          <div style="font-size:1.2rem;font-weight:600" class="blue">${status.gpus[0]?.driver_version || 'N/A'}</div>
          <div style="font-size:0.82rem">CUDA ${status.gpus[0]?.cuda_version || 'N/A'}</div>
        </div>
      </div>
    `;

    const jobsHtml = (jobs.jobs || []).length
      ? `<div class="table-wrap"><table>
          <thead><tr><th>ID</th><th>Name</th><th>Status</th><th>GPUs</th><th>Memory</th><th>Runtime</th><th>Priority</th></tr></thead>
          <tbody>${jobs.jobs.slice().reverse().map(j => `
            <tr>
              <td><code>${j.job_id}</code></td>
              <td>${j.name}</td>
              <td><span class="badge ${j.status === 'running' ? 'badge-green' : j.status === 'queued' ? 'badge-yellow' : 'badge-red'}">${j.status}</span></td>
              <td>${j.required_gpus}</td>
              <td>${j.required_memory_gb} GB</td>
              <td>${j.estimated_runtime_hours}h</td>
              <td>${j.priority}</td>
            </tr>
          `).join('')}</tbody></table></div>`
      : '<p class="placeholder">No jobs submitted</p>';
    document.getElementById('rtxJobs').innerHTML = jobsHtml;
  } catch (err) {
    console.error('RTX load error:', err);
    document.getElementById('rtxGpus').innerHTML = `<p class="placeholder error">Error: ${err.message}</p>`;
  }
}

async function submitRtxJob(e) {
  e.preventDefault();
  const btn = document.getElementById('rtxSubmitBtn');
  btn.disabled = true;
  btn.textContent = 'Submitting...';

  const job = {
    name: document.getElementById('rtxJobName').value || 'RTX Job',
    required_gpus: parseInt(document.getElementById('rtxGpuCount').value) || 1,
    required_memory_gb: parseFloat(document.getElementById('rtxMemory').value) || 8,
    estimated_runtime_hours: parseFloat(document.getElementById('rtxRuntime').value) || 1,
    priority: parseInt(document.getElementById('rtxPriority').value) || 5,
  };

  try {
    const result = await api.rtxSubmit(job);
    const card = document.getElementById('rtxResultCard');
    const container = document.getElementById('rtxResult');
    card.style.display = 'block';
    const type = result.success ? 'success' : 'warning';
    container.innerHTML = `<div class="result-box ${type}">
      <strong>${result.success ? '✓ Job Submitted' : '⚠ Job Queued'}</strong>
      <p>${result.message}</p>
      <pre>${JSON.stringify(result.job, null, 2)}</pre>
    </div>`;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    await loadRtxData();
  } catch (err) {
    const card = document.getElementById('rtxResultCard');
    card.style.display = 'block';
    document.getElementById('rtxResult').innerHTML = `<div class="result-box error">Error: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Submit Job';
  }
}

async function simulateRtxJob() {
  const job = {
    name: (document.getElementById('rtxJobName').value || 'Simulation') + ' (sim)',
    required_gpus: parseInt(document.getElementById('rtxGpuCount').value) || 1,
    required_memory_gb: parseFloat(document.getElementById('rtxMemory').value) || 8,
    estimated_runtime_hours: parseFloat(document.getElementById('rtxRuntime').value) || 1,
    priority: parseInt(document.getElementById('rtxPriority').value) || 5,
  };

  try {
    const result = await api.rtxSimulate(job);
    const card = document.getElementById('rtxResultCard');
    const container = document.getElementById('rtxResult');
    card.style.display = 'block';
    const type = result.simulation.feasible ? 'success' : 'warning';
    container.innerHTML = `<div class="result-box ${type}">
      <strong>${result.simulation.feasible ? '✓ Simulation: Feasible' : '⚠ Simulation: Resource Constrained'}</strong>
      <div class="twin-metrics" style="margin-top:8px">
        <div class="twin-metric"><div class="twin-val green">${(result.simulation.predicted_success_rate * 100).toFixed(0)}%</div><div class="twin-label">Success Rate</div></div>
        <div class="twin-metric"><div class="twin-val">${result.simulation.candidate_gpus.length}</div><div class="twin-label">Candidate GPUs</div></div>
        <div class="twin-metric"><div class="twin-val">${result.simulation.estimated_power_cost_kwh} kWh</div><div class="twin-label">Est. Power</div></div>
      </div>
      <pre style="margin-top:8px">${JSON.stringify(result, null, 2)}</pre>
    </div>`;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch (err) {
    const card = document.getElementById('rtxResultCard');
    card.style.display = 'block';
    document.getElementById('rtxResult').innerHTML = `<div class="result-box error">Error: ${err.message}</div>`;
  }
}

function showToast(msg, type = 'info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = msg;
  Object.assign(toast.style, {
    position: 'fixed',
    bottom: '20px', right: '20px',
    padding: '12px 20px',
    borderRadius: '6px',
    background: type === 'success' ? 'rgba(52,211,153,0.95)' : type === 'error' ? 'rgba(248,113,113,0.95)' : 'rgba(79,140,255,0.95)',
    color: '#fff',
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    fontSize: '0.88rem',
    zIndex: '1000',
    boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
  });
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.3s'; setTimeout(() => toast.remove(), 300); }, 3000);
}