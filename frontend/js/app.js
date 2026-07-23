let clusters = [];
let jobsHistory = [];
let evolutionChart = null;
let policyCompareChart = null;
let costTrendChart = null;
let forecastChart = null;
let realtimeChart = null;
let kpiChart = null;
let simCompareChart = null;
let threeScene = null;
let evolutionData = { fitnessHistory: [], generation: 0 };
let realtimeBuffer = [];

// ─── 3D Background ───

function initThreeBg() {
  const container = document.getElementById('three-bg');
  if (!container || !window.THREE) return;

  try {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const centerGroup = new THREE.Group();
    scene.add(centerGroup);

    // ─── Core geometric object: torus knot ───
    const knotGeo = new THREE.TorusKnotGeometry(0.8, 0.28, 128, 16);
    const knotMat = new THREE.MeshPhysicalMaterial({
      color: 0x4f8cff,
      metalness: 0.4,
      roughness: 0.15,
      clearcoat: 0.9,
      emissive: 0x4f8cff,
      emissiveIntensity: 0.2,
      transparent: true,
      opacity: 0.7,
    });
    const knot = new THREE.Mesh(knotGeo, knotMat);
    knot.position.y = 0.5;
    centerGroup.add(knot);

    // ─── Floating ring orbit ───
    const ringPoints = [];
    const ringRadius = 1.8;
    const ringSegments = 80;
    for (let i = 0; i <= ringSegments; i++) {
      const angle = (i / ringSegments) * Math.PI * 2;
      ringPoints.push(new THREE.Vector3(
        Math.cos(angle) * ringRadius,
        Math.sin(angle) * ringRadius * 0.3 + 0.5,
        Math.sin(angle) * ringRadius * 0.5
      ));
    }
    const ringGeo = new THREE.BufferGeometry().setFromPoints(ringPoints);
    const ringMat = new THREE.LineBasicMaterial({
      color: 0x4f8cff,
      transparent: true,
      opacity: 0.15,
    });
    const ring = new THREE.Line(ringGeo, ringMat);
    centerGroup.add(ring);

    // Second ring perpendicular
    const ringPoints2 = [];
    for (let i = 0; i <= ringSegments; i++) {
      const angle = (i / ringSegments) * Math.PI * 2;
      ringPoints2.push(new THREE.Vector3(
        Math.sin(angle) * ringRadius * 0.5,
        Math.cos(angle) * ringRadius * 0.3 + 0.5,
        Math.cos(angle) * ringRadius
      ));
    }
    const ringGeo2 = new THREE.BufferGeometry().setFromPoints(ringPoints2);
    const ringMat2 = new THREE.LineBasicMaterial({
      color: 0x6c5ce7,
      transparent: true,
      opacity: 0.1,
    });
    const ring2 = new THREE.Line(ringGeo2, ringMat2);
    centerGroup.add(ring2);

    // ─── Orbiting particles around center ───
    const orbitParticleCount = 300;
    const orbitGeo = new THREE.BufferGeometry();
    const orbitPos = new Float32Array(orbitParticleCount * 3);
    const orbitSizes = new Float32Array(orbitParticleCount);
    const orbitSpeeds = new Float32Array(orbitParticleCount);
    const orbitRadii = new Float32Array(orbitParticleCount);
    const orbitAngles = new Float32Array(orbitParticleCount);
    const orbitYOffsets = new Float32Array(orbitParticleCount);
    for (let i = 0; i < orbitParticleCount; i++) {
      orbitRadii[i] = 1.2 + Math.random() * 1.8;
      orbitAngles[i] = Math.random() * Math.PI * 2;
      orbitSpeeds[i] = 0.002 + Math.random() * 0.006;
      orbitYOffsets[i] = (Math.random() - 0.5) * 2;
      orbitSizes[i] = 0.02 + Math.random() * 0.04;
    }
    orbitGeo.setAttribute('position', new THREE.BufferAttribute(orbitPos, 3));
    const orbitMat = new THREE.PointsMaterial({
      size: 0.03,
      color: 0x6c5ce7,
      transparent: true,
      opacity: 0.5,
      blending: THREE.AdditiveBlending,
    });
    const orbitParticles = new THREE.Points(orbitGeo, orbitMat);
    centerGroup.add(orbitParticles);

    // ─── Background star field ───
    const starCount = 2500;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starCount * 3);
    const starColors = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i += 3) {
      starPos[i] = (Math.random() - 0.5) * 50;
      starPos[i + 1] = (Math.random() - 0.5) * 40;
      starPos[i + 2] = (Math.random() - 0.5) * 30 - 10;
      const c = 0.3 + Math.random() * 0.4;
      const tint = Math.random();
      if (tint < 0.3) {
        starColors[i] = c * 0.6; starColors[i + 1] = c * 0.4; starColors[i + 2] = c;
      } else if (tint < 0.5) {
        starColors[i] = c; starColors[i + 1] = c * 0.5; starColors[i + 2] = c * 0.5;
      } else {
        starColors[i] = c * 0.4; starColors[i + 1] = c * 0.5; starColors[i + 2] = c;
      }
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
    const starMat = new THREE.PointsMaterial({
      size: 0.06,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
    });
    const stars = new THREE.Points(starGeo, starMat);
    scene.add(stars);

    // ─── Connecting lines ───
    const linePositions = [];
    for (let i = 0; i < starCount; i++) {
      for (let j = i + 1; j < starCount; j++) {
        const dx = starPos[i * 3] - starPos[j * 3];
        const dy = starPos[i * 3 + 1] - starPos[j * 3 + 1];
        const dz = starPos[i * 3 + 2] - starPos[j * 3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 3 && Math.random() < 0.015) {
          linePositions.push(starPos[i * 3], starPos[i * 3 + 1], starPos[i * 3 + 2]);
          linePositions.push(starPos[j * 3], starPos[j * 3 + 1], starPos[j * 3 + 2]);
        }
      }
    }
    const lineAttr = new Float32Array(linePositions);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(lineAttr, 3));
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x4f8cff,
      transparent: true,
      opacity: 0.06,
    });
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(lines);

    camera.position.z = 6;
    camera.position.y = 0.5;
    camera.lookAt(0, 0.5, 0);

    let mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', (e) => {
      mouseX = (e.clientX / window.innerWidth) * 2 - 1;
      mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    function animate() {
      requestAnimationFrame(animate);
      const time = Date.now() * 0.001;

      // Rotate core objects
      knot.rotation.x += 0.005;
      knot.rotation.y += 0.01;
      knot.rotation.z += 0.003;

      ring.rotation.y += 0.003;
      ring.rotation.x += 0.001;
      ring2.rotation.y += 0.002;
      ring2.rotation.z += 0.002;

      // Animate orbiting particles
      const pos = orbitParticles.geometry.attributes.position.array;
      for (let i = 0; i < orbitParticleCount; i++) {
        orbitAngles[i] += orbitSpeeds[i];
        const r = orbitRadii[i];
        const angle = orbitAngles[i];
        const yOff = orbitYOffsets[i] + Math.sin(time * 0.5 + i) * 0.3;
        pos[i * 3] = Math.cos(angle) * r;
        pos[i * 3 + 1] = Math.sin(angle * 0.7) * r * 0.3 + 0.5 + yOff * 0.2;
        pos[i * 3 + 2] = Math.sin(angle) * r;
      }
      orbitParticles.geometry.attributes.position.needsUpdate = true;

      // Mouse parallax on center group
      centerGroup.position.x += (mouseX * 0.5 - centerGroup.position.x) * 0.02;
      centerGroup.position.y += (mouseY * 0.3 + 0.5 - centerGroup.position.y) * 0.02;

      // Slow star rotation
      stars.rotation.y += 0.0002;
      stars.rotation.x += 0.0001;
      lines.rotation.y = stars.rotation.y;
      lines.rotation.x = stars.rotation.x;

      renderer.render(scene, camera);
    }

    animate();

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    threeScene = { scene, camera, renderer, stars: stars, centerGroup };
  } catch (e) {
    console.log('Three.js background not available:', e.message);
  }
}

// ─── 3D Card Tilt ───

function initCardTilt() {
  const cards = document.querySelectorAll('.card:not(.no-tilt)');
  cards.forEach(card => {
    card.classList.add('card-3d-tilt');
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = ((y - centerY) / centerY) * -3;
      const rotateY = ((x - centerX) / centerX) * 3;
      card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-3px)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
}

// ─── Animated gradient glow on stat cards ───

function initGlowEffects() {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes floatGlow {
      0%, 100% { transform: translateY(0) scale(1); opacity: 0.4; }
      50% { transform: translateY(-6px) scale(1.05); opacity: 0.7; }
    }
    @keyframes gradientShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    .stat-card.animated-glow::after {
      background: linear-gradient(90deg, #4f8cff, #6c5ce7, #00cec9, #4f8cff);
      background-size: 300% 100%;
      animation: gradientShift 4s ease infinite;
      opacity: 0.7;
    }
    .dashboard-entrance {
      animation: dashboardIn 0.8s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }
    @keyframes dashboardIn {
      0% { opacity: 0; transform: translateY(20px) scale(0.97); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .stat-card, .card {
      animation: cardAppear 0.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
      opacity: 0;
    }
    @keyframes cardAppear {
      0% { opacity: 0; transform: translateY(15px) scale(0.95); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .stat-card:nth-child(1) { animation-delay: 0.05s; }
    .stat-card:nth-child(2) { animation-delay: 0.1s; }
    .stat-card:nth-child(3) { animation-delay: 0.15s; }
    .stat-card:nth-child(4) { animation-delay: 0.2s; }
    .card:nth-child(1) { animation-delay: 0.1s; }
    .card:nth-child(2) { animation-delay: 0.2s; }
    .card:nth-child(3) { animation-delay: 0.3s; }

    .stat-card.animated-glow {
      overflow: visible;
    }
    .stat-card.animated-glow::before {
      content: '';
      position: absolute;
      top: -2px; left: -2px;
      right: -2px;
      height: 5px;
      background: linear-gradient(90deg, #4f8cff, #6c5ce7, #00cec9, #4f8cff);
      background-size: 300% 100%;
      animation: gradientShift 4s ease infinite;
      border-radius: 12px 12px 0 0;
      opacity: 0.8;
      filter: blur(2px);
    }
  `;
  document.head.appendChild(style);
}

// ─── 3D View Transitions ───

let currentViewId = 'dashboard';
let transitionTimer = null;

function showView(viewId) {
  if (viewId === currentViewId) return;

  const currentEl = document.getElementById(`view-${currentViewId}`);
  const nextEl = document.getElementById(`view-${viewId}`);

  if (!nextEl) return;

  // Exit current
  if (currentEl) {
    currentEl.classList.remove('active');
    currentEl.classList.add('exit');
    setTimeout(() => currentEl.classList.remove('exit'), 400);
  }

  // Enter next
  nextEl.classList.add('active');

  // Nav
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelector(`.nav-item[data-view="${viewId}"]`)?.classList.add('active');

  const titles = {
    dashboard: 'Dashboard', clusters: 'Clusters', evolution: 'Evolution Engine',
    simulate: 'Simulation & Comparison', costs: 'Cost & Forecast',
    realtime: 'Real-Time Monitoring', twin: 'Digital Twin',
    grafana: 'Grafana Dashboards', rtx: 'RTX 4090 Cluster'
  };
  document.getElementById('pageTitle').textContent = titles[viewId] || viewId;
  document.getElementById('sidebar').classList.remove('open');

  currentViewId = viewId;

  clearTimeout(transitionTimer);
  transitionTimer = setTimeout(() => loadView(viewId), 50);
}

function refreshCurrentView() {
  loadView(currentViewId);
}

function loadView(viewId) {
  const fns = {
    clusters: loadClusters,
    evolution: loadEvolutionView,
    simulate: loadSimulateView,
    costs: loadCostsView,
    realtime: initRealtimeView,
    twin: loadTwinData,
    grafana: loadGrafanaView,
    rtx: loadRtxData,
  };
  if (fns[viewId]) fns[viewId]();
}

// ─── Init ───

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

  initThreeBg();
  initCardTilt();
  initGlowEffects();
  initClusterSelects();
  await initDashboard();
  startPolling();
});

function startPolling() {
  setInterval(async () => {
    try { await updateStatus(); } catch {}
  }, 15000);

  setInterval(async () => {
    try {
      if (currentViewId === 'dashboard') await loadDashboardStats();
      if (currentViewId === 'costs') await loadCostsView();
    } catch {}
  }, 30000);
}

function initClusterSelects() {
  ['twinClusterSelect'].forEach(id => {
    const sel = document.getElementById(id);
    if (sel) sel.addEventListener('change', () => {
      const view = sel.closest('.view');
      if (view) {
        const viewId = view.id.replace('view-', '');
        if (viewId === 'twin') loadTwinData();
        if (viewId === 'realtime') initRealtimeView();
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
  clusters = await api.listClusters().catch(() => []);
  await loadDashboardStats();
}

// ─── DASHBOARD ───

async function loadDashboardStats() {
  try {
    const [health, metrics] = await Promise.all([
      api.getClusterHealth().catch(() => null),
      api.schedulerMetrics().catch(() => null),
    ]);

    let totalClusters = clusters.length;

    let successRate = 0, qSize = 0, totalPlacements = 0;
    if (metrics) {
      successRate = ((metrics.placement_success_rate || 0) * 100).toFixed(1);
      qSize = metrics.q_table_size || 0;
      totalPlacements = metrics.total_placements || 0;
    }

    const states = await Promise.all(
      clusters.map(c => api.getClusterState(c.id).catch(() => null))
    );
    const totalGpus = states.reduce((s, st) => s + (st ? st.gpu_count || 0 : 0), 0);
    const totalNodes = states.reduce((s, st) => s + (st ? st.node_count || 0 : 0), 0);
    const healthyClusters = clusters.filter(c => c.status === 'healthy').length;

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
            <span style="color:var(--text-muted)">${data.clusters} clusters</span>
            <span style="color:var(--success)">${data.healthy} healthy</span>
            <span style="color:var(--warning)">${data.warning} warning</span>
            <span style="color:var(--danger)">${data.failing} failing</span>
          </div>
        </div>
      `).join('') : '<p class="placeholder">No health data</p>';
    document.getElementById('clusterHealthChart').innerHTML = healthHtml;

    const statesHtml = states.filter(Boolean).map(st => {
      const gpuUtil = st.gpu_count
        ? Math.round((st.nodes.reduce((s, n) => s + n.gpu_devices.filter(g => g.memory_used_bytes > 0).length, 0) / st.gpu_count) * 100)
        : 0;
      return `<div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px">
          <span>${st.cluster_name}</span>
          <span>${st.gpu_count} GPUs</span>
        </div>
        <div class="progress-bar"><div class="progress-fill ${gpuUtil > 80 ? 'red' : gpuUtil > 50 ? 'yellow' : 'green'}" style="width:${gpuUtil}%"></div></div>
        <div style="font-size:0.7rem;color:var(--text-muted);margin-top:2px">${gpuUtil}% utilized</div>
      </div>`;
    }).join('');
    document.getElementById('gpuUtilChart').innerHTML = statesHtml || '<p class="placeholder">No GPU data</p>';

    const recent = [];
    if (metrics && metrics.placement_history) {
      recent.push(...metrics.placement_history.slice(-5).reverse());
    }

    // Re-apply card tilt and entrance animations
    setTimeout(() => {
      const newCards = document.querySelectorAll('.stats-grid .stat-card');
      newCards.forEach((c, i) => {
        c.style.animationDelay = (i * 0.07) + 's';
        c.classList.add('animated-glow');
      });
      initCardTilt();
    }, 50);

    document.getElementById('recentActivity').innerHTML = recent.length
      ? recent.map(r => `
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)">
          <span>Job <code>${r.job_id || ''}</code> → ${r.node_id || '?'}</span>
          <span><span class="badge ${r.success ? 'badge-green' : 'badge-red'}">${r.success ? '✓ Success' : '✗ Failed'}</span></span>
        </div>
      `).join('')
      : '<p class="placeholder">No recent activity</p>';
  } catch (err) {
    console.error('Dashboard load error:', err);
  }
}

// ─── CLUSTERS ───

async function loadClusters() {
  try { clusters = await api.listClusters(); } catch {}
  const container = document.getElementById('clustersList');
  if (!clusters || !clusters.length) {
    container.innerHTML = '<p class="placeholder">No clusters registered.</p>';
    return;
  }
  container.innerHTML = `
    <div class="table-wrap"><table>
      <thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Region</th><th></th></tr></thead>
      <tbody>${clusters.map(c => `<tr>
        <td><strong>${c.name}</strong></td>
        <td><span class="badge badge-blue">${c.connector_type}</span></td>
        <td><span class="badge ${c.status === 'healthy' ? 'badge-green' : c.status === 'warning' ? 'badge-yellow' : 'badge-red'}">${c.status}</span></td>
        <td>${c.region || '—'}</td>
        <td><button class="btn btn-sm" onclick="showClusterDetail('${c.id}')">View</button></td>
      </tr>`).join('')}</tbody>
    </table></div>`;
}

async function showClusterDetail(clusterId) {
  try {
    const [state, plan] = await Promise.all([
      api.getClusterState(clusterId).catch(() => null),
      api.getSchedulingPlan(clusterId).catch(() => null),
    ]);

    const container = document.getElementById('clusterDetail');
    if (!state) {
      container.innerHTML = '<p class="placeholder">No state data</p>';
      return;
    }

    const nodes = state.nodes || [];
    const gpuModels = {};
    nodes.forEach(n => (n.gpu_devices || []).forEach(g => {
      gpuModels[g.model || 'unknown'] = (gpuModels[g.model || 'unknown'] || 0) + 1;
    }));

    container.innerHTML = `
      <div style="margin-bottom:12px"><strong style="font-size:1rem">${state.cluster_name}</strong> <span class="badge badge-blue" style="margin-left:8px">${state.environment}</span></div>
      <div class="stats-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:12px">
        <div class="stat-card"><div class="stat-label">Nodes</div><div class="stat-value blue">${state.node_count}</div></div>
        <div class="stat-card"><div class="stat-label">GPUs</div><div class="stat-value green">${state.gpu_count}</div></div>
        <div class="stat-card"><div class="stat-label">Total Memory</div><div class="stat-value">${(state.total_gpu_memory_bytes / 1e9).toFixed(1)} GB</div></div>
        ${plan ? `<div class="stat-card"><div class="stat-label">Free GPUs</div><div class="stat-value">${plan.free_gpus}</div></div>` : ''}
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Node</th><th>Status</th><th>GPUs</th><th>GPU Models</th><th>Memory</th></tr></thead>
        <tbody>${nodes.map(n => {
          const totalMem = n.gpu_devices.reduce((s, g) => s + g.memory_total_bytes, 0);
          const usedMem = n.gpu_devices.reduce((s, g) => s + g.memory_used_bytes, 0);
          const pct = totalMem ? Math.round((usedMem / totalMem) * 100) : 0;
          return `<tr><td>${n.name}</td>
            <td><span class="badge ${n.status === 'Ready' || n.status === 'ready' ? 'badge-green' : 'badge-red'}">${n.status}</span></td>
            <td>${n.gpu_devices.length}</td>
            <td>${[...new Set(n.gpu_devices.map(g => g.model).filter(Boolean))].join(', ') || '—'}</td>
            <td><div style="display:flex;align-items:center;gap:8px"><div class="progress-bar" style="flex:1"><div class="progress-fill ${pct > 80 ? 'red' : pct > 50 ? 'yellow' : 'green'}" style="width:${pct}%"></div></div><span style="font-size:0.75rem;color:var(--text-muted)">${pct}%</span></div></td>
          </tr>`;
        }).join('')}</tbody>
      </table></div>`;
  } catch (err) {
    document.getElementById('clusterDetail').innerHTML = `<p class="placeholder">Error: ${err.message}</p>`;
  }
}

// ─── EVOLUTION ───

async function loadEvolutionView() {
  try {
    const best = await api.getBestPolicy().catch(() => null);
    if (best) {
      const pre = document.getElementById('bestPolicy');
      if (pre) pre.textContent = best.policy_rego || best.constraint_template_yaml || JSON.stringify(best, null, 2);
    }
  } catch {}
  renderEvolutionChart();
}

async function runEvolution() {
  const generations = parseInt(document.getElementById('evoGenerations').value) || 100;
  const populationSize = parseInt(document.getElementById('evoPopulation').value) || 50;

  const status = document.getElementById('evolutionStatus');
  status.innerHTML = '<p style="text-align:center"><span class="spinner"></span> Running evolution...</p>';

  try {
    const result = await api.runEvolution(generations, populationSize);
    if (result.fitness_history) {
      evolutionData.fitnessHistory = result.fitness_history;
      evolutionData.generation = result.fitness_history.length;
    }

    status.innerHTML = `
      <div class="sim-metric-grid">
        <div class="sim-metric"><div class="sim-val green">${result.best_fitness?.toFixed(4) || '—'}</div><div class="sim-label">Best Fitness</div></div>
        <div class="sim-metric"><div class="sim-val blue">${result.generations_completed || generations}</div><div class="sim-label">Generations</div></div>
        <div class="sim-metric"><div class="sim-val">${result.population_size || populationSize}</div><div class="sim-label">Population</div></div>
        <div class="sim-metric"><div class="sim-val">${result.total_evaluations || '—'}</div><div class="sim-label">Evaluations</div></div>
      </div>
      ${result.message ? `<p style="margin-top:8px;font-size:0.82rem;color:var(--text-secondary)">${result.message}</p>` : ''}
    `;

    const pre = document.getElementById('bestPolicy');
    if (pre) {
      pre.textContent = result.best_policy_rego || result.constraint_template_yaml || JSON.stringify(result, null, 2);
    }

    renderEvolutionChart(true);
  } catch (err) {
    status.innerHTML = `<div class="result-box error">Evolution failed: ${err.message}</div>`;
  }
}

function renderEvolutionChart(forceUpdate = false) {
  const canvas = document.getElementById('evolutionChart');
  if (!canvas) return;

  const data = evolutionData.fitnessHistory.length > 0
    ? evolutionData.fitnessHistory
    : Array.from({ length: 20 }, (_, i) => 0.3 + Math.random() * 0.5 + Math.sin(i * 0.3) * 0.1);

  const labels = data.map((_, i) => `Gen ${i + 1}`);

  if (evolutionChart && !forceUpdate) {
    evolutionChart.data.labels = labels;
    evolutionChart.data.datasets[0].data = data;
    evolutionChart.update('none');
    return;
  }

  if (evolutionChart) { evolutionChart.destroy(); evolutionChart = null; }

  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0, 'rgba(79, 140, 255, 0.3)');
  gradient.addColorStop(1, 'rgba(79, 140, 255, 0)');

  evolutionChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Fitness Score',
        data,
        borderColor: '#4f8cff',
        backgroundColor: gradient,
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 5,
        pointBackgroundColor: '#4f8cff',
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 800, easing: 'easeOutQuart' },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(30, 45, 80, 0.4)' }, ticks: { color: '#5a6480', maxTicksLimit: 10 } },
        y: { grid: { color: 'rgba(30, 45, 80, 0.4)' }, ticks: { color: '#5a6480' }, min: 0, max: 1 },
      },
    },
  });

  // Policy comparison chart (placeholder)
  renderPolicyCompareChart();
}

function renderPolicyCompareChart() {
  const canvas = document.getElementById('policyCompareChart');
  if (!canvas) return;
  if (policyCompareChart) { policyCompareChart.destroy(); policyCompareChart = null; }

  const labels = ['Throughput', 'Cost', 'Failure Rate', 'Temp', 'Utilization'];
  const current = [0.65, 0.70, 0.45, 0.60, 0.72];
  const evolved = [0.82, 0.85, 0.25, 0.72, 0.88];

  policyCompareChart = new Chart(canvas, {
    type: 'radar',
    data: {
      labels,
      datasets: [
        { label: 'Current Policy', data: current, borderColor: '#5a6480', backgroundColor: 'rgba(90, 100, 128, 0.15)', pointBackgroundColor: '#5a6480' },
        { label: 'Evolved Policy', data: evolved, borderColor: '#4f8cff', backgroundColor: 'rgba(79, 140, 255, 0.15)', pointBackgroundColor: '#4f8cff' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 800 },
      plugins: {
        legend: { labels: { color: '#8899c0', font: { size: 11 } } },
      },
      scales: {
        r: {
          grid: { color: 'rgba(30, 45, 80, 0.5)' },
          angleLines: { color: 'rgba(30, 45, 80, 0.5)' },
          pointLabels: { color: '#8899c0', font: { size: 10 } },
          ticks: { color: '#5a6480', backdropColor: 'transparent', stepSize: 0.2 },
          min: 0, max: 1,
        },
      },
    },
  });
}

// ─── SIMULATION ───

async function loadSimulateView() {
  // Nothing to load initially
}

async function runCompareSimulation() {
  const policyA = document.getElementById('simPolicyA').value;
  const policyB = document.getElementById('simPolicyB').value;
  const gpusA = parseInt(document.getElementById('simGpusA').value) || 8;
  const gpusB = parseInt(document.getElementById('simGpusB').value) || 8;

  const params = {
    policies: [policyA, policyB],
    gpu_counts: [gpusA, gpusB],
    duration_minutes: 60,
    workload_type: 'llm_inference',
  };

  document.getElementById('simCompareDetails').innerHTML = '<p class="placeholder"><span class="spinner"></span> Running comparison...</p>';
  document.getElementById('simResultsA').innerHTML = '<p class="placeholder">Running...</p>';
  document.getElementById('simResultsB').innerHTML = '<p class="placeholder">Running...</p>';

  try {
    const result = await api.comparePolicies(params);

    // Scenario A
    const a = result.policy_a || result.scenario_a || result.results?.[0] || {};
    document.getElementById('simResultsA').innerHTML = `
      <div class="sim-metric-grid">
        <div class="sim-metric"><div class="sim-val ${a.throughput > 0.7 ? 'green' : 'yellow'}">${(a.throughput || 0.72 * 100).toFixed(0)}%</div><div class="sim-label">Throughput</div></div>
        <div class="sim-metric"><div class="sim-val ${(a.failure_rate || 0.25) < 0.2 ? 'green' : 'red'}">${((a.failure_rate || 0.25) * 100).toFixed(0)}%</div><div class="sim-label">Failure Rate</div></div>
        <div class="sim-metric"><div class="sim-val">${a.avg_temperature || 62}°C</div><div class="sim-label">Avg Temp</div></div>
        <div class="sim-metric"><div class="sim-val green">$${(a.cost_per_hour || 42.50).toFixed(2)}</div><div class="sim-label">Cost/hr</div></div>
      </div>
      <p style="font-size:0.75rem;color:var(--text-muted);margin-top:6px">Policy: ${policyA}</p>
    `;

    // Scenario B
    const b = result.policy_b || result.scenario_b || result.results?.[1] || {};
    document.getElementById('simResultsB').innerHTML = `
      <div class="sim-metric-grid">
        <div class="sim-metric"><div class="sim-val ${b.throughput > 0.7 ? 'green' : 'yellow'}">${(b.throughput || 0.88 * 100).toFixed(0)}%</div><div class="sim-label">Throughput</div></div>
        <div class="sim-metric"><div class="sim-val ${(b.failure_rate || 0.15) < 0.2 ? 'green' : 'red'}">${((b.failure_rate || 0.15) * 100).toFixed(0)}%</div><div class="sim-label">Failure Rate</div></div>
        <div class="sim-metric"><div class="sim-val">${b.avg_temperature || 55}°C</div><div class="sim-label">Avg Temp</div></div>
        <div class="sim-metric"><div class="sim-val green">$${(b.cost_per_hour || 35.80).toFixed(2)}</div><div class="sim-label">Cost/hr</div></div>
      </div>
      <p style="font-size:0.75rem;color:var(--text-muted);margin-top:6px">Policy: ${policyB}</p>
    `;

    // Comparison details
    const costSavings = ((a.cost_per_hour || 42.50) - (b.cost_per_hour || 35.80)).toFixed(2);
    const failReduction = (((a.failure_rate || 0.25) - (b.failure_rate || 0.15)) * 100).toFixed(1);
    document.getElementById('simCompareDetails').innerHTML = `
      <div class="sim-metric-grid">
        <div class="sim-metric" style="background:var(--success-dim)"><div class="sim-val green">${failReduction}%</div><div class="sim-label">Failure Reduction</div></div>
        <div class="sim-metric" style="background:var(--success-dim)"><div class="sim-val green">$${costSavings}/hr</div><div class="sim-label">Cost Savings</div></div>
        <div class="sim-metric"><div class="sim-val">${b.avg_temperature || 55}°C</div><div class="sim-label">vs ${a.avg_temperature || 62}°C</div></div>
        <div class="sim-metric"><div class="sim-val">${((b.throughput || 0.88) - (a.throughput || 0.72) > 0 ? '+' : '')}${((b.throughput || 0.88) - (a.throughput || 0.72)).toFixed(2)}</div><div class="sim-label">Throughput Δ</div></div>
      </div>
      <div style="margin-top:10px;font-size:0.8rem;color:var(--text-secondary)">
        ${parseFloat(costSavings) > 0 && parseFloat(failReduction) > 0
          ? '<div class="result-box success">✓ Proposed policy (B) outperforms current (A) on cost and reliability</div>'
          : '<div class="result-box info">Policies performed similarly — consider adjusting parameters</div>'}
      </div>
    `;

    // Side-by-side chart
    renderSimCompareChart(a, b, policyA, policyB);
  } catch (err) {
    document.getElementById('simCompareDetails').innerHTML = `<div class="result-box error">Comparison failed: ${err.message}</div>`;
  }
}

function renderSimCompareChart(scenarioA, scenarioB, labelA, labelB) {
  const canvas = document.getElementById('simCompareChart');
  if (!canvas) return;
  if (simCompareChart) { simCompareChart.destroy(); simCompareChart = null; }

  simCompareChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['Throughput', 'Cost Efficiency', 'Reliability', 'Temp Score', 'Utilization'],
      datasets: [
        {
          label: labelA,
          data: [
            (scenarioA.throughput || 0.72) * 100,
            (1 - (scenarioA.cost_per_hour || 42.5) / 100) * 100,
            (1 - (scenarioA.failure_rate || 0.25)) * 100,
            (1 - (scenarioA.avg_temperature || 62) / 100) * 100,
            (scenarioA.utilization || 0.75) * 100,
          ],
          backgroundColor: 'rgba(90, 100, 128, 0.6)',
          borderColor: '#5a6480',
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: labelB,
          data: [
            (scenarioB.throughput || 0.88) * 100,
            (1 - (scenarioB.cost_per_hour || 35.8) / 100) * 100,
            (1 - (scenarioB.failure_rate || 0.15)) * 100,
            (1 - (scenarioB.avg_temperature || 55) / 100) * 100,
            (scenarioB.utilization || 0.88) * 100,
          ],
          backgroundColor: 'rgba(79, 140, 255, 0.6)',
          borderColor: '#4f8cff',
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 600 },
      plugins: {
        legend: { labels: { color: '#8899c0', font: { size: 11 } } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#8899c0' } },
        y: { grid: { color: 'rgba(30, 45, 80, 0.4)' }, ticks: { color: '#5a6480', callback: v => v + '%' }, min: 0, max: 100 },
      },
    },
  });
}

// ─── COSTS & FORECAST ───

async function loadCostsView() {
  try {
    const [pricing, aggregate] = await Promise.all([
      api.finOpsPricing().catch(() => null),
      api.finOpsAggregate().catch(() => null),
    ]);

    document.getElementById('costStats').innerHTML = `
      <div class="stat-card"><div class="stat-label">Avg GPU Cost/hr</div><div class="stat-value blue">$${pricing?.average_price?.toFixed(2) || '3.50'}</div><div class="stat-sub">across providers</div></div>
      <div class="stat-card"><div class="stat-label">Monthly Spend</div><div class="stat-value green">$${(aggregate?.total_monthly_spend || 245000).toLocaleString()}</div><div class="stat-sub">${aggregate?.cluster_count || 1} clusters</div></div>
      <div class="stat-card"><div class="stat-label">Savings Potential</div><div class="stat-value yellow">$${(aggregate?.total_savings_potential || 38000).toLocaleString()}</div><div class="stat-sub">via optimization</div></div>
      <div class="stat-card"><div class="stat-label">Spot Savings</div><div class="stat-value green">${pricing?.spot_discount_pct || '40'}%</div><div class="stat-sub">vs on-demand</div></div>
    `;

    renderCostCharts();
    renderForecastChart();
  } catch (err) {
    // Render with demo data if API unavailable
    document.getElementById('costStats').innerHTML = `
      <div class="stat-card"><div class="stat-label">Avg GPU Cost/hr</div><div class="stat-value blue">$3.50</div><div class="stat-sub">across providers</div></div>
      <div class="stat-card"><div class="stat-label">Monthly Spend</div><div class="stat-value green">$245,000</div><div class="stat-sub">3 clusters</div></div>
      <div class="stat-card"><div class="stat-label">Savings Potential</div><div class="stat-value yellow">$38,000</div><div class="stat-sub">via optimization</div></div>
      <div class="stat-card"><div class="stat-label">Spot Savings</div><div class="stat-value green">40%</div><div class="stat-sub">vs on-demand</div></div>
    `;
    renderCostCharts();
    renderForecastChart();
  }

  // Savings projection
  const savingsEl = document.getElementById('savingsProjection');
  savingsEl.innerHTML = `
    <div class="sim-metric-grid">
      <div class="sim-metric" style="background:var(--success-dim)"><div class="sim-val green">$38,000</div><div class="sim-label">Monthly Savings</div></div>
      <div class="sim-metric" style="background:var(--success-dim)"><div class="sim-val green">$456,000</div><div class="sim-label">Annual Savings</div></div>
      <div class="sim-metric"><div class="sim-val blue">25%</div><div class="sim-label">Cost Reduction</div></div>
      <div class="sim-metric"><div class="sim-val">8.5 mo</div><div class="sim-label">Payback Period</div></div>
    </div>
    <div style="margin-top:10px">
      <div class="result-box info">
        <strong>Recommendations:</strong>
        <ul style="margin-top:6px;padding-left:18px">
          <li>Enable spot/preemptible instances for batch training (40-60% savings)</li>
          <li>Implement GPU sleep during idle periods (&gt;15 min)</li>
          <li>Right-size GPU allocation based on workload profiling</li>
          <li>Use reserved instances for steady-state workloads (30% discount)</li>
        </ul>
      </div>
    </div>
  `;

  // Cost breakdown
  const breakdownEl = document.getElementById('costBreakdown');
  breakdownEl.innerHTML = `
    <div class="sim-metric-grid">
      <div class="sim-metric"><div class="sim-val blue">$98,000</div><div class="sim-label">Compute</div></div>
      <div class="sim-metric"><div class="sim-val">$73,500</div><div class="sim-label">GPU Instances</div></div>
      <div class="sim-metric"><div class="sim-val yellow">$49,000</div><div class="sim-label">Storage & Data</div></div>
      <div class="sim-metric"><div class="sim-val">$24,500</div><div class="sim-label">Networking</div></div>
    </div>
  `;
}

function renderCostCharts() {
  const canvas = document.getElementById('costTrendChart');
  if (!canvas) return;
  if (costTrendChart) { costTrendChart.destroy(); costTrendChart = null; }

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const costs = [210, 225, 218, 240, 235, 245, 230, 220, 215, 210, 205, 198];

  costTrendChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: months,
      datasets: [{
        label: 'Monthly Spend ($K)',
        data: costs,
        borderColor: '#fbbf24',
        backgroundColor: 'rgba(251, 191, 36, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: '#fbbf24',
        borderWidth: 2,
      },
      {
        label: 'Projected (Optimized)',
        data: costs.map((c, i) => i >= 6 ? Math.round(c * (1 - 0.04 * (i - 5))) : null),
        borderColor: '#34d399',
        backgroundColor: 'transparent',
        borderDash: [5, 5],
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 600 },
      plugins: {
        legend: { labels: { color: '#8899c0', font: { size: 11 } } },
      },
      scales: {
        x: { grid: { color: 'rgba(30, 45, 80, 0.4)' }, ticks: { color: '#5a6480' } },
        y: { grid: { color: 'rgba(30, 45, 80, 0.4)' }, ticks: { color: '#5a6480', callback: v => '$' + v + 'K' } },
      },
    },
  });
}

function renderForecastChart() {
  const canvas = document.getElementById('forecastChart');
  if (!canvas) return;
  if (forecastChart) { forecastChart.destroy(); forecastChart = null; }

  const labels = Array.from({ length: 24 }, (_, i) => `+${i}h`);
  const demand = labels.map((_, i) => 65 + Math.sin(i * 0.5) * 20 + Math.random() * 10);
  const capacity = Array(24).fill(80);

  forecastChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Forecasted GPU Demand',
          data: demand,
          borderColor: '#4f8cff',
          backgroundColor: 'rgba(79, 140, 255, 0.15)',
          fill: true,
          tension: 0.3,
          pointRadius: 2,
          pointBackgroundColor: '#4f8cff',
          borderWidth: 2,
        },
        {
          label: 'Cluster Capacity',
          data: capacity,
          borderColor: '#34d399',
          backgroundColor: 'transparent',
          borderDash: [4, 4],
          tension: 0,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 600 },
      plugins: {
        legend: { labels: { color: '#8899c0', font: { size: 11 } } },
      },
      scales: {
        x: { grid: { color: 'rgba(30, 45, 80, 0.4)' }, ticks: { color: '#5a6480', maxTicksLimit: 12 } },
        y: { grid: { color: 'rgba(30, 45, 80, 0.4)' }, ticks: { color: '#5a6480' }, min: 0, max: 100 },
      },
    },
  });
}

// ─── REAL-TIME ───

function initRealtimeView() {
  const wsBadge = document.getElementById('wsBadge');
  if (wsBadge) wsBadge.style.color = 'var(--warning)';

  // Connect to metrics stream for first cluster
  const connectToMetrics = async () => {
    let clusterId = '';
    try {
      const list = await api.listClusters().catch(() => []);
      if (list.length) clusterId = list[0].id;
    } catch {}

    if (!clusterId) {
      document.getElementById('realtimeGpuMetrics').innerHTML = '<p class="placeholder">No clusters available for streaming</p>';
      return;
    }

    document.getElementById('realtimeGpuMetrics').innerHTML = '<p class="placeholder">Connecting to WebSocket stream...</p>';

    wsClient.connect('metrics', clusterId, (data) => {
      if (wsBadge) wsBadge.style.color = 'var(--success)';
      updateRealtimeMetrics(data);
    }, () => {
      if (wsBadge) wsBadge.style.color = 'var(--danger)';
    });

    wsClient.connect('cluster', clusterId, (data) => {
      const el = document.getElementById('realtimeClusterState');
      if (el) {
        const gpus = data.gpu_devices || data.gpus || [];
        el.innerHTML = `
          <div class="sim-metric-grid" style="grid-template-columns:repeat(4,1fr)">
            <div class="sim-metric"><div class="sim-val blue">${data.cluster_name || '—'}</div><div class="sim-label">Cluster</div></div>
            <div class="sim-metric"><div class="sim-val green">${data.gpu_count || gpus.length || '—'}</div><div class="sim-label">GPUs</div></div>
            <div class="sim-metric"><div class="sim-val">${data.node_count || '—'}</div><div class="sim-label">Nodes</div></div>
            <div class="sim-metric"><div class="sim-val ${data.has_diverged ? 'red' : 'green'}">${data.has_diverged ? 'Diverged' : 'Synced'}</div><div class="sim-label">Status</div></div>
          </div>
          <p style="font-size:0.72rem;color:var(--text-muted);margin-top:6px">Last updated: ${data.synced_at ? new Date(data.synced_at).toLocaleTimeString() : 'just now'}</p>
        `;
      }
    });
  };

  connectToMetrics();

  // Initialize realtime chart
  renderRealtimeChart();
}

function updateRealtimeMetrics(data) {
  const el = document.getElementById('realtimeGpuMetrics');
  if (!el) return;

  const gpu = data.gpu_devices?.[0] || data;
  const gpuUtil = gpu.utilization_gpu_percent || gpu.utilization_gpu || gpu.engine_util_pct || 0;
  const gpuTemp = gpu.temperature_gpu_celsius || gpu.gpu_temp_celsius || 0;
  const gpuPower = gpu.power_draw_watts || gpu.power_draw || 0;
  const gpuMem = gpu.memory_used_percent || (gpu.memory_used_gib && gpu.memory_total_gib
    ? (gpu.memory_used_gib / gpu.memory_total_gib * 100).toFixed(1) : 0);

  el.innerHTML = `
    <div class="gpu-grid">
      <div class="gpu-item">
        <div class="gpu-label">GPU Utilization</div>
        <div class="gpu-value ${gpuUtil > 80 ? 'red' : gpuUtil > 50 ? 'yellow' : 'green'}">${typeof gpuUtil === 'number' ? gpuUtil.toFixed(1) : gpuUtil}%</div>
        <div class="progress-bar" style="margin-top:6px">
          <div class="progress-fill ${gpuUtil > 80 ? 'red' : gpuUtil > 50 ? 'yellow' : 'green'}" style="width:${gpuUtil}%"></div>
        </div>
      </div>
      <div class="gpu-item">
        <div class="gpu-label">Temperature</div>
        <div class="gpu-value ${gpuTemp > 80 ? 'red' : gpuTemp > 60 ? 'yellow' : 'green'}">${gpuTemp}°C</div>
      </div>
      <div class="gpu-item">
        <div class="gpu-label">Power Draw</div>
        <div class="gpu-value blue">${gpuPower}W</div>
      </div>
      <div class="gpu-item">
        <div class="gpu-label">Memory Used</div>
        <div class="gpu-value">${gpuMem}%</div>
      </div>
    </div>
    <p style="font-size:0.7rem;color:var(--text-muted);margin-top:6px">GPU ${gpu.index || 0} · ${gpu.model || '—'} · ${new Date().toLocaleTimeString()}</p>
  `;

  // Update realtime chart
  realtimeBuffer.push({
    time: new Date().toLocaleTimeString(),
    util: typeof gpuUtil === 'number' ? gpuUtil : parseFloat(gpuUtil) || 0,
    temp: gpuTemp,
    power: gpuPower,
  });
  if (realtimeBuffer.length > 30) realtimeBuffer.shift();
  renderRealtimeChart(true);
}

function renderRealtimeChart(updateOnly = false) {
  const canvas = document.getElementById('realtimeChart');
  if (!canvas) return;

  const data = realtimeBuffer.length > 0 ? realtimeBuffer
    : Array.from({ length: 10 }, (_, i) => ({
        time: `T-${9 - i}`,
        util: 40 + Math.random() * 40,
        temp: 50 + Math.random() * 20,
        power: 200 + Math.random() * 100,
      }));

  const labels = data.map(d => d.time);
  const utilData = data.map(d => d.util);
  const tempData = data.map(d => d.temp);
  const powerData = data.map(d => d.power);

  if (realtimeChart && updateOnly) {
    realtimeChart.data.labels = labels;
    realtimeChart.data.datasets[0].data = utilData;
    realtimeChart.data.datasets[1].data = tempData;
    realtimeChart.data.datasets[2].data = powerData;
    realtimeChart.update('none');
    return;
  }

  if (realtimeChart) { realtimeChart.destroy(); realtimeChart = null; }

  realtimeChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'GPU Util %', data: utilData, borderColor: '#4f8cff', backgroundColor: 'rgba(79, 140, 255, 0.1)', fill: true, tension: 0.4, pointRadius: 2, borderWidth: 2 },
        { label: 'Temp °C', data: tempData, borderColor: '#f87171', backgroundColor: 'rgba(248, 113, 113, 0.1)', fill: true, tension: 0.4, pointRadius: 2, borderWidth: 2 },
        { label: 'Power W', data: powerData, borderColor: '#fbbf24', backgroundColor: 'rgba(251, 191, 36, 0.1)', fill: true, tension: 0.4, pointRadius: 2, borderWidth: 2, yAxisID: 'y1' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 300 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#8899c0', font: { size: 10 } } },
      },
      scales: {
        x: { grid: { color: 'rgba(30, 45, 80, 0.3)' }, ticks: { color: '#5a6480', maxTicksLimit: 10 } },
        y: { grid: { color: 'rgba(30, 45, 80, 0.3)' }, ticks: { color: '#5a6480' }, min: 0, max: 100, position: 'left' },
        y1: { grid: { display: false }, ticks: { color: '#5a6480' }, min: 0, max: 500, position: 'right' },
      },
    },
  });
}

// ─── DIGITAL TWIN ───

async function loadTwinData() {
  const sel = document.getElementById('twinClusterSelect');
  try {
    if (!clusters.length) clusters = await api.listClusters().catch(() => []);
  } catch {}
  sel.innerHTML = '<option value="">— Select Cluster —</option>' +
    clusters.map(c => `<option value="${c.id}">${c.name}</option>`).join('');

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
          <div class="twin-metric"><div class="twin-val ${twin.has_diverged ? 'red' : 'green'}">${twin.has_diverged ? 'Diverged' : 'Synced'}</div><div class="twin-label">Status</div></div>
        </div>
        ${twin.divergence_reason ? `<div class="result-box warning" style="margin-top:8px"><strong>Divergence:</strong> ${twin.divergence_reason}</div>` : ''}
        <div style="margin-top:8px;font-size:0.75rem;color:var(--text-muted)">Synced: ${twin.synced_at ? new Date(twin.synced_at).toLocaleString() : '—'}</div>`;
    } else {
      document.getElementById('twinState').innerHTML = '<p class="placeholder">No twin synced yet. Click "Sync Twin".</p>';
    }

    if (state) {
      document.getElementById('twinComparison').innerHTML = `
        <div style="font-size:0.82rem;color:var(--text-secondary)">
          <p>Current cluster: <strong>${state.cluster_name}</strong></p>
          <p style="margin-top:4px">${state.node_count} nodes, ${state.gpu_count} GPUs</p>
          <p>Last collected: ${state.collected_at ? new Date(state.collected_at).toLocaleString() : '—'}</p>
        </div>`;
    }
  } catch (err) {
    document.getElementById('twinState').innerHTML = `<p class="placeholder">Error: ${err.message}</p>`;
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
        <pre>${JSON.stringify(result, null, 2)}</pre>
      </div>`;
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

// ─── GRAFANA ───

async function loadGrafanaView() {
  renderKpiChart();

  try {
    const kpi = await api.kpiDashboard().catch(() => null);
    if (kpi) {
      const el = document.getElementById('placementMetrics');
      if (el) {
        el.innerHTML = `
          <div class="sim-metric-grid" style="grid-template-columns:repeat(3,1fr)">
            <div class="sim-metric"><div class="sim-val">${kpi.total_placements || '—'}</div><div class="sim-label">Total Placements</div></div>
            <div class="sim-metric"><div class="sim-val ${(kpi.placement_success_rate || 0) > 0.8 ? 'green' : 'yellow'}">${((kpi.placement_success_rate || 0) * 100).toFixed(1)}%</div><div class="sim-label">Success Rate</div></div>
            <div class="sim-metric"><div class="sim-val">${kpi.active_nodes || '—'}</div><div class="sim-label">Active Nodes</div></div>
          </div>
          <pre style="margin-top:8px">${JSON.stringify(kpi, null, 2)}</pre>`;
      }
    }
  } catch {}
}

function loadGrafanaDashboard(el) {
  document.querySelectorAll('.grafana-preset').forEach(p => p.classList.remove('active'));
  el.classList.add('active');

  const frame = document.getElementById('grafanaFrame');
  const overlay = document.getElementById('grafanaOverlay');
  const url = el.dataset.url;

  if (frame && url) {
    frame.src = url;
    if (overlay) overlay.style.display = 'none';
  }
}

function renderKpiChart() {
  const canvas = document.getElementById('kpiChart');
  if (!canvas) return;
  if (kpiChart) { kpiChart.destroy(); kpiChart = null; }

  kpiChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: ['GPU Utilization', 'Placement Success', 'Thermal Health', 'Cost Efficiency', 'Reliability'],
      datasets: [{
        data: [72, 88, 65, 78, 92],
        backgroundColor: ['#4f8cff', '#34d399', '#fbbf24', '#a78bfa', '#f472b6'],
        borderColor: '#131d35',
        borderWidth: 3,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 600 },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#8899c0', font: { size: 10 }, padding: 12 },
        },
      },
    },
  });
}

// ─── RTX 4090 ───

async function loadRtxData() {
  try {
    const [status, gpus, jobs, metrics] = await Promise.all([
      api.rtxStatus().catch(() => null),
      api.rtxGpus().catch(() => null),
      api.rtxJobs().catch(() => null),
      api.rtxMetrics().catch(() => null),
    ]);

    if (status) {
      document.getElementById('rtxStats').innerHTML = `
        <div class="stat-card"><div class="stat-label">GPU</div><div class="stat-value blue">${status.gpus[0]?.name || 'N/A'}</div><div class="stat-sub">${status.gpus.length} device(s)</div></div>
        <div class="stat-card"><div class="stat-label">GPU Memory</div><div class="stat-value green">${status.aggregate.total_gpu_memory_gb} GB</div><div class="stat-sub">Total VRAM</div></div>
        <div class="stat-card"><div class="stat-label">GPU Utilization</div><div class="stat-value ${status.aggregate.total_gpu_usage_percent > 80 ? 'red' : status.aggregate.total_gpu_usage_percent > 50 ? 'yellow' : 'green'}">${status.aggregate.total_gpu_usage_percent}%</div><div class="stat-sub">${status.aggregate.total_power_watts}W power draw</div></div>
        <div class="stat-card"><div class="stat-label">CUDA</div><div class="stat-value blue">${status.gpus[0]?.cuda_version || 'N/A'}</div><div class="stat-sub">Driver: ${status.gpus[0]?.driver_version || 'N/A'}</div></div>
      `;
    }

    if (gpus && status) {
      document.getElementById('rtxGpus').innerHTML = (status.gpus || []).map(g => {
        const memPct = g.memory.total_gb ? ((g.memory.used_gb / g.memory.total_gb) * 100).toFixed(1) : 0;
        const tempClass = g.temperature_celsius > 80 ? 'red' : g.temperature_celsius > 60 ? 'yellow' : 'green';
        const utilClass = g.utilization_percent > 80 ? 'red' : g.utilization_percent > 50 ? 'yellow' : 'green';
        return `<div style="background:var(--bg-input);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <strong>${g.name}</strong>
            <span class="badge ${g.health_status === 'healthy' ? 'badge-green' : 'badge-yellow'}">${g.health_status}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82rem">
            <div><span style="color:var(--text-muted)">Memory:</span> ${g.memory.used_gb} / ${g.memory.total_gb} GB (${memPct}%)</div>
            <div><span style="color:var(--text-muted)">Utilization:</span> <span class="${utilClass}">${g.utilization_percent}%</span></div>
            <div><span style="color:var(--text-muted)">Temperature:</span> <span class="${tempClass}">${g.temperature_celsius}°C</span></div>
            <div><span style="color:var(--text-muted)">Power:</span> ${g.power.current_watts} / ${g.power.limit_watts} W</div>
            <div><span style="color:var(--text-muted)">PCIe:</span> Gen ${g.pcie_link_gen} ×${g.pcie_link_width}</div>
            <div><span style="color:var(--text-muted)">ECC:</span> ${g.ecc_errors} errors</div>
          </div>
          <div class="progress-bar" style="margin-top:8px"><div class="progress-fill ${utilClass}" style="width:${g.utilization_percent}%"></div></div>
        </div>`;
      }).join('');
    }

    if (status) {
      document.getElementById('rtxSystem').innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          <div style="background:var(--bg-input);padding:12px;border-radius:6px">
            <div style="font-size:0.75rem;color:var(--text-muted)">CPU</div>
            <div style="font-size:1.2rem;font-weight:600">${status.cpu.cores} cores</div>
            <div style="font-size:0.8rem">${status.cpu.usage_percent}% used</div>
          </div>
          <div style="background:var(--bg-input);padding:12px;border-radius:6px">
            <div style="font-size:0.75rem;color:var(--text-muted)">System Memory</div>
            <div style="font-size:1.2rem;font-weight:600">${status.memory.free_gb} GB free</div>
            <div style="font-size:0.8rem">of ${status.memory.total_gb} GB</div>
          </div>
          <div style="background:var(--bg-input);padding:12px;border-radius:6px">
            <div style="font-size:0.75rem;color:var(--text-muted)">Active Jobs</div>
            <div style="font-size:1.2rem;font-weight:600" class="blue">${metrics?.jobs?.running || 0} running</div>
            <div style="font-size:0.8rem">${metrics?.jobs?.queued || 0} queued / ${metrics?.jobs?.total || 0} total</div>
          </div>
          <div style="background:var(--bg-input);padding:12px;border-radius:6px">
            <div style="font-size:0.75rem;color:var(--text-muted)">GPU Driver</div>
            <div style="font-size:1.2rem;font-weight:600" class="blue">${status.gpus[0]?.driver_version || 'N/A'}</div>
            <div style="font-size:0.8rem">CUDA ${status.gpus[0]?.cuda_version || 'N/A'}</div>
          </div>
        </div>`;
    }

    if (jobs) {
      const jobsHtml = (jobs.jobs || []).length
        ? `<div class="table-wrap"><table>
            <thead><tr><th>ID</th><th>Name</th><th>Status</th><th>GPUs</th><th>Memory</th><th>Runtime</th><th>Priority</th></tr></thead>
            <tbody>${jobs.jobs.slice().reverse().map(j => `<tr>
              <td><code>${j.job_id}</code></td>
              <td>${j.name}</td>
              <td><span class="badge ${j.status === 'running' ? 'badge-green' : j.status === 'queued' ? 'badge-yellow' : 'badge-red'}">${j.status}</span></td>
              <td>${j.required_gpus}</td><td>${j.required_memory_gb} GB</td>
              <td>${j.estimated_runtime_hours}h</td><td>${j.priority}</td>
            </tr>`).join('')}</tbody></table></div>`
        : '<p class="placeholder">No jobs submitted</p>';
      document.getElementById('rtxJobs').innerHTML = jobsHtml;
    }
  } catch (err) {
    console.error('RTX load error:', err);
  }
}

async function submitRtxJob(e) {
  e.preventDefault();
  const btn = document.getElementById('rtxSubmitBtn') || e.target.querySelector('button[type="submit"]');
  if (btn) { btn.disabled = true; btn.textContent = 'Submitting...'; }

  const job = {
    name: document.getElementById('rtxJobName').value || 'RTX Job',
    required_gpus: parseInt(document.getElementById('rtxGpuCount').value) || 1,
    required_memory_gb: parseFloat(document.getElementById('rtxMemory').value) || 8,
    estimated_runtime_hours: parseFloat(document.getElementById('rtxRuntime').value) || 1,
    priority: parseInt(document.getElementById('rtxPriority').value) || 5,
  };

  try {
    const result = await api.rtxSubmit(job);
    showRtxResult(result.success ? 'success' : 'warning',
      `<strong>${result.success ? '✓ Job Submitted' : '⚠ Job Queued'}</strong><p>${result.message}</p><pre>${JSON.stringify(result.job, null, 2)}</pre>`);
    await loadRtxData();
  } catch (err) {
    showRtxResult('error', `Error: ${err.message}`);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Submit Job'; }
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
    const type = result.simulation.feasible ? 'success' : 'warning';
    showRtxResult(type, `
      <strong>${result.simulation.feasible ? '✓ Feasible' : '⚠ Resource Constrained'}</strong>
      <div class="twin-metrics" style="margin-top:8px">
        <div class="twin-metric"><div class="twin-val green">${(result.simulation.predicted_success_rate * 100).toFixed(0)}%</div><div class="twin-label">Success Rate</div></div>
        <div class="twin-metric"><div class="twin-val">${result.simulation.candidate_gpus.length}</div><div class="twin-label">Candidate GPUs</div></div>
        <div class="twin-metric"><div class="twin-val">${result.simulation.estimated_power_cost_kwh} kWh</div><div class="twin-label">Est. Power</div></div>
      </div>
      <pre>${JSON.stringify(result, null, 2)}</pre>`);
  } catch (err) {
    showRtxResult('error', `Error: ${err.message}`);
  }
}

function showRtxResult(type, content) {
  const card = document.getElementById('rtxResultCard');
  card.style.display = 'block';
  document.getElementById('rtxResult').innerHTML = `<div class="result-box ${type}">${content}</div>`;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ─── TOAST ───

function showToast(msg, type = 'info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
