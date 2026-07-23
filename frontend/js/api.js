const API_BASE = '';

const api = {
  async request(method, path, body) {
    const opts = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(`${API_BASE}${path}`, opts);
    const data = await res.json();
    if (!res.ok && !data.status) throw new Error(data.detail || `HTTP ${res.status}`);
    return data;
  },

  get(path) { return this.request('GET', path); },
  post(path, body) { return this.request('POST', path, body); },

  // System
  health() { return this.get('/api/v2/health'); },

  // Clusters
  listClusters() { return this.get('/api/v2/clusters'); },
  getCluster(id) { return this.get(`/api/v2/clusters/${id}`); },
  getClusterState(id) { return this.get(`/api/v1/clusters/${id}/state`); },
  getGpuUsage(id) { return this.get(`/api/v1/inference/clusters/${id}/gpu-usage`); },
  getClusterHealth() { return this.get('/api/v1/state/summary'); },

  // Scheduling
  schedulerMetrics() { return this.get('/api/v2/scheduler-metrics'); },
  getPlacement(clusterId, req) { return this.post(`/api/v1/clusters/${clusterId}/scheduler/placement`, req); },
  getSchedulingPlan(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/scheduler/plan`); },
  simulatePlacement(clusterId, req) { return this.post(`/api/v1/clusters/${clusterId}/scheduler/simulate`, req); },
  forecastDemand(clusterId, horizonHours = 48) { return this.post(`/api/v1/clusters/${clusterId}/scheduler/forecast`, { horizon_hours: horizonHours }); },

  // Digital Twin
  getTwin(id) { return this.get(`/api/v1/clusters/${id}/twin`); },
  syncTwin(id) { return this.post(`/api/v1/clusters/${id}/twin`, { force_sync: true }); },
  compareTwin(id) { return this.post(`/api/v1/clusters/${id}/twin/compare`, {}); },
  resetTwin(id) { return this.post(`/api/v1/clusters/${id}/twin/reset`, {}); },

  // Monitor
  getDomainCounts() { return this.get('/api/v2/domains'); },
  getGpuSnapshot(clusterId) { return this.get(`/api/v1/monitoring/gpu/snapshot?cluster_id=${clusterId}`); },

  // Jobs
  submitJob(job) { return this.post('/api/v2/submit-job', job); },
  predict(telemetry) { return this.post('/api/v2/predict', telemetry); },
  analyzeCluster(clusterId, nodeCount = 4) { return this.post('/api/v1/predictor/analyze-cluster', { cluster_id: clusterId, node_count: nodeCount }); },

  // EVOLUTION
  runEvolution(generations = 100, populationSize = 50) {
    return this.post('/api/v1/policy/evolve', { generations, population_size: populationSize });
  },
  getBestPolicy() { return this.get('/api/v1/policy/best-policy'); },
  deployPolicy(template) { return this.post('/api/v1/policy/deploy-policy', { constraint_template_yaml: template }); },
  gatekeeperHealth() { return this.get('/api/v1/policy/gatekeeper-health'); },

  // SIMULATION
  simulateGpuWorkload(params) { return this.post('/api/v1/ml/simulate', params); },
  simulateEnhanced(params) { return this.post('/api/v1/ml/simulate-enhanced', params); },
  comparePolicies(params) { return this.post('/api/v1/ml/compare-policies', params); },
  simulateFailure(params) { return this.post('/api/v1/ml/simulate-failure', params); },
  closedLoopTrain(params) { return this.post('/api/v1/ml/closed-loop-train', params); },

  // COST / FINOPS
  costReport(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/costs/report`); },
  costProjections(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/costs/projections`); },
  costSummary(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/costs/summary`); },
  finOpsPricing() { return this.get('/api/v1/finops/pricing'); },
  finOpsCompare() { return this.get('/api/v1/finops/compare'); },
  finOpsAggregate() { return this.get('/api/v1/finops/aggregate'); },
  cloudPricing(provider) { return this.get(`/api/v1/cloud/pricing/${provider}`); },
  cloudCompare() { return this.get('/api/v1/cloud/compare'); },
  cloudEstimate(provider, gpuModel, count, hours, purchaseOption) {
    return this.get(`/api/v1/cloud/estimate?provider=${provider}&gpu_model=${gpuModel}&count=${count}&monthly_hours=${hours}&purchase_option=${purchaseOption}`);
  },
  costForecast(clusterId, months = 6) { return this.get(`/api/v1/finops/forecast/${clusterId}?months=${months}`); },

  // KPI / METRICS
  kpiDashboard() { return this.get('/api/v1/metrics-kpi/dashboard'); },
  kpiGpu() { return this.get('/api/v1/metrics-kpi/gpu'); },
  kpiThermal() { return this.get('/api/v1/metrics-kpi/thermal'); },
  kpiPlacement() { return this.get('/api/v1/metrics-kpi/placement'); },
  kpiEconomics() { return this.get('/api/v1/metrics-kpi/economics'); },

  // RTX 4090
  rtxStatus() { return this.get('/api/v1/rtx/status'); },
  rtxGpus() { return this.get('/api/v1/rtx/gpus'); },
  rtxJobs() { return this.get('/api/v1/rtx/jobs'); },
  rtxSubmit(job) { return this.post('/api/v1/rtx/submit', job); },
  rtxSimulate(job) { return this.post('/api/v1/rtx/simulate', job); },
  rtxDetect() { return this.post('/api/v1/rtx/detect'); },
  rtxMetrics() { return this.get('/api/v1/rtx/metrics'); },
};

// ─── WebSocket Client ───

const wsClient = {
  connections: {},
  handlers: {},

  connect(channel, clusterId, onMessage, onError) {
    const key = `${channel}:${clusterId || ''}`;
    if (this.connections[key]) return this.connections[key];

    let url = '';
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = 'localhost:8080';

    switch (channel) {
      case 'cluster': url = `${protocol}//${host}/api/v1/stream/cluster/${clusterId}`; break;
      case 'alerts': url = clusterId
        ? `${protocol}//${host}/api/v1/stream/alerts/${clusterId}`
        : `${protocol}//${host}/api/v1/stream/alerts`; break;
      case 'metrics': url = `${protocol}//${host}/api/v1/stream/metrics/${clusterId}`; break;
    }

    if (!url) return null;

    const ws = new WebSocket(url);

    ws.onopen = () => { console.log(`WS connected: ${key}`); };
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (onMessage) onMessage(data);
      } catch { if (onMessage) onMessage(evt.data); }
    };
    ws.onerror = (err) => { console.error(`WS error: ${key}`, err); if (onError) onError(err); };
    ws.onclose = () => {
      console.log(`WS closed: ${key}`);
      delete this.connections[key];
      setTimeout(() => this.connect(channel, clusterId, onMessage, onError), 5000);
    };

    this.connections[key] = ws;
    return ws;
  },

  disconnect(channel, clusterId) {
    const key = `${channel}:${clusterId || ''}`;
    if (this.connections[key]) {
      this.connections[key].close();
      delete this.connections[key];
    }
  },

  disconnectAll() {
    Object.values(this.connections).forEach(ws => ws.close());
    this.connections = {};
  },

  isConnected(channel, clusterId) {
    const key = `${channel}:${clusterId || ''}`;
    const ws = this.connections[key];
    return ws && ws.readyState === WebSocket.OPEN;
  },
};
