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

  health() { return this.get('/api/v2/health'); },
  listClusters() { return this.get('/api/v2/clusters'); },
  getCluster(id) { return this.get(`/api/v2/clusters/${id}`); },
  getClusterState(id) { return this.get(`/api/v1/clusters/${id}/state`); },
  getGpuUsage(id) { return this.get(`/api/v1/inference/clusters/${id}/gpu-usage`); },
  getTwin(id) { return this.get(`/api/v1/clusters/${id}/twin`); },
  syncTwin(id) { return this.post(`/api/v1/clusters/${id}/twin`, { force_sync: true }); },
  compareTwin(id) { return this.post(`/api/v1/clusters/${id}/twin/compare`, {}); },
  resetTwin(id) { return this.post(`/api/v1/clusters/${id}/twin/reset`, {}); },
  schedulerMetrics() { return this.get('/api/v2/scheduler-metrics'); },
  getDomainCounts() { return this.get('/api/v2/domains'); },
  getGpuSnapshot(clusterId) { return this.get(`/api/v1/monitoring/gpu/snapshot?cluster_id=${clusterId}`); },

  submitJob(job) { return this.post('/api/v2/submit-job', job); },
  predict(telemetry) { return this.post('/api/v2/predict', telemetry); },
  analyzeCluster(clusterId, nodeCount = 4) {
    return this.post('/api/v1/predictor/analyze-cluster', { cluster_id: clusterId, node_count: nodeCount });
  },

  getClusterHealth() {
    return this.get('/api/v1/state/summary');
  },

  getPlacement(clusterId, req) {
    return this.post(`/api/v1/clusters/${clusterId}/scheduler/placement`, req);
  },

  getSchedulingPlan(clusterId) {
    return this.get(`/api/v1/clusters/${clusterId}/scheduler/plan`);
  },

  // RTX 4090 endpoints
  rtxStatus() { return this.get('/api/v1/rtx/status'); },
  rtxGpus() { return this.get('/api/v1/rtx/gpus'); },
  rtxJobs() { return this.get('/api/v1/rtx/jobs'); },
  rtxSubmit(job) { return this.post('/api/v1/rtx/submit', job); },
  rtxSimulate(job) { return this.post('/api/v1/rtx/simulate', job); },
  rtxDetect() { return this.post('/api/v1/rtx/detect'); },
  rtxMetrics() { return this.get('/api/v1/rtx/metrics'); },
};