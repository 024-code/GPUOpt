const API_BASE = '';

const api = {
  async request(method, path, body) {
    const opts = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(`${API_BASE}${path}`, opts);
    if (res.status === 204) return {};
    const data = await res.json();
    if (!res.ok && !data.status) throw new Error(data.detail || `HTTP ${res.status}`);
    return data;
  },

  get(path) { return this.request('GET', path); },
  post(path, body) { return this.request('POST', path, body); },
  put(path, body) { return this.request('PUT', path, body); },
  patch(path, body) { return this.request('PATCH', path, body); },
  del(path) { return this.request('DELETE', path); },

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
  rtxPartitions() { return this.get('/api/v1/rtx/partitions'); },

  // ── Health (v1) ──
  healthLive() { return this.get('/health/live'); },
  healthReady() { return this.get('/health/ready'); },
  healthDetailed() { return this.get('/health/detailed'); },

  // ── Clusters CRUD ──
  createCluster(payload) { return this.post('/api/v1/clusters', payload); },
  upsertCluster(name, payload) { return this.put(`/api/v1/clusters/by-name/${encodeURIComponent(name)}`, payload); },
  deleteCluster(id) { return this.request('DELETE', `/api/v1/clusters/${id}`); },

  // ── Environment Checks ──
  runClusterCheck(clusterId) { return this.post(`/api/v1/clusters/${clusterId}/checks`); },
  latestClusterCheck(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/checks/latest`); },
  checkAllEnvironments() { return this.post('/api/v1/environments/check-all'); },
  environmentSummary() { return this.get('/api/v1/environments/summary'); },

  // ── Cluster State ──
  collectClusterState(clusterId) { return this.post(`/api/v1/clusters/${clusterId}/state`); },

  // ── Trace Replay ──
  listTraces(clusterId, limit = 50, offset = 0) { return this.get(`/api/v1/clusters/${clusterId}/traces?limit=${limit}&offset=${offset}`); },
  getTrace(clusterId, traceId) { return this.get(`/api/v1/clusters/${clusterId}/traces/${traceId}`); },
  replayTrace(clusterId, traceId = null) { const q = traceId ? `?trace_id=${traceId}` : ''; return this.post(`/api/v1/clusters/${clusterId}/replay${q}`); },
  setBaseline(clusterId) { return this.post(`/api/v1/clusters/${clusterId}/baseline`); },
  getBaseline(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/baseline`); },
  compareTraces(clusterId, traceIdA = null, traceIdB = null) { const q = traceIdA ? `?trace_id_a=${traceIdA}${traceIdB ? `&trace_id_b=${traceIdB}` : ''}` : ''; return this.post(`/api/v1/clusters/${clusterId}/compare${q}`); },

  // ── Workload Analysis ──
  analyzeClusterWorkload(clusterId, maxTraces = 100) { return this.post(`/api/v1/clusters/${clusterId}/analyze?max_traces=${maxTraces}`); },
  latestAnalysis(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/analysis/latest`); },
  listAnalyses(clusterId, limit = 10) { return this.get(`/api/v1/clusters/${clusterId}/analysis/list?limit=${limit}`); },

  // ── Recommendations ──
  generateRecommendations(clusterId) { return this.post(`/api/v1/clusters/${clusterId}/recommendations`); },
  latestRecommendations(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/recommendations/latest`); },
  listRecommendations(clusterId, limit = 10) { return this.get(`/api/v1/clusters/${clusterId}/recommendations/list?limit=${limit}`); },
  updateRecommendationStatus(clusterId, recId, status, reason = '') { return this.post(`/api/v1/clusters/${clusterId}/recommendations/${recId}/status`, { status, reason }); },
  whatIfRecommendations(clusterId) { return this.post(`/api/v1/clusters/${clusterId}/recommendations/what-if`); },

  // ── Actuation ──
  actuateRecommendation(clusterId, recId, dryRun = true, reason = '') { return this.post(`/api/v1/clusters/${clusterId}/actuate`, { rec_id: recId, dry_run: dryRun, reason }); },
  rollbackActuation(clusterId, actuationId) { return this.post(`/api/v1/clusters/${clusterId}/actuations/${actuationId}/rollback`); },
  actuationSummary(clusterId) { return this.get(`/api/v1/clusters/${clusterId}/actuations/summary`); },
  listActuations(clusterId, limit = 20) { return this.get(`/api/v1/clusters/${clusterId}/actuations?limit=${limit}`); },
  getActuation(clusterId, actuationId) { return this.get(`/api/v1/clusters/${clusterId}/actuations/${actuationId}`); },

  // ── Digital Twin (extended) ──
  applyToTwin(clusterId, recId) { return this.post(`/api/v1/clusters/${clusterId}/twin/apply/${recId}`); },

  // ── Training ──
  trainingRegisterJob(clusterId, jobName, framework = 'custom', gpuCount = 1, nodeCount = 1, batchSize = 0, precision = 'fp32', maxDurationHours = 0, metadata = '{}') {
    return this.post(`/api/v1/training/jobs?cluster_id=${clusterId}&job_name=${encodeURIComponent(jobName)}&framework=${framework}&gpu_count=${gpuCount}&node_count=${nodeCount}&batch_size=${batchSize}&precision=${precision}&max_duration_hours=${maxDurationHours}&metadata=${encodeURIComponent(metadata)}`);
  },
  trainingListJobs(clusterId = null) { const q = clusterId ? `?cluster_id=${clusterId}` : ''; return this.get(`/api/v1/training/jobs${q}`); },
  trainingGetJob(jobId) { return this.get(`/api/v1/training/jobs/${jobId}`); },
  trainingUpdateJob(jobId, updates) { const q = Object.entries(updates).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&'); return this.request('PATCH', `/api/v1/training/jobs/${jobId}?${q}`); },
  trainingDeleteJob(jobId) { return this.request('DELETE', `/api/v1/training/jobs/${jobId}`); },
  trainingProfileJob(jobId) { return this.post(`/api/v1/training/jobs/${jobId}/profile`); },
  trainingRunHPO(jobId, config = null) { return this.post(`/api/v1/training/jobs/${jobId}/hpo`, config); },
  trainingDistributedConfig(totalGpus, gpuModel = '', modelSizeGb = 0, perGpuMemoryGb = 80) {
    return this.post(`/api/v1/training/distributed-config?total_gpus=${totalGpus}&gpu_model=${encodeURIComponent(gpuModel)}&model_size_gb=${modelSizeGb}&per_gpu_memory_gb=${perGpuMemoryGb}`);
  },

  // ── Inference ──
  inferenceRegisterEndpoint(clusterId, endpointName, modelName, framework = 'custom', gpuCount = 1, gpuModel = '', quantisation = 'fp16', maxBatchSize = 1, maxInputTokens = 4096, maxOutputTokens = 1024, concurrency = 1, modelVersion = 'latest', metadata = '{}') {
    return this.post(`/api/v1/inference/endpoints?cluster_id=${clusterId}&endpoint_name=${encodeURIComponent(endpointName)}&model_name=${encodeURIComponent(modelName)}&framework=${framework}&gpu_count=${gpuCount}&gpu_model=${encodeURIComponent(gpuModel)}&quantisation=${quantisation}&max_batch_size=${maxBatchSize}&max_input_tokens=${maxInputTokens}&max_output_tokens=${maxOutputTokens}&concurrency=${concurrency}&model_version=${encodeURIComponent(modelVersion)}&metadata=${encodeURIComponent(metadata)}`);
  },
  inferenceListEndpoints(clusterId = null) { const q = clusterId ? `?cluster_id=${clusterId}` : ''; return this.get(`/api/v1/inference/endpoints${q}`); },
  inferenceGetEndpoint(endpointId) { return this.get(`/api/v1/inference/endpoints/${endpointId}`); },
  inferenceUpdateEndpoint(endpointId, updates) { const q = Object.entries(updates).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&'); return this.request('PATCH', `/api/v1/inference/endpoints/${endpointId}?${q}`); },
  inferenceDeleteEndpoint(endpointId) { return this.request('DELETE', `/api/v1/inference/endpoints/${endpointId}`); },
  inferenceProfileEndpoint(endpointId) { return this.post(`/api/v1/inference/endpoints/${endpointId}/profile`); },
  inferenceDeploymentConfig(modelName = '', modelSizeGb = 0, contextLength = 4096, targetLatencyMs = 200, expectedRps = 10, gpuBudget = '') {
    return this.post(`/api/v1/inference/deployment-config?model_name=${encodeURIComponent(modelName)}&model_size_gb=${modelSizeGb}&context_length=${contextLength}&target_latency_ms=${targetLatencyMs}&expected_requests_per_sec=${expectedRps}&gpu_budget=${encodeURIComponent(gpuBudget)}`);
  },

  // ── FinOps (extended) ──
  spotSavings(clusterId) { return this.get(`/api/v1/finops/spot-savings/${clusterId}`); },
  reservedRecs(clusterId) { return this.get(`/api/v1/finops/reserved-recs/${clusterId}`); },
  finopsBudget(clusterId, monthlyBudget = 0) { return this.get(`/api/v1/finops/budget/${clusterId}?monthly_budget=${monthlyBudget}`); },
  finopsWhatIf(scenarioName, description = '', currentMonthlyCost = 0, gpuCountChange = 0, utilizationChange = 0, provider = '', tier = '') {
    return this.post(`/api/v1/finops/what-if?scenario_name=${encodeURIComponent(scenarioName)}&description=${encodeURIComponent(description)}&current_monthly_cost=${currentMonthlyCost}&gpu_count_change=${gpuCountChange}&utilization_change=${utilizationChange}&provider=${encodeURIComponent(provider)}&tier=${encodeURIComponent(tier)}`);
  },
  finopsAllocation(clusterId) { return this.get(`/api/v1/finops/allocation/${clusterId}`); },
  finopsRecommendations(clusterId) { return this.get(`/api/v1/finops/recommendations/${clusterId}`); },

  // ── Power ──
  powerProfiles() { return this.get('/api/v1/power/profiles'); },
  powerProfile(gpuModel) { return this.get(`/api/v1/power/profile/${encodeURIComponent(gpuModel)}`); },
  powerAnalysis(clusterId) { return this.get(`/api/v1/power/analysis/${clusterId}`); },
  powerCarbon(clusterId) { return this.get(`/api/v1/power/carbon/${clusterId}`); },
  powerCapSuggestion(gpuModel = 'a100', gpuCount = 8, currentPowerWatts = 0) { return this.get(`/api/v1/power/cap-suggestion?gpu_model=${encodeURIComponent(gpuModel)}&gpu_count=${gpuCount}&current_power_watts=${currentPowerWatts}`); },
  powerRecommendations(clusterId) { return this.get(`/api/v1/power/recommendations/${clusterId}`); },

  // ── Guarded Automation ──
  gaListPolicies() { return this.get('/api/v1/guarded/policies'); },
  gaGetPolicy(policyId) { return this.get(`/api/v1/guarded/policies/${policyId}`); },
  gaCreatePolicy(payload) { return this.post('/api/v1/guarded/policies', payload); },
  gaUpdatePolicy(policyId, payload) { return this.request('PATCH', `/api/v1/guarded/policies/${policyId}`, payload); },
  gaDeletePolicy(policyId) { return this.request('DELETE', `/api/v1/guarded/policies/${policyId}`); },
  gaPreFlight(clusterId, recId, environment = '') { return this.post(`/api/v1/guarded/pre-flight/${clusterId}/${recId}?environment=${encodeURIComponent(environment)}`); },
  gaCreateApproval(payload) { return this.post('/api/v1/guarded/approvals', payload); },
  gaListApprovals(clusterId = null) { const q = clusterId ? `?cluster_id=${clusterId}` : ''; return this.get(`/api/v1/guarded/approvals${q}`); },
  gaGetApproval(approvalId) { return this.get(`/api/v1/guarded/approvals/${approvalId}`); },
  gaApprove(approvalId, approver, reason = '') { return this.post(`/api/v1/guarded/approvals/${approvalId}/approve?approver=${encodeURIComponent(approver)}&reason=${encodeURIComponent(reason)}`); },
  gaReject(approvalId, approver, reason = '') { return this.post(`/api/v1/guarded/approvals/${approvalId}/reject?approver=${encodeURIComponent(approver)}&reason=${encodeURIComponent(reason)}`); },
  gaCreateExperiment(payload) { return this.post('/api/v1/guarded/chaos-experiments', payload); },
  gaListExperiments(clusterId = null) { const q = clusterId ? `?cluster_id=${clusterId}` : ''; return this.get(`/api/v1/guarded/chaos-experiments${q}`); },
  gaGetExperiment(experimentId) { return this.get(`/api/v1/guarded/chaos-experiments/${experimentId}`); },
  gaRunExperiment(experimentId) { return this.post(`/api/v1/guarded/chaos-experiments/${experimentId}/run`); },
  gaDeleteExperiment(experimentId) { return this.request('DELETE', `/api/v1/guarded/chaos-experiments/${experimentId}`); },
  gaRecommendations(clusterId) { return this.get(`/api/v1/guarded/recommendations/${clusterId}`); },

  // ── Alerting / Observability ──
  alertCreateRule(payload) { return this.post('/api/v1/alerts/rules', payload); },
  alertListRules(clusterId = null) { const q = clusterId ? `?cluster_id=${clusterId}` : ''; return this.get(`/api/v1/alerts/rules${q}`); },
  alertGetRule(ruleId) { return this.get(`/api/v1/alerts/rules/${ruleId}`); },
  alertUpdateRule(ruleId, payload) { return this.request('PATCH', `/api/v1/alerts/rules/${ruleId}`, payload); },
  alertDeleteRule(ruleId) { return this.request('DELETE', `/api/v1/alerts/rules/${ruleId}`); },
  alertEvaluate(clusterId) { return this.post(`/api/v1/alerts/evaluate/${clusterId}`); },
  alertList(clusterId = null, status = '') { const q = clusterId ? `?cluster_id=${clusterId}${status ? `&status=${status}` : ''}` : status ? `?status=${status}` : ''; return this.get(`/api/v1/alerts${q}`); },
  alertAcknowledge(alertId, user = '') { return this.post(`/api/v1/alerts/${alertId}/acknowledge?user=${encodeURIComponent(user)}`); },
  alertResolve(alertId) { return this.post(`/api/v1/alerts/${alertId}/resolve`); },
  notificationCreateChannel(payload) { return this.post('/api/v1/notifications/channels', payload); },
  notificationListChannels() { return this.get('/api/v1/notifications/channels'); },
  notificationGetChannel(channelId) { return this.get(`/api/v1/notifications/channels/${channelId}`); },
  notificationUpdateChannel(channelId, payload) { return this.request('PATCH', `/api/v1/notifications/channels/${channelId}`, payload); },
  notificationDeleteChannel(channelId) { return this.request('DELETE', `/api/v1/notifications/channels/${channelId}`); },
  notificationTestChannel(channelId) { return this.post(`/api/v1/notifications/channels/${channelId}/test`); },
  notificationListMessages(channelId = null) { const q = channelId ? `?channel_id=${channelId}` : ''; return this.get(`/api/v1/notifications/messages${q}`); },

  // ── Multi-Tenancy ──
  tenantCreateTeam(payload) { return this.post('/api/v1/tenants/teams', payload); },
  tenantListTeams() { return this.get('/api/v1/tenants/teams'); },
  tenantGetTeam(teamId) { return this.get(`/api/v1/tenants/teams/${teamId}`); },
  tenantDeleteTeam(teamId) { return this.request('DELETE', `/api/v1/tenants/teams/${teamId}`); },
  tenantCreateProject(payload) { return this.post('/api/v1/tenants/projects', payload); },
  tenantListProjects(teamId = null) { const q = teamId ? `?team_id=${teamId}` : ''; return this.get(`/api/v1/tenants/projects${q}`); },
  tenantGetProject(projectId) { return this.get(`/api/v1/tenants/projects/${projectId}`); },
  tenantDeleteProject(projectId) { return this.request('DELETE', `/api/v1/tenants/projects/${projectId}`); },
  tenantGetQuota(projectId) { return this.get(`/api/v1/tenants/projects/${projectId}/quota`); },

  // ── Cost Anomaly ──
  costAnomaly(clusterId) { return this.get(`/api/v1/anomaly/cost/${clusterId}`); },
  costAnomalyAll() { return this.get('/api/v1/anomaly/cost'); },

  // ── Compliance ──
  complianceReport(clusterId, framework = 'soc2') { return this.get(`/api/v1/compliance/report/${clusterId}?framework=${encodeURIComponent(framework)}`); },

  // ── Dashboard & Reporting ──
  dashboardSummary(clusterId) { return this.get(`/api/v1/dashboard/${clusterId}`); },
  dashboardAll() { return this.get('/api/v1/dashboard'); },
  reportCreate(payload) { return this.post('/api/v1/reports', payload); },
  reportList() { return this.get('/api/v1/reports'); },
  reportGet(reportId) { return this.get(`/api/v1/reports/${reportId}`); },
  reportUpdate(reportId, payload) { return this.request('PATCH', `/api/v1/reports/${reportId}`, payload); },
  reportDelete(reportId) { return this.request('DELETE', `/api/v1/reports/${reportId}`); },
  reportGenerate(reportId) { return this.post(`/api/v1/reports/${reportId}/generate`); },

  // ── System ──
  systemInfo() { return this.get('/api/v1/info'); },

  // ── ML Engine ──
  mlRecommendationStatus() { return this.get('/api/v1/ml/recommendation/status'); },
  mlRecommendationTrain() { return this.post('/api/v1/ml/recommendation/train'); },
  mlRecommendationReset() { return this.post('/api/v1/ml/recommendation/reset'); },
  mlForecastStatus() { return this.get('/api/v1/ml/forecast/status'); },
  mlForecastReset() { return this.post('/api/v1/ml/forecast/reset'); },
  mlDriftStatus() { return this.get('/api/v1/ml/drift/status'); },
  mlDriftReset() { return this.post('/api/v1/ml/drift/reset'); },

  // ── Slurm ──
  slurmTelemetry(clusterId) { return this.get(`/api/v1/slurm/telemetry/${clusterId}`); },
  slurmTopology(clusterId) { return this.get(`/api/v1/slurm/topology/${clusterId}`); },
  slurmMonitorSnapshot(clusterId) { return this.get(`/api/v1/slurm/monitor/snapshot/${clusterId}`); },
  slurmMonitorStart(clusterId, jobId, config = null) { return this.post(`/api/v1/slurm/monitor/start/${clusterId}/${jobId}`, config); },
  slurmMonitorStop(clusterId, jobId) { return this.post(`/api/v1/slurm/monitor/stop/${clusterId}/${jobId}`); },
  slurmMonitorHistory(clusterId, jobId) { return this.get(`/api/v1/slurm/monitor/history/${clusterId}/${jobId}`); },

  // ── V2 Domains ──
  v2DomainsCollect(data) { return this.post('/api/v2/domains/collect', data); },
  v2DomainsQuery(domainType, limit = 10) { return this.get(`/api/v2/domains/${domainType}?limit=${limit}`); },

  // ── V2 Scheduler ──
  v2Schedule(data) { return this.post('/api/v2/schedule', data); },
  v2TrainScheduler(episodes = 100) { return this.post(`/api/v2/train-scheduler?episodes=${episodes}`); },

  // ── V2 Optimizer ──
  v2Optimize(data) { return this.post('/api/v2/optimize', data); },

  // ── V2 Governance ──
  v2RegisterModel(modelName, version, actionClass, owner = '') { return this.post(`/api/v2/governance/models/register?model_name=${encodeURIComponent(modelName)}&version=${encodeURIComponent(version)}&action_class=${actionClass}&owner=${encodeURIComponent(owner)}`); },
  v2ListModels(actionClass = null, status = null) { let q = ''; if (actionClass) q += `action_class=${actionClass}`; if (status) q += `${q ? '&' : ''}status=${status}`; return this.get(`/api/v2/governance/models${q ? `?${q}` : ''}`); },

  // ── V2 Predictor ──
  v2TrainPredictor(telemetryData, labels) { return this.post('/api/v2/train-predictor', { telemetry_data: telemetryData, labels }); },

  // ── V2 Healing ──
  v2HealingCheck(telemetry) { return this.post('/api/v2/healing/check', telemetry); },
  v2HealingExecute(telemetry, nodeId = 'unknown') { return this.post(`/api/v2/healing/execute?node_id=${encodeURIComponent(nodeId)}`, telemetry); },
  v2HealingHistory(limit = 50) { return this.get(`/api/v2/healing/history?limit=${limit}`); },

  // ── V2 Version ──
  v2Version() { return this.get('/api/v2/version'); },

  // ── V2 Intelligence ──
  v2CrossClusterOptimize(requirements, objective = 'balanced') { return this.post(`/api/v2/intelligence/cross-cluster/optimize?objective=${objective}`, requirements); },
  v2OrchestratePlan(clusterId) { return this.post(`/api/v2/intelligence/orchestrate/plan?cluster_id=${clusterId}`); },
  v2OrchestrateRunCycle() { return this.post('/api/v2/intelligence/orchestrate/run-cycle'); },
  v2ScanIdleGpus(clusterId) { return clusterId ? this.get(`/api/v2/intelligence/idle-gpus/scan?cluster_id=${clusterId}`) : this.get('/api/v2/intelligence/idle-gpus/scan'); },
  v2ReclaimIdleGpus(clusterId, dryRun = true) { return this.post(`/api/v2/intelligence/idle-gpus/reclaim?cluster_id=${clusterId}&dry_run=${dryRun}`); },
  v2AdaptiveSchedule(requirements, strategy = 'balanced') { return this.post(`/api/v2/intelligence/adaptive/schedule?strategy=${strategy}`, requirements); },
  v2IntelligenceHealth() { return this.get('/api/v2/intelligence/health'); },
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
    const host = window.location.host;

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
