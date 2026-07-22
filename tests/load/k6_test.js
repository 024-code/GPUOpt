import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || '';

const params = {
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  },
};

const failureRate = new Rate('failed_requests');
const latencyTrend = new Trend('request_latency');

const CLUSTER_PAYLOAD = JSON.stringify({
  name: 'load-test-cluster',
  environment: 'loadtest',
  connector_type: 'mock',
  options: { snapshot_path: 'sandbox/mock-clusters/local-kind.json' },
});

export const options = {
  stages: [
    { duration: '30s', target: 10 },
    { duration: '1m', target: 50 },
    { duration: '30s', target: 100 },
    { duration: '1m', target: 100 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    failed_requests: ['rate<0.05'],
    http_req_duration: ['p(95)<2000'],
  },
};

export function setup() {
  const res = http.post(`${BASE_URL}/api/v1/clusters`, CLUSTER_PAYLOAD, params);
  const cluster = res.json();
  const cid = cluster.id;

  http.post(`${BASE_URL}/api/v1/clusters/${cid}/state`, null, params);
  http.post(`${BASE_URL}/api/v1/clusters/${cid}/state`, null, params);
  http.post(`${BASE_URL}/api/v1/clusters/${cid}/analyze`, null, params);
  http.post(`${BASE_URL}/api/v1/clusters/${cid}/recommendations`, null, params);

  return { clusterId: cid };
}

export default function (data) {
  const cid = data.clusterId;

  const endpoints = [
    { url: `${BASE_URL}/api/v1/clusters`, method: 'GET' },
    { url: `${BASE_URL}/api/v1/clusters/${cid}`, method: 'GET' },
    { url: `${BASE_URL}/api/v1/clusters/${cid}/state`, method: 'GET' },
    { url: `${BASE_URL}/api/v1/clusters/${cid}/checks/latest`, method: 'GET' },
    { url: `${BASE_URL}/api/v1/clusters/${cid}/recommendations/latest`, method: 'GET' },
    { url: `${BASE_URL}/api/v1/clusters/${cid}/analysis/latest`, method: 'GET' },
    { url: `${BASE_URL}/health/live`, method: 'GET' },
    { url: `${BASE_URL}/health/ready`, method: 'GET' },
    { url: `${BASE_URL}/api/v1/dashboard/${cid}`, method: 'GET' },
    { url: `${BASE_URL}/api/v1/alerts?cluster_id=${cid}`, method: 'GET' },
  ];

  const ep = endpoints[Math.floor(Math.random() * endpoints.length)];
  const start = Date.now();
  const res = http.request(ep.method, ep.url, null, params);
  const duration = Date.now() - start;

  latencyTrend.add(duration);
  failureRate.add(res.status >= 400);

  check(res, {
    'status is 2xx': (r) => r.status >= 200 && r.status < 300,
    'response time < 2s': (r) => duration < 2000,
  });

  sleep(Math.random() * 0.5);
}

export function teardown(data) {
  http.del(`${BASE_URL}/api/v1/clusters/${data.clusterId}`, null, params);
}
