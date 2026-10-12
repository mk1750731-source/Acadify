import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'https://acadify-backend-production-d80e.up.railway.app',
  withCredentials: true,
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  },
  maxBodyLength: Infinity,
  maxContentLength: Infinity,
});

// In-memory cache: { url: { data, ts } }
const cache = {};
const CACHE_TTL = 5 * 60 * 1000;

const originalGet = api.get.bind(api);

api.get = (url, config = {}) => {
  const { noCache, onCached, ...restConfig } = config;
  const now = Date.now();
  const hit = cache[url];

  if (!noCache && hit) {
    if (now - hit.ts < CACHE_TTL) return Promise.resolve(hit.data);
    originalGet(url, restConfig).then(res => { cache[url] = { data: res, ts: Date.now() }; onCached?.(res); });
    return Promise.resolve(hit.data);
  }

  return originalGet(url, restConfig).then(res => {
    if (!noCache) cache[url] = { data: res, ts: Date.now() };
    return res;
  });
};

api.clearCache = (urlPrefix) => {
  Object.keys(cache).forEach(k => {
    if (!urlPrefix || k.startsWith(urlPrefix)) delete cache[k];
  });
};

api.interceptors.response.use(
  (response) => {
    if (response.data?.redirect?.includes('/login')) {
      window.location.href = '/login';
    }
    return response;
  },
  (error) => {
    if (error.response?.status === 401 && !window.location.pathname.includes('/login')) {
      api.clearCache(); // clear stale cache on auth failure
      localStorage.removeItem('edu_user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
