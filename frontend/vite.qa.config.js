// Isolated browser QA; never proxy test mutations to the normal backend.
import { fileURLToPath } from 'node:url';
import base from './vite.config.js';

export default {
  ...base,
  root: fileURLToPath(new URL('.', import.meta.url)),
  server: {
    host: '127.0.0.1', port: 3001, strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8001', changeOrigin: true } },
  },
};
