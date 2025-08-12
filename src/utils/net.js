import got from 'got';

export const http = got.extend({
  timeout: { request: 15000 },
  retry: { limit: 2, methods: ['GET', 'HEAD'], statusCodes: [408, 413, 429, 500, 502, 503, 504] },
  headers: {
    'user-agent': 'tmjs/1.0 (+https://github.com/nixaut-codelabs/teachable-machine.js)'
  }
});
