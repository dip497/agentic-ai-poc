/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/ag-ui/:path*',
        destination: 'http://localhost:8000/:path*',
      },
      {
        source: '/api/agent-studio/:path*',
        destination: 'http://localhost:8000/api/agent-studio/:path*',
      },
      {
        source: '/api/system/:path*',
        destination: 'http://localhost:8000/api/system/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
