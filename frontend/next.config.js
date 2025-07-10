/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/ag-ui/:path*',
        destination: 'http://localhost:8081/:path*',
      },
      {
        source: '/api/agent-studio/:path*',
        destination: 'http://localhost:8081/api/agent-studio/:path*',
      },
      {
        source: '/api/system/:path*',
        destination: 'http://localhost:8081/api/system/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
