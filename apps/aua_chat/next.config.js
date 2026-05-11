/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    AUA_ROUTER_URL: process.env.AUA_ROUTER_URL || 'http://localhost:8000',
    NEXTAUTH_URL: process.env.NEXTAUTH_URL || 'http://localhost:3001',
    NEXTAUTH_SECRET: process.env.NEXTAUTH_SECRET || 'aua-dev-secret-change-in-production',
  },
}
module.exports = nextConfig
