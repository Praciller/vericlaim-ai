/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  ...(process.env.VERCEL === "1" ? {} : { output: "standalone" }),
};
export default nextConfig;
