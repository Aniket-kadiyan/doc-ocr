import type { NextConfig } from "next";
import path from "path";
import { fileURLToPath } from "url";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const canvasStub = path.join(rootDir, "src/lib/stubs/canvas.js");

const nextConfig: NextConfig = {
  outputFileTracingRoot: rootDir,
  turbopack: {
    resolveAlias: {
      canvas: canvasStub,
    },
    resolveExtensions: [".tsx", ".ts", ".jsx", ".js", ".mjs", ".json"],
  },
  webpack: (config, { isServer }) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      canvas: canvasStub,
    };
    if (!isServer) {
      config.resolve.mainFields = ["browser", "module", "main"];
    }
    return config;
  },
};

export default nextConfig;
