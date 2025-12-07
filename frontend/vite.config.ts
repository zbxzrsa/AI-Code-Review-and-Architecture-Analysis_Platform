import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { mockApiPlugin } from "./src/mocks/mockServer";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const useMockApi = env.VITE_USE_MOCK_API === "true" || mode === "development";

  return {
    plugins: [
      react(),
      // Use mock API in development when backend is not running
      useMockApi && mockApiPlugin(),
    ].filter(Boolean),
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      port: 5173,
      strictPort: false, // Allow fallback to next available port
      proxy: useMockApi
        ? undefined
        : {
            // CSRF token - route to auth server
            "/api/csrf-token": {
              target: "http://localhost:8001",
              changeOrigin: true,
              rewrite: (path) => path.replace(/^\/api/, ""),
            },
            // Auth service
            "/api/auth": {
              target: "http://localhost:8001",
              changeOrigin: true,
              rewrite: (path) => path.replace(/^\/api/, ""),
            },
            // Other API services
            "/api": {
              target: "http://localhost:8000",
              changeOrigin: true,
            },
            "/ws": {
              target: "ws://localhost:8000",
              ws: true,
            },
          },
    },
    build: {
      outDir: "dist",
      sourcemap: mode !== "production", // Disable sourcemaps in production
      minify: "terser",
      terserOptions: {
        compress: {
          drop_console: mode === "production", // Remove console.log in production
          drop_debugger: true,
        },
      },
      rollupOptions: {
        output: {
          // Optimize chunk splitting for better caching
          manualChunks: {
            // Vendor chunks - separate large dependencies
            "vendor-react": ["react", "react-dom", "react-router-dom"],
            "vendor-antd": ["antd", "@ant-design/icons"],
            "vendor-charts": ["echarts", "recharts"],
            "vendor-editor": ["monaco-editor", "@monaco-editor/react"],
            "vendor-utils": ["lodash-es", "date-fns", "axios"],
          },
          // Optimize asset naming for caching
          chunkFileNames: "assets/js/[name]-[hash].js",
          entryFileNames: "assets/js/[name]-[hash].js",
          assetFileNames: "assets/[ext]/[name]-[hash].[ext]",
        },
      },
      // Increase chunk size warning limit
      chunkSizeWarningLimit: 1000,
      // Enable CSS code splitting
      cssCodeSplit: true,
      // Target modern browsers for smaller bundles
      target: "es2020",
    },
    test: {
      globals: true,
      environment: "jsdom",
      setupFiles: "./src/test/setup.ts",
      css: true,
      include: ["src/**/*.{test,spec}.{js,jsx,ts,tsx}"],
      coverage: {
        provider: "v8",
        reporter: ["text", "json", "html"],
        exclude: ["node_modules/", "src/test/"],
      },
    },
  };
});
