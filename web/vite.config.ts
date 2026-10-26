import { defineConfig } from "vite";

export default defineConfig({
  build: { outDir: "dist", emptyOutDir: true },
  server: { proxy: { "/ws": { target: "ws://localhost:8000", ws: true }, "/runs": "http://localhost:8000" } },
});
