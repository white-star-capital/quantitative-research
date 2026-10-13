import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API base URL comes from VITE_API_BASE_URL at build time (FRONTEND.md F-1).
// Output keeps Vite's default `assets/index-[hash].js` entry naming; the reload-on-deploy
// check relies on it (front_end_design.md §9.7).
export default defineConfig({
  plugins: [react()],
  build: {
    sourcemap: false,
  },
});
