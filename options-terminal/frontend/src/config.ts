// FRONTEND.md F-1: the API base URL is build-time config, never a hard-coded host.
export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
