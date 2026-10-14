import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app/App.tsx";
import "./views/bs_pricer/index.tsx"; // registers the bs_pricer bespoke view
import "./views/parity_monitor/index.tsx"; // registers the parity_monitor bespoke view
import "./styles/tokens.css";
import "./styles/app.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root missing from index.html");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
