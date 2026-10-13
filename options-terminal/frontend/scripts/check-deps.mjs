// building_secure_front_ends.md §4.2: runtime dependencies must stay at the approved baseline.
// Exceptions approved under §6 are added to EXCEPTIONS and documented in README.md.
import { readFileSync } from "node:fs";

const BASELINE = ["react", "react-dom"];
const DEV_BASELINE = [
  "@types/react",
  "@types/react-dom",
  "@vitejs/plugin-react",
  "typescript",
  "vite",
];
const EXCEPTIONS = [];

const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
const allowed = new Set([...BASELINE, ...DEV_BASELINE, ...EXCEPTIONS]);
const problems = [];

for (const field of ["dependencies", "devDependencies"]) {
  for (const [name, version] of Object.entries(pkg[field] ?? {})) {
    if (!allowed.has(name)) problems.push(`${field}: ${name} is not approved`);
    if (!/^\d+\.\d+\.\d+$/.test(version)) problems.push(`${field}: ${name}@${version} is not exact`);
  }
}
for (const name of Object.keys(pkg.dependencies ?? {})) {
  if (!BASELINE.includes(name) && !EXCEPTIONS.includes(name)) {
    problems.push(`dependencies: ${name} exceeds the runtime baseline`);
  }
}

if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("dependencies ok");
