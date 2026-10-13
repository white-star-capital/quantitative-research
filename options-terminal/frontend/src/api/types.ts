// Mirrors contracts/schemas/envelope/v1.json and API.md §3–4. Kept in sync by validate.ts tests
// against contracts fixtures (CONTRACTS C-1); no generator, per building_secure_front_ends.md §2.

export const PAYLOAD_KINDS = [
  "scalar_set",
  "curve",
  "surface",
  "timeseries",
  "table",
  "composite",
] as const;
export type PayloadKind = (typeof PAYLOAD_KINDS)[number];

export type Scale = "decimal" | "percent" | "bps" | "absolute";

export interface Unit {
  quantity: string;
  scale: Scale;
  convention?: string;
}

export interface Message {
  code: string;
  message: string;
}

export type Scope = Record<string, string>;

interface EnvelopeBase {
  envelope_version: "1";
  run_id: string;
  slot_id: string;
  implementation_id: string;
  implementation_version: string;
  scope: Scope;
  as_of: string;
  started_at: string;
  finished_at: string;
  inputs: { sources: { name: string; snapshot_at: string }[]; hash: string };
  code: { git_sha: string };
  warnings: Message[];
  payload_kind: PayloadKind;
  payload_version: string;
}

export interface SuccessEnvelope extends EnvelopeBase {
  status: "success";
  payload: Record<string, unknown>;
}

export interface FailedEnvelope extends EnvelopeBase {
  status: "failed";
  payload: null;
  error: Message;
}

export type Envelope = SuccessEnvelope | FailedEnvelope;

export interface RunMeta {
  api_version: string;
  role: "production" | "shadow";
  age_seconds: number;
  max_age_seconds: number;
  stale: boolean;
  served_at: string;
}

export interface LatestResponse {
  data: Envelope;
  meta: RunMeta;
}

// GET /v1/slots item (API.md §3). Field names follow the slots table; confirm against
// /openapi.json once the API exists.
export interface SlotSummary {
  slot_id: string;
  title: string;
  description: string;
  payload_kind: PayloadKind;
  payload_version: string;
  scope: { dimensions: string[]; allowed?: Record<string, string[]> };
  max_age_seconds: number;
  production_impl: string | null;
  production_version: string | null;
  display?: { default_view?: string; order?: number };
}

export interface Problem {
  type: string;
  title: string;
  status: number;
  detail?: string;
  request_id?: string;
}
