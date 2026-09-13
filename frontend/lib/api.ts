import type { Meta, PredictResponse } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

// Every call resolves to one of these. Nothing throws past this module, so each failure
// has its own screen instead of collapsing into a generic "something went wrong".
export type ApiResult<T> =
  | { kind: "ok"; data: T }
  | { kind: "http"; status: number; detail: string }
  | { kind: "network"; message: string }
  | { kind: "malformed"; message: string };

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStr = (v: unknown): v is string => typeof v === "string";
const isStrArr = (v: unknown): v is string[] => Array.isArray(v) && v.every(isStr);

const GUARD_STATUSES = new Set([
  "ok",
  "framing_below_training_range",
  "framing_above_training_range",
  "no_retina_detected",
]);

function checkGuard(g: unknown, graded: boolean): string | null {
  if (!isObj(g)) return "guard is missing";
  if (!isStr(g.status) || !GUARD_STATUSES.has(g.status)) return "guard.status is unknown";
  if (!isStr(g.message)) return "guard.message is missing";
  if (!isNum(g.low) || !isNum(g.high)) return "guard bounds are missing";
  if (graded) {
    if (g.status === "no_retina_detected") return "a graded response carried no_retina_detected";
    if (!isNum(g.coverage)) return "guard.coverage is missing";
  } else {
    if (g.status !== "no_retina_detected") return "a declined response carried a framing status";
    if (g.coverage !== null) return "a declined response carried a coverage value";
  }
  return null;
}

export function validateMeta(v: unknown): string | null {
  if (!isObj(v)) return "not an object";
  if (!isStr(v.run_id) || !isStr(v.provenance)) return "run_id or provenance is missing";
  if (!isNum(v.reported_val_qwk) || !isNum(v.reported_val_sens_at_spec95))
    return "reported metrics are missing";
  if (!isStrArr(v.disclaimer) || v.disclaimer.length !== 4)
    return "the disclaimer does not have four statements";
  const c = v.coverage_calibration;
  if (!isObj(c) || !isNum(c.low) || !isNum(c.high) || !isNum(c.n_images))
    return "coverage_calibration is missing";
  return null;
}

export function validatePredict(v: unknown): string | null {
  if (!isObj(v) || typeof v.graded !== "boolean") return "graded is missing";
  if (!isStr(v.run_id)) return "run_id is missing";
  if (!isStrArr(v.disclaimer) || v.disclaimer.length !== 4)
    return "the disclaimer does not have four statements";
  const guardProblem = checkGuard(v.guard, v.graded);
  if (guardProblem) return guardProblem;
  if (!v.graded) {
    if ("grade" in v) return "a declined response carried a grade";
    return null;
  }
  if (!Number.isInteger(v.grade) || (v.grade as number) < 0 || (v.grade as number) > 4)
    return "grade is out of range";
  if (!isStr(v.grade_name)) return "grade_name is missing";
  if (!isNum(v.score) || !isNum(v.referral_threshold)) return "score or threshold is missing";
  if (typeof v.referable !== "boolean") return "referable is missing";
  if (!Array.isArray(v.cuts) || v.cuts.length !== 4 || !v.cuts.every(isNum))
    return "cuts must be four numbers";
  return null;
}

async function readDetail(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (isObj(body) && isStr(body.detail)) return body.detail;
  } catch {
    // fall through to the status text
  }
  return res.statusText || `HTTP ${res.status}`;
}

async function call<T>(
  path: string,
  init: RequestInit,
  validate: (v: unknown) => string | null,
): Promise<ApiResult<T>> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, cache: "no-store" });
  } catch {
    return {
      kind: "network",
      message: `The grading service did not respond at ${API_BASE}.`,
    };
  }
  if (!res.ok) return { kind: "http", status: res.status, detail: await readDetail(res) };

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return { kind: "malformed", message: "The response was not valid JSON." };
  }
  const problem = validate(body);
  if (problem) return { kind: "malformed", message: problem };
  return { kind: "ok", data: body as T };
}

export function getMeta(): Promise<ApiResult<Meta>> {
  return call<Meta>("/meta", { method: "GET" }, validateMeta);
}

export function predict(file: File, signal?: AbortSignal): Promise<ApiResult<PredictResponse>> {
  const form = new FormData();
  form.append("file", file);
  return call<PredictResponse>(
    "/predict?explain=true",
    { method: "POST", body: form, signal },
    validatePredict,
  );
}
