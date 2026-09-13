// The API contract, typed from docs/api-contract.md and fixtures/expected/*.json.
// The /predict response is a discriminated union on `graded`: the declined variant has no
// `grade` key at all, so nothing can read a grade without narrowing first.

export type GuardStatus =
  | "ok"
  | "framing_below_training_range"
  | "framing_above_training_range"
  | "no_retina_detected";

export interface GuardInRange {
  status: Exclude<GuardStatus, "no_retina_detected">;
  coverage: number;
  low: number;
  high: number;
  message: string;
}

export interface GuardNoRetina {
  status: "no_retina_detected";
  coverage: null;
  low: number;
  high: number;
  message: string;
}

export interface CoverageCalibration {
  split: string;
  n_images: number;
  low: number;
  high: number;
}

export interface Meta {
  run_id: string;
  seed: number;
  arch: string;
  head: string;
  image_size: number;
  provenance: string;
  reported_val_qwk: number;
  reported_val_sens_at_spec95: number;
  disclaimer: string[];
  coverage_calibration: CoverageCalibration;
}

export type GradeIndex = 0 | 1 | 2 | 3 | 4;

export interface PredictGraded {
  graded: true;
  grade: GradeIndex;
  grade_name: string;
  score: number;
  referable: boolean;
  referral_threshold: number;
  cuts: [number, number, number, number];
  guard: GuardInRange;
  run_id: string;
  disclaimer: string[];
  overlay_png?: string;
  preprocessed_png?: string;
}

export interface PredictDeclined {
  graded: false;
  guard: GuardNoRetina;
  run_id: string;
  disclaimer: string[];
}

export type PredictResponse = PredictGraded | PredictDeclined;

// The ladder of names by index, used only to label ruler segments. The grade of a result
// always comes from `grade_name` in the response, never from this list.
export const GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"] as const;
