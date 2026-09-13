"use client";

import { useEffect, useState } from "react";
import { GRADE_NAMES } from "@/lib/types";

// The model's decision geometry drawn to scale: four cut points divide the score axis into
// grades, and the referral threshold is a separate line on the same axis. Reading the grade
// (which segment) and the referral (which side of the line) off one ruler is what makes a
// "Mild, refer" result legible as two decisions rather than a contradiction.
//
// Deliberately NOT a gauge or a meter: there is no fill and no sense of progress, and the
// segment widths are proportional to the real cut points. Equal-width segments would look
// tidier and would put the threshold in the wrong place.
//
// Rows, top to bottom: threshold label / score label / the bar / segment names. The bar
// itself carries no text, so neither vertical line ever crosses a word.

const DOMAIN_MIN = -0.5;
const DOMAIN_MAX = 4.5;
const SEG_BG = ["bg-sev-0", "bg-sev-1", "bg-sev-2", "bg-sev-3", "bg-sev-4"];

const clampPct = (x: number) =>
  ((Math.min(Math.max(x, DOMAIN_MIN), DOMAIN_MAX) - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * 100;

// toFixed turns -0.0002 into "-0.00"; a sign on a value that rounds to zero says nothing.
export const fmt = (x: number, d = 2) => {
  const s = x.toFixed(d);
  return Number(s) === 0 ? (0).toFixed(d) : s;
};

// Keep a label inside the ruler near either edge instead of centring it off the page.
const anchor = (p: number) => (p > 78 ? "-100%" : p < 22 ? "0%" : "-50%");

export function ScoreRuler({
  score,
  cuts,
  threshold,
}: {
  score: number;
  cuts: [number, number, number, number];
  threshold: number;
}) {
  const edges = [DOMAIN_MIN, ...cuts, DOMAIN_MAX];
  const widths = GRADE_NAMES.map((_, i) => clampPct(edges[i + 1]) - clampPct(edges[i]));

  // The one orchestrated motion on the page: the score marker travels from the start of the
  // axis to where this image landed — it shows the thing that just changed. Reduced-motion
  // users get it in place (globals.css).
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const s = clampPct(score);
  const t = clampPct(threshold);
  const off = score < DOMAIN_MIN ? " (below the scale)" : score > DOMAIN_MAX ? " (above the scale)" : "";

  return (
    <figure className="mb-5">
      <div className="relative" aria-hidden="true">
        {/* row 1: threshold label */}
        <div className="relative h-6">
          <span
            className="absolute top-0 whitespace-nowrap text-sm text-ink-2"
            style={{ left: `${t}%`, transform: `translateX(${anchor(t)})` }}
          >
            Referral threshold {fmt(threshold, 3)}
          </span>
        </div>

        {/* row 2: score label, on its own row so it never collides with row 1 */}
        <div className="relative h-8">
          <span
            className="absolute top-1 z-10 whitespace-nowrap bg-ink px-1.5 py-0.5 text-sm font-bold text-paper transition-[left] duration-700 ease-out"
            style={{ left: `${shown ? s : 0}%`, transform: `translateX(${anchor(s)})` }}
          >
            Score {fmt(score)}
            {off}
          </span>
        </div>

        {/* row 3: the bar, no text inside it */}
        <div className="relative">
          <div className="flex h-10 w-full">
            {GRADE_NAMES.map((name, i) => (
              <div
                key={name}
                className={`${SEG_BG[i]} h-full border-r-2 border-paper last:border-r-0`}
                style={{ width: `${widths[i]}%` }}
              />
            ))}
          </div>

          {/* threshold: dashed, from just under its label through the bar. Starting it at
              row 2 rather than row 1 keeps the line out of its own label. */}
          <div
            className="absolute bottom-0 border-l-2 border-dashed border-ink"
            style={{ left: `${t}%`, top: "-2rem" }}
          />
          {/* score: solid, from row 2 through the bar */}
          <div
            className="absolute -top-2 bottom-0 w-1 -translate-x-1/2 bg-ink transition-[left] duration-700 ease-out"
            style={{ left: `${shown ? s : 0}%` }}
          />
        </div>

        {/* row 4: segment names under their own segments (wide screens) */}
        <div className="mt-1.5 hidden w-full md:flex">
          {GRADE_NAMES.map((name, i) => (
            <div
              key={name}
              className="overflow-hidden px-0.5 text-center text-[0.8rem] leading-tight text-ink-2"
              style={{ width: `${widths[i]}%` }}
            >
              {name}
            </div>
          ))}
        </div>

        {/* narrow screens: a key, because the Moderate segment is too narrow to label */}
        <ol className="mt-2 grid grid-cols-3 gap-x-3 gap-y-1 text-[0.8rem] leading-tight text-ink-2 sm:grid-cols-5 md:hidden">
          {GRADE_NAMES.map((name, i) => (
            <li key={name} className="flex items-center gap-1.5">
              <span className={`${SEG_BG[i]} inline-block h-3 w-3 shrink-0 border border-ink-2/40`} />
              {name}
            </li>
          ))}
        </ol>
      </div>

      <figcaption className="sr-only">
        Score {fmt(score)}{off}. Grade cut points at {cuts.map((c) => fmt(c, 3)).join(", ")}.
        Referral threshold at {fmt(threshold, 3)}.
      </figcaption>
    </figure>
  );
}
