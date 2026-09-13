import type { PredictGraded } from "@/lib/types";
import { ScoreRuler } from "./ScoreRuler";

const f3 = (x: number) => x.toFixed(3);

function gradeBasis(score: number, cuts: PredictGraded["cuts"], grade: number): string {
  if (grade === 0) return `Score is below the first cut point, ${f3(cuts[0])}.`;
  if (grade === 4) return `Score is above the last cut point, ${f3(cuts[3])}.`;
  return `Score is between the cut points ${f3(cuts[grade - 1])} and ${f3(cuts[grade])}.`;
}

// Two answers, same size, same weight, side by side. Neither is derived from the other and
// neither is styled as the "real" one. Grade 0 gets exactly the treatment grade 4 gets.
export function Readout({ r }: { r: PredictGraded }) {
  const gradeSaysRefer = r.grade >= 2;
  const differ = gradeSaysRefer !== r.referable;

  return (
    <section aria-labelledby="readout" className="border-t-4 border-ink pt-5">
      <h2 id="readout" className="sr-only">
        Estimate
      </h2>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-1">
        <div>
          <dt className="text-ink-2">Estimated grade</dt>
          <dd className="mt-1 text-3xl font-bold leading-tight sm:text-4xl">{r.grade_name}</dd>
          <dd className="mt-2 max-w-[34ch] text-[0.95rem] leading-snug text-ink-2">
            Grade {r.grade} on the 0 to 4 scale. {gradeBasis(r.score, r.cuts, r.grade)}
          </dd>
        </div>
        <div className="border-l border-rule pl-6">
          <dt className="text-ink-2">Referral</dt>
          <dd className="mt-1 text-3xl font-bold leading-tight sm:text-4xl">
            {r.referable ? "Refer" : "Not flagged"}
          </dd>
          <dd className="mt-2 max-w-[34ch] text-[0.95rem] leading-snug text-ink-2">
            Score is {r.referable ? "at or above" : "below"} the referral threshold,{" "}
            {f3(r.referral_threshold)}.
          </dd>
        </div>
      </dl>

      <div className="mt-6">
        <ScoreRuler score={r.score} cuts={r.cuts} threshold={r.referral_threshold} />
      </div>

      <p className="max-w-[68ch] text-[0.95rem] leading-snug">
        These are two separate decisions read off the same score. The grade uses four cut
        points fitted to agree with human graders; referral uses one threshold fitted separately
        for screening.
        {differ && (
          <>
            {" "}
            <strong>
              For this image they differ, and both are shown as computed: the score is past the
              referral threshold but not past the cut point for Moderate.
            </strong>
          </>
        )}
      </p>
    </section>
  );
}
