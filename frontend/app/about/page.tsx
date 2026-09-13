"use client";

import { API_BASE } from "@/lib/api";
import { useMeta } from "@/lib/useMeta";
import { Disclaimer } from "@/components/Disclaimer";
import { Provenance } from "@/components/Provenance";

export default function AboutPage() {
  const [meta, reload] = useMeta();

  if (meta.kind === "loading") {
    return <p className="text-ink-2" role="status">Connecting to the grading service…</p>;
  }
  if (meta.kind !== "ok") {
    return (
      <div role="alert">
        <h1 className="text-3xl font-bold">The grading service is not reachable</h1>
        <p className="mt-3 max-w-[62ch] text-lg">
          This page reads the model&apos;s figures from {API_BASE} rather than printing its own
          copies, so it cannot be shown until the service responds.
        </p>
        <button type="button" onClick={reload} className="mt-6 bg-ink px-5 py-2.5 font-bold text-paper">
          Try again
        </button>
      </div>
    );
  }

  const m = meta.data;

  return (
    <article className="max-w-[70ch]">
      <h1 className="text-4xl font-bold leading-tight tracking-tight">About this model</h1>
      <p className="mt-4 text-lg leading-relaxed">
        This is a final-year research prototype. It estimates the severity of diabetic
        retinopathy on the standard five-step scale from a single fundus photograph, and it is
        built to report the limits of that estimate as plainly as the estimate itself.
      </p>

      <div className="mt-8">
        <Disclaimer items={m.disclaimer} id="about-limits" />
      </div>

      <h2 className="mt-10 text-2xl font-bold">What it was measured to do</h2>
      <p className="mt-3 leading-relaxed">
        On held-out validation images, averaged over three training runs, agreement with the
        reference grades is a quadratic weighted kappa of {m.reported_val_qwk.toFixed(4)}. When
        tuned so that 95% of images not needing referral are left unflagged, it flags{" "}
        {(m.reported_val_sens_at_spec95 * 100).toFixed(1)}% of the images that do need referral —
        so roughly one referable case in four is missed.
      </p>

      <h2 className="mt-10 text-2xl font-bold">How a result is produced</h2>
      <p className="mt-3 leading-relaxed">
        The photograph is cropped to the retina and normalised with the same preprocessing used
        on the training images, then reduced to {m.image_size} × {m.image_size} pixels. The model
        ({m.arch}) outputs one continuous score. Two separate decisions are read off that score:
        a grade, from four cut points fitted to agree with human graders, and a referral flag,
        from one threshold fitted for screening. Because they were fitted separately, a result
        can be graded Mild and still be flagged for referral. The page shows both as computed.
      </p>

      <h2 className="mt-10 text-2xl font-bold">The framing check</h2>
      <p className="mt-3 leading-relaxed">
        On an external dataset the model&apos;s score was found to depend partly on how much of the
        frame the retina fills — a property of the camera and clinic, not the eye — and more so
        than on its own training data. Under criteria fixed before that test, the model is not
        considered to generalise beyond its training population.
      </p>
      <p className="mt-3 leading-relaxed">
        Each upload is therefore checked against the framing range measured on{" "}
        {m.coverage_calibration.n_images.toLocaleString("en-GB")} training images, and a result
        outside that range carries a warning. The check judges one image at a time. It cannot
        tell that a whole clinic&apos;s photographs are framed differently from the training set,
        which is the situation in which the dependence was found.
      </p>

      <h2 className="mt-10 text-2xl font-bold">The heatmap</h2>
      <p className="mt-3 leading-relaxed">
        Grad-CAM is computed on a coarse 7 × 7 grid and enlarged. It indicates which broad regions
        moved the score. It does not locate individual lesions, and no claim is made that it does.
      </p>

      <Provenance meta={m} />
    </article>
  );
}
