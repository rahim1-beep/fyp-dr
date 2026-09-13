"use client";

import { useState } from "react";

type View = "graded" | "gradcam";

// The image shown as "graded" is always `preprocessed_png` — the 224 × 224 crop the model
// received. The user's own file is shown smaller and labelled for what it is.
export function ImageViewer({
  preprocessed,
  overlay,
  originalUrl,
  fileName,
}: {
  preprocessed?: string;
  overlay?: string;
  originalUrl?: string;
  fileName: string;
}) {
  const [view, setView] = useState<View>("graded");
  const [strength, setStrength] = useState(100);
  const canCam = Boolean(preprocessed && overlay);

  return (
    <figure>
      {canCam && (
        <div role="group" aria-label="Image view" className="mb-3 inline-flex border border-ink">
          {(
            [
              ["graded", "Graded image"],
              ["gradcam", "Grad-CAM"],
            ] as const
          ).map(([v, label]) => (
            <button
              key={v}
              type="button"
              aria-pressed={view === v}
              onClick={() => setView(v)}
              className={`px-3 py-1.5 text-[0.95rem] ${
                view === v ? "bg-ink text-paper" : "bg-transparent text-ink hover:bg-sheet"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      <div className="bg-viewer p-4 sm:p-6">
        {preprocessed ? (
          <div className="relative mx-auto aspect-square w-full max-w-[448px]">
            {/* eslint-disable-next-line @next/next/no-img-element -- data URIs from the API; next/image adds nothing here */}
            <img
              src={preprocessed}
              alt={`The ${fileName} crop the model graded, 224 by 224 pixels`}
              className="absolute inset-0 h-full w-full"
            />
            {canCam && view === "gradcam" && (
              /* eslint-disable-next-line @next/next/no-img-element -- data URI from the API */
              <img
                src={overlay}
                alt="Grad-CAM heatmap over the graded crop"
                className="absolute inset-0 h-full w-full"
                style={{ opacity: strength / 100 }}
              />
            )}
          </div>
        ) : (
          <p className="py-16 text-center text-paper">No graded image was returned.</p>
        )}
      </div>

      {canCam && view === "gradcam" && (
        <label className="mt-3 flex max-w-sm items-center gap-3 text-[0.95rem]">
          <span className="shrink-0">Heatmap strength</span>
          <input
            type="range"
            min={0}
            max={100}
            value={strength}
            onChange={(e) => setStrength(Number(e.target.value))}
            className="w-full accent-[var(--color-sev-4)]"
          />
        </label>
      )}

      <figcaption className="mt-3 max-w-[60ch] text-[0.9rem] leading-snug text-ink-2">
        {view === "gradcam" && canCam
          ? "Grad-CAM from the final convolutional block. It is computed on a 7 × 7 grid and enlarged, so it shows which broad regions moved the score, not where individual lesions are."
          : "This is the 224 × 224 crop the model received after the same preprocessing its training images went through — not the file as uploaded."}
      </figcaption>

      {originalUrl && (
        <div className="mt-5 flex items-start gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element -- local object URL of the user's file */}
          <img
            src={originalUrl}
            alt={`Uploaded file ${fileName}`}
            className="h-20 w-20 shrink-0 bg-viewer object-contain"
          />
          <p className="text-[0.9rem] leading-snug text-ink-2">
            Your uploaded file, {fileName}, for comparison. The model did not see this version.
          </p>
        </div>
      )}
    </figure>
  );
}
