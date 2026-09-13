"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, predict, type ApiResult } from "@/lib/api";
import type { Meta, PredictResponse } from "@/lib/types";
import { useMeta } from "@/lib/useMeta";
import { Disclaimer } from "@/components/Disclaimer";
import { GuardNotice } from "@/components/GuardNotice";
import { ImageViewer } from "@/components/ImageViewer";
import { Provenance } from "@/components/Provenance";
import { Readout } from "@/components/Readout";

type Flow =
  | { kind: "idle" }
  | { kind: "grading"; file: File; url: string }
  | { kind: "done"; file: File; url: string; result: ApiResult<PredictResponse> };

export default function GradePage() {
  const [meta, reloadMeta] = useMeta();
  const [flow, setFlow] = useState<Flow>({ kind: "idle" });
  const urlRef = useRef<string | null>(null);

  // Nothing persists: the preview URL is released as soon as it is replaced or left.
  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
  }, []);

  const grade = useCallback(async (file: File) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    const url = URL.createObjectURL(file);
    urlRef.current = url;
    setFlow({ kind: "grading", file, url });
    const result = await predict(file);
    setFlow({ kind: "done", file, url, result });
  }, []);

  const reset = useCallback(() => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setFlow({ kind: "idle" });
  }, []);

  if (meta.kind === "loading") {
    return <p className="text-ink-2" role="status">Connecting to the grading service…</p>;
  }

  if (meta.kind !== "ok") {
    return (
      <ServiceProblem
        title="The grading service is not reachable"
        body={`This page reads the model's reported figures and its limits from the service at ${API_BASE}, so it cannot be used until the service responds.`}
        detail={meta.kind === "http" ? `HTTP ${meta.status}: ${meta.detail}` : meta.message}
        action="Try again"
        onAction={reloadMeta}
        hint="Start it with: uvicorn backend.app:app"
      />
    );
  }

  if (flow.kind === "idle") return <UploadView meta={meta.data} onFile={grade} />;

  if (flow.kind === "grading") {
    return (
      <div role="status" aria-live="polite">
        <h1 className="text-3xl font-bold leading-tight">Grading {flow.file.name}</h1>
        <p className="mt-2 text-ink-2">
          The image is being preprocessed, graded, and explained. The page will update when the
          estimate is ready.
        </p>
        <div className="mt-8">
          <Disclaimer items={meta.data.disclaimer} />
        </div>
        <Provenance meta={meta.data} />
      </div>
    );
  }

  return <ResultView meta={meta.data} flow={flow} onAgain={reset} onFile={grade} />;
}

// ------------------------------------------------------------------ upload
function UploadView({ meta, onFile }: { meta: Meta; onFile: (f: File) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <div>
      <h1 className="max-w-[22ch] text-4xl font-bold leading-[1.1] tracking-tight sm:text-5xl">
        Estimate a diabetic retinopathy grade from a fundus photograph
      </h1>
      <p className="mt-4 max-w-[62ch] text-lg leading-relaxed text-ink-2">
        Upload one retinal photograph. It is cropped and normalised the way the training images
        were, then you get an estimated grade, a separate referral decision, the image the model
        actually graded, and a map of which regions moved the score.
      </p>

      <div className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] lg:items-start">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) onFile(f);
          }}
          className={`flex min-h-64 flex-col items-start justify-end border-2 border-dashed p-6 ${
            over ? "border-ink bg-sev-0" : "border-ink-2 bg-sheet"
          }`}
        >
          <p className="text-xl font-bold">Drop a fundus photograph here</p>
          <p className="mt-1 text-ink-2">JPEG or PNG, one image at a time.</p>
          <button
            type="button"
            onClick={() => input.current?.click()}
            className="mt-5 bg-ink px-5 py-2.5 font-bold text-paper hover:bg-sev-4"
          >
            Choose a photograph
          </button>
          <input
            ref={input}
            type="file"
            accept="image/*"
            className="sr-only"
            tabIndex={-1}
            aria-hidden="true"
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) onFile(f);
            }}
          />
        </div>

        <Disclaimer items={meta.disclaimer} />
      </div>

      <Provenance meta={meta} />
    </div>
  );
}

// ------------------------------------------------------------------ result
function ResultView({
  meta,
  flow,
  onAgain,
  onFile,
}: {
  meta: Meta;
  flow: Extract<Flow, { kind: "done" }>;
  onAgain: () => void;
  onFile: (f: File) => void;
}) {
  const { result, file, url } = flow;

  if (result.kind === "http") {
    const tooBig = result.status === 413;
    return (
      <ServiceProblem
        title={tooBig ? "The file is larger than the upload limit" : result.status === 400 ? `${file.name} could not be graded` : `The grading service returned an error (HTTP ${result.status})`}
        body={tooBig ? "Choose a smaller photograph and upload it again." : result.status === 400 ? "Choose a JPEG or PNG fundus photograph and upload it again." : "Try the upload again. If it keeps failing, check the service log."}
        detail={result.detail}
        action="Choose another photograph"
        onAction={onAgain}
        meta={meta}
      />
    );
  }
  if (result.kind === "network") {
    return (
      <ServiceProblem
        title="The grading service stopped responding"
        body="The upload did not reach the service, so no estimate was produced."
        detail={result.message}
        action="Try this photograph again"
        onAction={() => onFile(file)}
        meta={meta}
        hint="If the service was restarted, wait for it to finish loading the model."
      />
    );
  }
  if (result.kind === "malformed") {
    return (
      <ServiceProblem
        title="The grading service sent a response this page does not recognise"
        body="No estimate is shown, because a partial response could be misread."
        detail={result.message}
        action="Choose another photograph"
        onAction={onAgain}
        meta={meta}
      />
    );
  }

  const r = result.data;

  // Declined: no retina. Built first because it is the state most easily treated as an
  // afterthought — and there is no grade to show, by design.
  if (!r.graded) {
    return (
      <div>
        <h1 className="text-3xl font-bold leading-tight sm:text-4xl">No grade produced</h1>
        <div className="mt-5 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:items-start">
          <div className="flex items-start gap-4">
            {/* eslint-disable-next-line @next/next/no-img-element -- local object URL */}
            <img src={url} alt={`Uploaded file ${file.name}`} className="h-28 w-28 shrink-0 bg-viewer object-contain" />
            <p className="text-[0.95rem] leading-snug text-ink-2">
              Uploaded file: {file.name}
            </p>
          </div>
          <div>
            <div role="status" className="border-4 border-ink bg-sheet px-5 py-4">
              <p className="max-w-[68ch] text-lg leading-snug">{r.guard.message}</p>
            </div>
            <button type="button" onClick={onAgain} className="mt-5 bg-ink px-5 py-2.5 font-bold text-paper hover:bg-sev-4">
              Upload a different image
            </button>
            <div className="mt-8">
              <Disclaimer items={r.disclaimer} />
            </div>
          </div>
        </div>
        <Provenance meta={meta} />
      </div>
    );
  }

  // Graded — including the two framing warnings, which qualify the grade rather than
  // withhold it.
  const warned = r.guard.status !== "ok";
  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-4">
        <h1 className="text-3xl font-bold leading-tight sm:text-4xl">Estimate for {file.name}</h1>
        <button type="button" onClick={onAgain} className="border border-ink px-4 py-2 font-bold hover:bg-sheet">
          Grade another image
        </button>
      </div>

      <div className="mt-7 grid gap-10 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)] lg:items-start">
        <div className="order-2 lg:order-1">
          <ImageViewer
            preprocessed={r.preprocessed_png}
            overlay={r.overlay_png}
            originalUrl={url}
            fileName={file.name}
          />
        </div>

        <div className="order-1 space-y-6 lg:order-2">
          {warned && <GuardNotice guard={r.guard} />}
          <Readout r={r} />
          <Disclaimer items={r.disclaimer} />
          {!warned && <GuardNotice guard={r.guard} />}
        </div>
      </div>

      <Provenance meta={meta} />
    </div>
  );
}

// ------------------------------------------------------------------ problems
function ServiceProblem({
  title,
  body,
  detail,
  action,
  onAction,
  hint,
  meta,
}: {
  title: string;
  body: string;
  detail: string;
  action: string;
  onAction: () => void;
  hint?: string;
  meta?: Meta;
}) {
  return (
    <div role="alert">
      <h1 className="max-w-[30ch] text-3xl font-bold leading-tight">{title}</h1>
      <p className="mt-3 max-w-[62ch] text-lg leading-relaxed">{body}</p>
      <p className="mt-3 max-w-[70ch] border-l-4 border-rule pl-4 text-[0.95rem] text-ink-2">
        Service said: {detail}
      </p>
      {hint && <p className="mt-2 text-[0.95rem] text-ink-2">{hint}</p>}
      <button type="button" onClick={onAction} className="mt-6 bg-ink px-5 py-2.5 font-bold text-paper hover:bg-sev-4">
        {action}
      </button>
      {meta && (
        <>
          <div className="mt-10">
            <Disclaimer items={meta.disclaimer} />
          </div>
          <Provenance meta={meta} />
        </>
      )}
    </div>
  );
}
