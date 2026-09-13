import type { GuardInRange } from "@/lib/types";

// `guard.message` is rendered verbatim in every state (CLAUDE.md frontend rule 5). Only the
// heading and the weight around it are the interface's own.
export function GuardNotice({ guard }: { guard: GuardInRange }) {
  if (guard.status === "ok") {
    return (
      <p className="max-w-[68ch] text-[0.95rem] leading-snug text-ink-2">
        <span className="font-bold text-ink">Framing check. </span>
        {guard.message}
      </p>
    );
  }

  const heading =
    guard.status === "framing_below_training_range"
      ? "The retina fills less of this image than in the training images"
      : "This image is cropped tighter than the training images";

  return (
    <div role="status" className="border-4 border-ink bg-sheet px-5 py-4">
      <p className="text-lg font-bold leading-snug">{heading}</p>
      <p className="mt-1 max-w-[68ch] leading-snug">{guard.message}</p>
    </div>
  );
}
