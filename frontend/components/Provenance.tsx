import type { Meta } from "@/lib/types";

// Built from /meta FIELDS, not from `meta.provenance`. That string deliberately ends with
// the shipped seed's own validation QWK (DECISION-069 keeps both figures side by side for
// auditability), and frontend rule 2 forbids that figure in the interface. The side-by-side
// record lives in analysis/deployment/deployment.json; the page quotes the 3-seed means.
export function Provenance({ meta }: { meta: Meta }) {
  return (
    <footer className="mt-14 border-t border-rule pt-4 text-[0.85rem] leading-snug text-ink-2">
      <p className="max-w-[80ch]">
        Model {meta.run_id}, {meta.arch} with an {meta.head.replace(/_/g, "-")} head, seed{" "}
        {meta.seed}. Reported figures are means over three training runs: validation quadratic
        weighted kappa {meta.reported_val_qwk.toFixed(4)}, referable sensitivity{" "}
        {meta.reported_val_sens_at_spec95.toFixed(4)} at 95% specificity.
      </p>
      <p className="mt-1">Nothing you upload is stored.</p>
    </footer>
  );
}
