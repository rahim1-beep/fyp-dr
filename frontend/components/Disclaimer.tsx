// The four statements from the API, verbatim and in the page flow. Never collapsed,
// never summarised, never behind a control — CLAUDE.md frontend rule 3.
export function Disclaimer({ items, id = "limits" }: { items: string[]; id?: string }) {
  return (
    <section aria-labelledby={id} className="border-l-4 border-ink bg-sheet px-5 py-4">
      <h2 id={id} className="text-base font-bold">
        Limits of this prototype
      </h2>
      <ul className="mt-2 space-y-1.5 text-[0.95rem] leading-snug">
        {items.map((text) => (
          <li key={text} className="max-w-[70ch]">
            {text}
          </li>
        ))}
      </ul>
    </section>
  );
}
