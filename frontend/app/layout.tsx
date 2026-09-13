import type { Metadata } from "next";
import { Atkinson_Hyperlegible_Next } from "next/font/google";
import Link from "next/link";
import "./globals.css";

// Designed by the Braille Institute for readers with low vision — the population
// diabetic retinopathy produces. Chosen for that reason, not as a neutral default.
const atkinson = Atkinson_Hyperlegible_Next({
  subsets: ["latin"],
  weight: "variable",
  variable: "--font-atkinson",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Retinopathy grade estimate — research prototype",
  description:
    "Estimates a diabetic retinopathy grade from a fundus photograph and reports the limits of that estimate. Not a medical device.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={atkinson.variable}>
      <body className="min-h-screen">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:bg-sheet focus:px-3 focus:py-2"
        >
          Skip to content
        </a>
        <header className="border-b border-rule">
          <nav
            aria-label="Primary"
            className="mx-auto flex max-w-6xl flex-wrap items-baseline justify-between gap-x-6 gap-y-2 px-5 py-4 sm:px-8"
          >
            <Link href="/" className="font-bold tracking-tight text-ink">
              Retinopathy grade estimate
            </Link>
            <div className="flex gap-5 text-ink-2">
              <Link href="/" className="hover:text-ink">
                Grade an image
              </Link>
              <Link href="/about" className="hover:text-ink">
                About this model
              </Link>
            </div>
          </nav>
        </header>
        <main id="main" className="mx-auto max-w-6xl px-5 py-8 sm:px-8 sm:py-12">
          {children}
        </main>
      </body>
    </html>
  );
}
