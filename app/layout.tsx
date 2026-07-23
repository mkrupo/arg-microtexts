import type { ReactNode } from "react";

import "./globals.css";

export const metadata = {
  title: "Arg-Microtexts EDU Explorer",
  description:
    "Inspect German ADUs, automatic EDU boundary proposals, model agreement, and aligned English gold EDUs.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
