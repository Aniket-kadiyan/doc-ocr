import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Doc OCR Box — Engineering Drawing Annotation",
  description:
    "In-house OCR annotation tool for technical drawings with balloon numbering and reading-order detection.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
