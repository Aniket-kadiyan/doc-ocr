import type { Metadata } from "next";
import ClientDiagnostics from "@/components/ClientDiagnostics";
import { AuthProvider } from "@/components/AuthProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "QTraX — Drawing to Quality Intelligence",
  description:
    "Engineering drawing inspection, ballooning and checksheet generation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased" suppressHydrationWarning>
        <AuthProvider>
          <ClientDiagnostics />
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
