import type { Metadata } from "next";
import "./globals.css";
import { QueryProvider } from "./query-provider";

export const metadata: Metadata = {
  title: "VeriClaim AI",
  description: "Evidence-driven verification for technical claims"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><QueryProvider>{children}</QueryProvider></body></html>;
}
