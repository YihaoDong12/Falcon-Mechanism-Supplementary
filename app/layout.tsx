import type { Metadata } from "next";
import "./globals.css";

const title = "Three-Joint, Five-DOF Falcon-Inspired Flapping Mechanism · Supplementary Materials";
const description = "Flight reconstruction, phase-locked mechanism motion, strict cold-start CMA-ES + SQP optimization evidence, paper, code, and source data.";

export const metadata: Metadata = {
  metadataBase: new URL("https://yihaodong12.github.io/Falcon-Mechanism-Supplementary/"),
  title,
  description,
  icons: { icon: "favicon.svg" },
  openGraph: { title, description, images: [{ url: "og.png", width: 1200, height: 630 }] },
  twitter: { card: "summary_large_image", title, description, images: ["og.png"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className="nature-type">{children}</body></html>;
}
