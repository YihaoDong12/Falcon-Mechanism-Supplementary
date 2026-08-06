import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geist = Geist({ variable: "--font-sans", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-mono", subsets: ["latin"] });

const title = "Falcon-Inspired Flapping Mechanism · Supplementary Materials";
const description = "Flight reconstruction, analytical mechanism motion, and sensitivity-partitioned optimization for a falcon-inspired flapping mechanism.";

export const metadata: Metadata = {
  metadataBase: new URL("https://yihaodong12.github.io/Falcon-Mechanism-Supplementary/"),
  title,
  description,
  icons: { icon: "favicon.svg" },
  openGraph: { title, description, images: [{ url: "og.png", width: 1200, height: 630 }] },
  twitter: { card: "summary_large_image", title, description, images: ["og.png"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geist.variable} ${mono.variable}`}>{children}</body></html>;
}
