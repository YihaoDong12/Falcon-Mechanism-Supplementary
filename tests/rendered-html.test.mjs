import assert from "node:assert/strict";
import { access, readFile, stat } from "node:fs/promises";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the complete supplementary archive", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Three-Joint, Five-DOF Falcon-Inspired Flapping Mechanism · Supplementary Materials<\/title>/i);
  assert.match(html, /UPDATED 12 AUG 2026/);
  assert.match(html, /media\/web\/optimization-convergence\.mp4/);
  assert.match(html, /media\/web\/optimized-mechanism\.mp4/);
  assert.equal((html.match(/media\/web\/optimized-mechanism\.mp4/g) ?? []).length, 1);
  assert.equal((html.match(/media\/web\/flight-side\.mp4/g) ?? []).length, 1);
  assert.match(html, /media\/web\/dlc-side\.mp4/);
  assert.match(html, /Side-view marker tracking · slow motion/);
  assert.match(html, /media\/web\/design-variables\.mp4/);
  assert.match(html, /media\/web\/trajectory-front\.mp4/);
  assert.match(html, /media\/web\/trajectory-side\.mp4/);
  assert.match(html, /media\/web\/trajectory-top\.mp4/);
  assert.match(html, /media\/web\/trajectory-oblique\.mp4/);
  assert.match(html, /1,074,004/);
  assert.doesNotMatch(html, /Full-cycle feasible/);
  assert.match(html, /Tip RMSE · mm/);
  assert.match(html, /45\.56/);
  assert.match(html, /07 — REPRODUCIBILITY/);
  assert.match(html, /reproducibility\/data\/cma_generations\.csv/);
  assert.doesNotMatch(html, /codex-preview|loading skeleton/i);
});

test("ships the paper, scientific videos, and replay data", async () => {
  const [page, layout, paperInfo, optimizationVideoInfo, variablesVideoInfo, resultVideoInfo] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    stat(new URL("../public/media/manuscript.pdf", import.meta.url)),
    stat(new URL("../public/media/web/optimization-convergence.mp4", import.meta.url)),
    stat(new URL("../public/media/web/design-variables.mp4", import.meta.url)),
    stat(new URL("../public/media/web/optimized-mechanism.mp4", import.meta.url)),
  ]);

  assert.ok(paperInfo.size > 10_000_000);
  assert.ok(optimizationVideoInfo.size > 1_000_000);
  assert.ok(variablesVideoInfo.size > 1_000_000);
  assert.ok(resultVideoInfo.size > 1_000_000);
  assert.match(page, /evaluation 1,074,004/);
  assert.match(page, /6,507 generations/);
  assert.match(layout, /CMA-ES \+ SQP optimization evidence/);

  await Promise.all([
    access(new URL("public/reproducibility/model/fourbar3d_python.py", templateRoot)),
    access(new URL("public/reproducibility/data/optimization_input.json", templateRoot)),
    access(new URL("public/reproducibility/data/inverse_rotated_strict_trajectories.csv", templateRoot)),
    access(new URL("public/reproducibility/data/sha256_manifest.json", templateRoot)),
    access(new URL("public/media/web/trajectory-front.mp4", templateRoot)),
    access(new URL("public/media/web/trajectory-side.mp4", templateRoot)),
    access(new URL("public/media/web/trajectory-top.mp4", templateRoot)),
    access(new URL("public/media/web/trajectory-oblique.mp4", templateRoot)),
  ]);
});
