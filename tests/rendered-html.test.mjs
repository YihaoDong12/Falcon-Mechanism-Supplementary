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
  assert.match(html, /<title>Trajectory Synthesis of a Three-Joint, Five-DOF Falcon-Inspired Flapping Mechanism by Sensitivity-Partitioned CMA-ES and SQP<\/title>/i);
  assert.match(html, /UPDATED 13 AUG 2026/);
  assert.match(html, /media\/web\/optimization-convergence\.mp4/);
  assert.match(html, /media\/web\/optimized-mechanism\.mp4/);
  assert.equal((html.match(/media\/web\/optimized-mechanism\.mp4/g) ?? []).length, 1);
  assert.equal((html.match(/media\/web\/flight-side\.mp4/g) ?? []).length, 1);
  assert.match(html, /media\/web\/dlc-side\.mp4/);
  assert.match(html, /Side-view marker tracking/);
  assert.match(html, /media\/web\/design-variables-v3\.mp4/);
  assert.match(html, /media\/web\/trajectory-front\.mp4/);
  assert.match(html, /media\/web\/trajectory-side\.mp4/);
  assert.match(html, /media\/web\/trajectory-top\.mp4/);
  assert.match(html, /media\/web\/trajectory-oblique\.mp4/);
  assert.match(html, /4,669,502/);
  assert.match(html, /3,520,143/);
  assert.match(html, /39\.04/);
  assert.match(html, /39\.60/);
  assert.match(html, /39\.32 mm/);
  assert.match(html, /13\.56%/);
  assert.match(html, /media\/mechanism\.png/);
  assert.match(html, /media\/optimization-framework\.png/);
  assert.match(html, /media\/periodic-l6-extension\.png/);
  assert.match(html, /Rejected by full-cycle checks/);
  assert.doesNotMatch(html, /Selected evaluation/);
  assert.doesNotMatch(html, /Full-cycle feasible/);
  assert.match(html, /Extended tip RMSE · mm/);
  assert.match(html, /Extended wrist RMSE · mm/);
  assert.match(html, /07 — REPRODUCIBILITY/);
  assert.match(html, /reproducibility\/data\/cma_generations\.csv/);
  assert.match(html, /reproducibility\/l6_extension\/results\/phasewise_replay\.csv/);
  assert.doesNotMatch(html, /codex-preview|loading skeleton/i);
});

test("ships the paper, scientific videos, and replay data", async () => {
  const [page, layout, paperInfo, optimizationVideoInfo, variablesVideoInfo, resultVideoInfo] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    stat(new URL("../public/media/manuscript.pdf", import.meta.url)),
    stat(new URL("../public/media/web/optimization-convergence.mp4", import.meta.url)),
    stat(new URL("../public/media/web/design-variables-v3.mp4", import.meta.url)),
    stat(new URL("../public/media/web/optimized-mechanism.mp4", import.meta.url)),
  ]);

  assert.ok(paperInfo.size > 20_000_000);
  assert.ok(optimizationVideoInfo.size > 1_000_000);
  assert.ok(variablesVideoInfo.size > 1_000_000);
  assert.ok(resultVideoInfo.size > 1_000_000);
  assert.match(page, /evaluation 1,074,004/);
  assert.match(page, /6,507 generations/);
  assert.match(layout, /periodic-L6 extension/);

  await Promise.all([
    access(new URL("public/reproducibility/model/fourbar3d_python.py", templateRoot)),
    access(new URL("public/reproducibility/data/optimization_input.json", templateRoot)),
    access(new URL("public/reproducibility/data/inverse_rotated_strict_trajectories.csv", templateRoot)),
    access(new URL("public/reproducibility/data/sha256_manifest.json", templateRoot)),
    access(new URL("public/media/web/trajectory-front.mp4", templateRoot)),
    access(new URL("public/media/web/trajectory-side.mp4", templateRoot)),
    access(new URL("public/media/web/trajectory-top.mp4", templateRoot)),
    access(new URL("public/media/web/trajectory-oblique.mp4", templateRoot)),
    access(new URL("public/media/mechanism.png", templateRoot)),
    access(new URL("public/media/optimization-framework.png", templateRoot)),
    access(new URL("public/media/periodic-l6-extension.png", templateRoot)),
    access(new URL("public/reproducibility/l6_extension/model/fourbar_optimization.py", templateRoot)),
    access(new URL("public/reproducibility/l6_extension/input/optimization_input.json", templateRoot)),
    access(new URL("public/reproducibility/l6_extension/results/phasewise_replay.csv", templateRoot)),
    access(new URL("public/reproducibility/l6_extension/results/independent_replay_audit.json", templateRoot)),
  ]);
});
