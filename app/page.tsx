"use client";

import { useEffect, useRef, useState } from "react";

const sections = [
  ["01", "Flight", "飞行观测"],
  ["02", "Reconstruction", "轨迹重建"],
  ["03", "Mechanism", "机构模型"],
  ["04", "Optimization", "优化过程"],
  ["05", "Result", "最终结果"],
  ["06", "Paper", "完整论文"],
];

function useCsv(path: string) {
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  useEffect(() => {
    fetch(path)
      .then((r) => r.text())
      .then((text) => {
        const [head, ...lines] = text.trim().split(/\r?\n/);
        const keys = head.split(",");
        setRows(lines.map((line) => Object.fromEntries(line.split(",").map((v, i) => [keys[i], v]))));
      });
  }, [path]);
  return rows;
}

function AnimatedCurve({ mode }: { mode: "trajectory" | "convergence" | "parameters" | "partition" }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const convergence = useCsv("/data/convergence.csv");
  const periodic = useCsv("/data/periodic-curves.csv");
  const splits = useCsv("/data/split-history.csv");
  const targets = useCsv("/data/target-curves.csv");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let start = performance.now();

    const draw = (now: number) => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#f3f1ea";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(19,32,37,.1)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 6; i++) {
        ctx.beginPath(); ctx.moveTo(42, (h - 45) * i / 6 + 8); ctx.lineTo(w - 18, (h - 45) * i / 6 + 8); ctx.stroke();
      }
      const progress = ((now - start) % 7000) / 7000;
      const plot = (values: number[], color: string, width = 2.5, limit = progress) => {
        if (values.length < 2) return;
        const clean = values.filter(Number.isFinite);
        const min = Math.min(...clean), max = Math.max(...clean);
        const n = Math.max(2, Math.floor(values.length * limit));
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = 42 + (w - 64) * i / Math.max(1, values.length - 1);
          const y = h - 28 - (h - 58) * ((values[i] - min) / Math.max(1e-9, max - min));
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
      };

      if (mode === "trajectory") {
        const points = targets;
        const n = Math.max(2, Math.floor(points.length * progress));
        const drawLoop = (tip: boolean, color: string) => {
          const xKey = tip ? "tip_x_mm" : "wrist_x_mm";
          const zKey = tip ? "tip_z_mm" : "wrist_z_mm";
          const allX = points.flatMap((r) => [Number(r.wrist_x_mm), Number(r.tip_x_mm)]).filter(Number.isFinite);
          const allZ = points.flatMap((r) => [Number(r.wrist_z_mm), Number(r.tip_z_mm)]).filter(Number.isFinite);
          if (!allX.length || !allZ.length) return;
          const minX = Math.min(...allX), maxX = Math.max(...allX), minZ = Math.min(...allZ), maxZ = Math.max(...allZ);
          ctx.strokeStyle = color; ctx.lineWidth = tip ? 4 : 3; ctx.beginPath();
          for (let i = 0; i < n; i++) {
            const x = 42 + (w - 64) * (Number(points[i][xKey]) - minX) / Math.max(1e-9, maxX - minX);
            const y = h - 28 - (h - 58) * (Number(points[i][zKey]) - minZ) / Math.max(1e-9, maxZ - minZ);
            i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          }
          ctx.stroke();
        };
        drawLoop(true, "#e45f4c"); drawLoop(false, "#087f8c");
        ctx.fillStyle = "#132025"; ctx.font = "600 12px ui-monospace"; ctx.fillText(`PHASE  ${(progress * 360).toFixed(0).padStart(3, "0")}°`, 18, 24);
      } else if (mode === "convergence") {
        const values = convergence.filter((r) => r.phase === "primary").map((r) => Number(r.score));
        plot(values.length ? values : [610, 420, 270, 180, 130, 105, 92, 84, 81], "#087f8c", 3);
        ctx.fillStyle = "#132025"; ctx.font = "600 12px ui-monospace"; ctx.fillText("BEST OBJECTIVE · EVALUATIONS →", 18, 24);
      } else if (mode === "parameters") {
        const keys = ["L3_mm", "L8_mm", "L32_mm", "B_y_mm"];
        const colors = ["#087f8c", "#e45f4c", "#d7a733", "#31566a"];
        keys.forEach((key, i) => plot(periodic.map((r) => Number(r[key])), colors[i], 2.4, progress));
        ctx.fillStyle = "#132025"; ctx.font = "600 12px ui-monospace"; ctx.fillText("PERIODIC DESIGN VARIABLES · PHASE →", 18, 24);
      } else {
        const rounds = splits.map((r) => Number(r.round)).filter(Number.isFinite);
        const maxRound = Math.max(1, ...rounds);
        const bars = Array.from({ length: maxRound + 1 }, (_, round) => rounds.filter((v) => v <= round).length + 1);
        const shown = Math.max(1, Math.floor(bars.length * progress));
        const max = Math.max(...bars);
        bars.slice(0, shown).forEach((v, i) => {
          const bw = (w - 65) / bars.length;
          ctx.fillStyle = i === shown - 1 ? "#e45f4c" : "#087f8c";
          ctx.fillRect(42 + i * bw, h - 28 - (h - 60) * v / max, Math.max(2, bw - 2), (h - 60) * v / max);
        });
        ctx.fillStyle = "#132025"; ctx.font = "600 12px ui-monospace"; ctx.fillText(`SENSITIVITY PARTITIONS · ROUND ${shown - 1}`, 18, 24);
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [mode, convergence, periodic, splits, targets]);

  return <canvas ref={canvasRef} className="data-canvas" aria-label={`${mode} animated data visualization`} />;
}

function MediaCard({ title, kicker, src, poster, children }: { title: string; kicker: string; src?: string; poster?: string; children?: React.ReactNode }) {
  return (
    <article className="media-card">
      <div className="media-frame">
        {src ? <video src={src} poster={poster} autoPlay muted loop playsInline controls={false} /> : children}
        <span className="loop-pill"><i /> LOOP</span>
      </div>
      <div className="card-copy"><span>{kicker}</span><h3>{title}</h3></div>
    </article>
  );
}

export default function Home() {
  const [view, setView] = useState<"front" | "side">("front");
  const [paperOpen, setPaperOpen] = useState(false);

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top"><span>FW</span><b>FALCON / SYNTHESIS</b></a>
        <nav aria-label="Primary navigation">
          {sections.map(([n, en]) => <a key={n} href={`#s${n}`}><small>{n}</small>{en}</a>)}
        </nav>
        <a className="paper-link" href="/media/manuscript.pdf" target="_blank">Paper ↗</a>
      </header>

      <section className="hero" id="top">
        <div className="hero-ghost">FALCON</div>
        <div className="hero-copy">
          <p className="eyebrow">SUPPLEMENTARY MATERIALS · JCDE · 2026</p>
          <h1>Trajectory Synthesis of a<br/><em>Falcon-Inspired</em><br/>Flapping Mechanism</h1>
          <p className="dek">A visual research archive connecting markerless flight reconstruction, analytical mechanism design, and sensitivity-partitioned optimization.</p>
          <div className="authors">Yihao Dong · Vladimir Parezanovic · Yusra Abdulrahman<br/><span>Khalifa University · Aerospace Engineering</span></div>
        </div>
        <div className="hero-media">
          <video src="/media/web/flight-side.mp4" poster="/media/web/flight-side.jpg" autoPlay muted loop playsInline />
          <div className="hero-tag"><span>01 / FIELD RECORDING</span><b>Side view</b></div>
        </div>
        <a href="#s01" className="scroll-cue">SCROLL TO EXPLORE <span>↓</span></a>
      </section>

      <section className="paper-strip">
        <p>Three joints</p><b>05</b><p>Degrees of freedom</p><b>54</b><p>Independent variables</p><b>13</b><p>Contract tests</p>
      </section>

      <section className="chapter" id="s01">
        <div className="section-head"><p>01 — OBSERVATION</p><h2>Flight, seen twice.</h2><span>猎隼飞行原始记录</span></div>
        <div className="split-media">
          <MediaCard kicker="CAMERA A · FRONT" title="Frontal flight sequence" src="/media/web/flight-front.mp4" poster="/media/web/flight-front.jpg" />
          <MediaCard kicker="CAMERA B · SIDE" title="Lateral flight sequence" src="/media/web/flight-side.mp4" poster="/media/web/flight-side.jpg" />
        </div>
        <p className="chapter-note">Synchronized camera views establish the spatial evidence base. The paired recordings expose sweep, extension, and distal-wing motion that cannot be recovered from one projection alone.</p>
      </section>

      <section className="chapter ink" id="s02">
        <div className="section-head light"><p>02 — RECONSTRUCTION</p><h2>Pixels become a<br/>coupled trajectory.</h2><span>DeepLabCut 标定与三维轨迹</span></div>
        <div className="view-switch" role="group" aria-label="Select camera view">
          <button className={view === "front" ? "active" : ""} onClick={() => setView("front")}>Front view</button>
          <button className={view === "side" ? "active" : ""} onClick={() => setView("side")}>Side view</button>
        </div>
        <div className="dlc-stage">
          <video key={view} src={view === "front" ? "/media/web/dlc-front.mp4" : "/media/web/dlc-side.mp4"} poster={view === "front" ? "/media/web/dlc-front.jpg" : "/media/web/dlc-side.jpg"} autoPlay muted loop playsInline />
          <div className="stage-index">DLC / {view === "front" ? "A" : "B"}</div>
        </div>
        <div className="curve-grid">
          <div className="curve-copy"><span>LIVE RECONSTRUCTION</span><h3>Wingbeat phase draws the wrist and wingtip curves.</h3><p>The two trajectories share one coupled six-dimensional arc-length coordinate, preserving physical simultaneity through the wingbeat.</p></div>
          <AnimatedCurve mode="trajectory" />
        </div>
      </section>

      <section className="chapter" id="s03">
        <div className="section-head"><p>03 — ANALYTICAL MODEL</p><h2>One drive.<br/>Five coordinated motions.</h2><span>机构组成与多视角运动</span></div>
        <figure className="publication-figure"><figcaption><b>FIGURE 01</b><span>Analytical mechanism definition · unified source figure</span></figcaption><img className="wide-figure" src="/media/mechanism.png" alt="Falcon-inspired mechanism composition and kinematic views" /></figure>
        <div className="mechanism-notes">
          <div><b>I</b><h3>Root drive</h3><p>Periodic shoulder-root input initiates the closed-loop transmission.</p></div>
          <div><b>II–VI</b><h3>Linked closure</h3><p>Sequential planar loops coordinate the elbow and wrist assemblies.</p></div>
          <div><b>VII–VIII</b><h3>Spatial expansion</h3><p>Out-of-plane rotations recover the distal three-dimensional motion.</p></div>
        </div>
      </section>

      <section className="chapter optimization" id="s04">
        <div className="section-head"><p>04 — SEARCH PROCESS</p><h2>The design space,<br/>made visible.</h2><span>目标函数、自变量与分区演化</span></div>
        <img className="framework" src="/media/optimization-framework.png" alt="Sensitivity-partitioned CMA-ES and SLSQP optimization framework" />
        <div className="animation-grid">
          <MediaCard kicker="OBJECTIVE" title="Best score over candidate evaluations"><AnimatedCurve mode="convergence" /></MediaCard>
          <MediaCard kicker="DESIGN VARIABLES" title="Periodic parameters over wingbeat phase"><AnimatedCurve mode="parameters" /></MediaCard>
          <MediaCard kicker="PARTITIONING" title="Feasible regions created by optimization round"><AnimatedCurve mode="partition" /></MediaCard>
        </div>
        <p className="method-note"><b>Method note.</b> Static sensitivity expands the design bounds; regional CMA-ES searches correlated feasible directions; active-set SLSQP refines the most promising basins.</p>
      </section>

      <section className="chapter result" id="s05">
        <div className="section-head light"><p>05 — SYNTHESIZED MOTION</p><h2>The mechanism<br/>in motion.</h2><span>优化机构与目标轨迹对比</span></div>
        <div className="result-stage">
          <video src="/media/web/optimized-mechanism.mp4" poster="/media/web/optimized-mechanism.jpg" autoPlay muted loop playsInline controls />
          <div className="result-caption"><span>FOUR SYNCHRONIZED VIEWS</span><h3>Generated wrist and wingtip trajectories versus biological targets</h3><p>Move through one full input cycle to inspect mechanism closure, phase correspondence, and spatial curve agreement.</p></div>
        </div>
        <div className="phase-band"><img src="/media/phase-contract.png" alt="Strict initialized phase correspondence between target and model"/><div><span>STRICT PHASE CONTRACT</span><h3>Same index. Same physical moment.</h3><p>Cyclic shifts and reversed traversals are explicitly rejected. The comparison preserves the measured ordering of the wingbeat.</p></div></div>
      </section>

      <section className="chapter paper" id="s06">
        <div className="section-head"><p>06 — ARTICLE</p><h2>Read the complete paper.</h2><span>全文、摘要与章节</span></div>
        <div className="abstract-grid">
          <div><span>ABSTRACT</span><p>Compact flapping mechanisms cannot readily reproduce the synchronized three-dimensional motion of an avian wrist and wingtip because both trajectories must emerge from one periodically driven, closed-loop transmission. Here we establish a falcon-inspired synthesis framework that links markerless flight reconstruction to a three-joint, five-degree-of-freedom analytical mechanism. DeepLabCut observations are corrected for confidence and anatomical length, anchored by a unique measured wrist event, and resampled with one coupled six-dimensional arc-length coordinate. Fifty-four independent variables describe the mechanism and common target pose, while sensitivity-guided partitioning, CMA-ES, and SLSQP provide a reproducible synthesis workflow.</p></div>
          <ol>{["Introduction", "Problem formulation & framework", "Biological trajectory preprocessing", "Sensitivity-guided optimization", "Results", "Conclusions", "Mechanism reconstruction", "Optimization domain"].map((x, i) => <li key={x}><b>{String(i + 1).padStart(2, "0")}</b>{x}</li>)}</ol>
        </div>
        <div className="paper-actions"><button onClick={() => setPaperOpen(!paperOpen)}>{paperOpen ? "Close embedded paper" : "Open paper in this window"}</button><a href="/media/manuscript.pdf" download>Download PDF ↓</a></div>
        {paperOpen && <iframe className="pdf-frame" src="/media/manuscript.pdf" title="Complete research paper" />}
      </section>

      <footer><div className="brand"><span>FW</span><b>FALCON / SYNTHESIS</b></div><p>Supplementary materials archive · Khalifa University · 2026</p><a href="#top">BACK TO TOP ↑</a></footer>
    </main>
  );
}
