"use client";

import { useEffect, useRef, useState } from "react";

const sections = [
  ["01", "Flight", "飞行观测"],
  ["02", "Reconstruction", "轨迹重建"],
  ["03", "Mechanism", "机构模型"],
  ["04", "Optimization", "优化过程"],
  ["05", "Result", "最终结果"],
  ["06", "Paper", "完整论文"],
  ["07", "Reproduce", "复现材料"],
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
  const convergence = useCsv("data/convergence.csv");
  const periodic = useCsv("data/periodic-curves.csv");
  const splits = useCsv("data/split-history.csv");
  const targets = useCsv("data/target-curves.csv");

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
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(31,41,51,.1)";
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
        drawLoop(true, "#E64B35"); drawLoop(false, "#3C5488");
        ctx.fillStyle = "#1F2933"; ctx.font = "600 14px Arial"; ctx.fillText(`PHASE  ${(progress * 360).toFixed(0).padStart(3, "0")}°`, 18, 24);
      } else if (mode === "convergence") {
        const values = convergence.filter((r) => r.phase === "primary").map((r) => Number(r.score));
        plot(values.length ? values : [610, 420, 270, 180, 130, 105, 92, 84, 81], "#3C5488", 3);
        ctx.fillStyle = "#1F2933"; ctx.font = "600 14px Arial"; ctx.fillText("BEST OBJECTIVE · EVALUATIONS →", 18, 24);
      } else if (mode === "parameters") {
        const keys = ["L3_mm", "L8_mm", "L32_mm", "B_y_mm"];
        const colors = ["#3C5488", "#E64B35", "#00A087", "#4DBBD5"];
        keys.forEach((key, i) => plot(periodic.map((r) => Number(r[key])), colors[i], 2.4, progress));
        ctx.fillStyle = "#1F2933"; ctx.font = "600 14px Arial"; ctx.fillText("PERIODIC DESIGN VARIABLES · PHASE →", 18, 24);
      } else {
        const rounds = splits.map((r) => Number(r.round)).filter(Number.isFinite);
        const maxRound = Math.max(1, ...rounds);
        const bars = Array.from({ length: maxRound + 1 }, (_, round) => rounds.filter((v) => v <= round).length + 1);
        const shown = Math.max(1, Math.floor(bars.length * progress));
        const max = Math.max(...bars);
        bars.slice(0, shown).forEach((v, i) => {
          const bw = (w - 65) / bars.length;
          ctx.fillStyle = i === shown - 1 ? "#E64B35" : "#3C5488";
          ctx.fillRect(42 + i * bw, h - 28 - (h - 60) * v / max, Math.max(2, bw - 2), (h - 60) * v / max);
        });
        ctx.fillStyle = "#1F2933"; ctx.font = "600 14px Arial"; ctx.fillText(`SENSITIVITY PARTITIONS · ROUND ${shown - 1}`, 18, 24);
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [mode, convergence, periodic, splits, targets]);

  return <canvas ref={canvasRef} className="data-canvas" aria-label={`${mode} animated data visualization`} />;
}

function MediaCard({ title, kicker, src, poster, playbackRate = 1, children }: { title: string; kicker: string; src?: string; poster?: string; playbackRate?: number; children?: React.ReactNode }) {
  return (
    <article className="media-card">
      <div className="media-frame">
        {src ? <video src={src} poster={poster} autoPlay muted loop playsInline controls={false} onLoadedMetadata={(event) => { event.currentTarget.playbackRate = playbackRate; }} /> : children}
        <span className="loop-pill"><i /> LOOP</span>
      </div>
      <div className="card-copy"><span>{kicker}</span><h3>{title}</h3></div>
    </article>
  );
}

function EvidenceVideo({ src, poster, label, caption }: { src: string; poster: string; label: string; caption: string }) {
  return <figure className="evidence-figure"><div className="figure-media figure-video-frame"><video src={src} poster={poster} autoPlay muted loop playsInline controls preload="metadata" aria-label={caption} /></div><figcaption><b>{label}</b><span>{caption}</span></figcaption></figure>;
}

function DownloadItem({ href, label, title, meta }: { href: string; label: string; title: string; meta: string }) {
  return <a className="download-item" href={href} download><span>{label}</span><h3>{title}</h3><p>{meta}</p><b>Download ↓</b></a>;
}

export default function Home() {
  const [paperOpen, setPaperOpen] = useState(false);

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top"><span>FW</span><b>FALCON / SYNTHESIS</b></a>
        <nav aria-label="Primary navigation">
          {sections.map(([n, en]) => <a key={n} href={`#s${n}`}><small>{n}</small>{en}</a>)}
        </nav>
        <a className="paper-link" href="media/manuscript.pdf" target="_blank">Paper ↗</a>
      </header>

      <section className="hero" id="top">
        <div className="hero-ghost">FALCON</div>
        <div className="hero-copy">
          <p className="eyebrow">SUPPLEMENTARY MATERIALS · JCDE · UPDATED 13 AUG 2026</p>
          <h1>Trajectory Synthesis of a<br/><em>Three-Joint, Five-DOF</em><br/>Falcon-Inspired Flapping Mechanism</h1>
          <p className="dek">By sensitivity-partitioned CMA-ES and SQP. This record links confidence-aware markerless flight reconstruction, analytical closed-loop synthesis, event-level replay, and the periodic-L6 extension reported in the updated manuscript.</p>
          <div className="authors">Yihao Dong · Vladimir Parezanovic · Yusra Abdulrahman<br/><span>Khalifa University · Aerospace Engineering</span></div>
        </div>
        <div className="hero-media">
          <video src="media/web/flight-side.mp4" poster="media/web/flight-side.jpg" autoPlay muted loop playsInline controls />
          <div className="hero-tag"><span>FIELD RECORDING · SLOW MOTION</span><b>High-speed lateral flight</b></div>
        </div>
        <a href="#s01" className="scroll-cue">SCROLL TO EXPLORE <span>↓</span></a>
      </section>

      <section className="paper-strip result-strip">
        <b>4,669,502</b><p>Mechanisms evaluated</p><b>3,520,143</b><p>Rejected by full-cycle checks</p><b>39.04</b><p>Extended tip RMSE · mm</p><b>39.60</b><p>Extended wrist RMSE · mm</p>
      </section>

      <section className="chapter" id="s01">
        <div className="section-head"><p>01 — OBSERVATION</p><h2>Source flight recordings</h2><span>猎隼飞行原始记录</span></div>
        <div className="split-media single-media">
          <MediaCard kicker="CAMERA A · REAR" title="Rear take-off sequence" src="media/web/flight-rear.mp4" poster="media/web/flight-rear.jpg" />
        </div>
        <p className="chapter-note">The rear take-off sequence complements the lateral slow-motion record above, exposing sweep, extension, and distal-wing motion from a second projection.</p>
      </section>

      <section className="chapter ink" id="s02">
        <div className="section-head light"><p>02 — RECONSTRUCTION</p><h2>Markerless trajectory reconstruction</h2><span>DeepLabCut 标定与三维轨迹</span></div>
        <div className="split-media dlc-pair">
          <MediaCard kicker="DLC / CAMERA A · FRONT" title="Front-view marker tracking" src="media/web/dlc-front.mp4" poster="media/web/dlc-front.jpg" />
          <MediaCard kicker="DLC / CAMERA B · SIDE" title="Side-view marker tracking" src="media/web/dlc-side.mp4" poster="media/web/dlc-side.jpg" playbackRate={0.2} />
        </div>
        <div className="curve-grid">
          <div className="curve-copy"><span>COUPLED PHASE COORDINATE</span><h3>Wrist and wingtip trajectories are reconstructed at the same wingbeat phase.</h3><p>A shared six-dimensional arc-length coordinate preserves the simultaneity of the paired biological observations.</p></div>
          <div className="trajectory-views">
            <MediaCard kicker="ORTHOGRAPHIC / X–Z" title="Front view" src="media/web/trajectory-front.mp4?v=2" poster="media/web/trajectory-front.jpg?v=2" />
            <MediaCard kicker="ORTHOGRAPHIC / Y–Z" title="Side view" src="media/web/trajectory-side.mp4?v=2" poster="media/web/trajectory-side.jpg?v=2" />
            <MediaCard kicker="ORTHOGRAPHIC / X–Y" title="Top view" src="media/web/trajectory-top.mp4?v=2" poster="media/web/trajectory-top.jpg?v=2" />
            <MediaCard kicker="ORTHOGRAPHIC / X–Y–Z" title="Oblique 3D view" src="media/web/trajectory-oblique.mp4?v=2" poster="media/web/trajectory-oblique.jpg?v=2" />
          </div>
        </div>
      </section>

      <section className="chapter" id="s03">
        <div className="section-head"><p>03 — ANALYTICAL MODEL</p><h2>Analytical mechanism and kinematics</h2><span>机构组成与多视角运动</span></div>
        <EvidenceVideo src="media/web/mechanism-figure-v1.mp4" poster="media/web/mechanism-figure-v1.jpg" label="ANALYTICAL TOPOLOGY" caption="Animated overview of the three-joint, five-DOF mechanism and sequential closed-loop construction" />
        <div className="mechanism-notes">
          <div><b>I</b><h3>Root drive</h3><p>Periodic shoulder-root input initiates the closed-loop transmission.</p></div>
          <div><b>II–VI</b><h3>Linked closure</h3><p>Sequential planar loops coordinate the elbow and wrist assemblies.</p></div>
          <div><b>54D</b><h3>Coupled synthesis</h3><p>Static geometry, periodic Point-B and link inputs, and one common target pose are optimized together.</p></div>
        </div>
      </section>

      <section className="chapter optimization" id="s04">
        <div className="section-head"><p>04 — SEARCH PROCESS</p><h2>Optimization history and convergence</h2><span>目标函数、自变量与分区演化</span></div>
        <div className="figure-media framework-frame figure-video-frame"><video src="media/web/optimization-framework-v1.mp4" poster="media/web/optimization-framework-v1.jpg" autoPlay muted loop playsInline controls preload="metadata" aria-label="Animated sensitivity-partitioned CMA-ES and SQP optimization framework" /></div>
        <div className="optimization-film">
          <video src="media/web/optimization-convergence.mp4" poster="media/web/optimization-convergence.jpg" autoPlay muted loop playsInline controls preload="metadata" />
          <div><span>60-SECOND ITERATION RECORD</span><h3>PCA exploration and CMA-ES convergence</h3><p>The animation separates tested candidates from accepted improvements and uses fixed PCA axes throughout the 54-dimensional search record.</p></div>
        </div>
        <div className="optimization-film variables-film">
          <div><span>54-VARIABLE EVOLUTION</span><h3>Design variables mapped to mechanism response</h3><p>All design variables evolve with the accepted search history while the synchronized top view reports their kinematic consequence.</p></div>
          <video src="media/web/design-variables-v3.mp4" poster="media/web/design-variables-v3.jpg" autoPlay muted loop playsInline controls preload="metadata" />
        </div>
        <div className="optimization-stats">
          <div><b>6,507</b><span>CMA-ES generations</span></div><div><b>639</b><span>Split decisions</span></div><div><b>340</b><span>Executed partitions</span></div><div><b>51 / 51</b><span>Accepted SQP calls</span></div>
        </div>
        <p className="method-note"><b>Method note.</b> Sensitivity screening partitions influential directions; regional CMA-ES explores correlated feasible regions; SQP refines candidates that pass every full-cycle closure and continuity check.</p>
      </section>

      <section className="chapter result" id="s05">
        <div className="section-head light"><p>05 — SYNTHESIZED MOTION</p><h2>Verified kinematics and periodic-L6 extension</h2><span>基础事件与扩展结果</span></div>
        <div className="result-stage">
          <video src="media/web/optimized-mechanism.mp4" poster="media/web/optimized-mechanism.jpg" autoPlay muted loop playsInline controls preload="metadata" />
          <div className="result-caption"><span>54D VERIFIED BASE EVENT</span><h3>Generated wrist and wingtip trajectories versus biological targets</h3><p>The 60-second orbit records the replayable 54-dimensional event before the additional periodic-link study. It reports the Point-B path and the periodic L3 and L8 inputs over one complete phase cycle.</p><dl><div><dt>Tip RMSE</dt><dd>45.43 mm</dd></div><div><dt>Wrist RMSE</dt><dd>45.56 mm</dd></div><div><dt>Replay error</dt><dd>≤ 6.82 × 10⁻¹³</dd></div></dl></div>
        </div>
        <div className="extension-result">
          <div className="extension-summary"><div><span>PERIODIC-L6 EXTENSION</span><h3>A short additional stroke reduces the coupled residual.</h3><p>Equal-budget screening identified L6 as the strongest of five tested single-link extensions. Joint reoptimization reduced the coupled RMSE from 45.49 to 39.32 mm while preserving continuous full-cycle assembly. The gain belongs to the enlarged design space rather than to L6 in isolation.</p></div><dl><div><dt>Coupled RMSE</dt><dd>39.32 mm</dd></div><div><dt>Reduction</dt><dd>13.56%</dd></div><div><dt>L6 range</dt><dd>0.85 mm</dd></div><div><dt>Selected links</dt><dd>1 of 5</dd></div></dl></div>
          <figure><video src="media/web/periodic-l6-extension-v1.mp4" poster="media/web/periodic-l6-extension-v1.jpg" autoPlay muted loop playsInline controls preload="metadata" aria-label="Animated periodic L6 screening, convergence, phase-resolved distance, optimized length, and mechanism states"/><figcaption><b>EXTENDED DESIGN SPACE</b><span>Link screening, endpoint convergence, phase-resolved improvement, periodic L6 law, and ten mechanism states.</span></figcaption></figure>
        </div>
      </section>

      <section className="chapter paper" id="s06">
        <div className="section-head"><p>06 — ARTICLE</p><h2>Manuscript and appendices</h2><span>全文、摘要与章节</span></div>
        <div className="abstract-grid">
          <div><span>ABSTRACT · UPDATED MANUSCRIPT</span><p>Bird wings combine rotary joint motion with muscle-driven shortening to coordinate wrist and wingtip trajectories that differ in spatial extent, orientation, and local progression. Compact flapping mechanisms must approximate this distributed actuation with few periodic inputs, turning mechanism synthesis into a high-dimensional inverse problem constrained by full-cycle closure. Here we link confidence-aware markerless reconstruction of falcon flight to a three-joint, five-degree-of-freedom analytical mechanism. Sensitivity-guided partitioning localizes responsive regions, regional CMA-ES explores correlated feasible directions, and SQP refines selected basins. The replayable design captured the paired motion envelope, while periodic variation of an additional link further reduced the residual. The remaining mismatch was dominated by insufficient relative wrist–wingtip motion authority rather than phase discontinuity. The present evidence is kinematic and awaits structural, aerodynamic, and flight validation.</p></div>
          <ol>{["Introduction", "Problem formulation & research framework", "Biological trajectory preprocessing", "Biologically inspired hybrid optimization", "Results, residual geometry & periodic L6", "Conclusions & outlook", "Replication of results", "Appendices: model & optimization domain"].map((x, i) => <li key={x}><b>{String(i + 1).padStart(2, "0")}</b>{x}</li>)}</ol>
        </div>
        <div className="paper-actions"><button onClick={() => setPaperOpen(!paperOpen)}>{paperOpen ? "Close embedded paper" : "Open paper in this window"}</button><a href="media/manuscript.pdf" download>Download PDF ↓</a></div>
        {paperOpen && <iframe className="pdf-frame" src="media/manuscript.pdf" title="Complete research paper" />}
      </section>

      <section className="chapter reproducibility" id="s07">
        <div className="section-head"><p>07 — REPRODUCIBILITY</p><h2>Code and data availability</h2><span>代码、原始数据与验证记录</span></div>
        <div className="repro-intro"><p>The archive separates the verified 54-dimensional event at evaluation 1,074,004 from the later periodic-L6 extension. It includes the frozen base contract and histories together with the extended model, winning event, link-screening comparison, phasewise replay, convergence record, and independent audit.</p><a href="https://github.com/YihaoDong12/Falcon-Mechanism-Supplementary/tree/main/public/reproducibility" target="_blank">Browse all files on GitHub ↗</a></div>
        <div className="download-grid">
          <DownloadItem href="reproducibility/model/fourbar3d_python.py" label="PYTHON · MODEL" title="Analytical closed-loop mechanism" meta="Three-dimensional sequential mechanism solver" />
          <DownloadItem href="reproducibility/model/fourbar_optimization.py" label="PYTHON · OPTIMIZATION" title="Optimization problem definition" meta="Bounds, feasibility contract, and paired objective" />
          <DownloadItem href="reproducibility/data/optimization_input.json" label="JSON · CONTRACT" title="Frozen 54-variable input" meta="Strict initialized equal-arc optimization contract" />
          <DownloadItem href="reproducibility/data/target_source_workbook.xlsx" label="XLSX · RAW SOURCE" title="Target source workbook" meta="Reconstruction source with initialization record" />
          <DownloadItem href="reproducibility/data/final_variables_and_bounds.csv" label="CSV · RESULT" title="Final variables and bounds" meta="Replayed event vector with engineering limits" />
          <DownloadItem href="reproducibility/data/inverse_rotated_strict_trajectories.csv" label="CSV · TRAJECTORIES" title="Strict same-phase trajectories" meta="Generated and target wrist / wingtip coordinates" />
          <DownloadItem href="reproducibility/data/cma_generations.csv" label="CSV · HISTORY" title="CMA-ES generation record" meta="6,507 generations and candidate statistics" />
          <DownloadItem href="reproducibility/data/partition_splits.csv" label="CSV · PARTITIONS" title="Sensitivity split history" meta="Resolved regional split decisions" />
          <DownloadItem href="reproducibility/data/slsqp_calls.csv" label="CSV · LOCAL SEARCH" title="Accepted SQP calls" meta="Event-level local refinement history" />
        </div>
        <div className="repro-subhead"><span>PERIODIC-L6 EXTENSION</span><h3>Code and source data for the enlarged design space</h3></div>
        <div className="download-grid extension-downloads">
          <DownloadItem href="reproducibility/l6_extension/model/fourbar_optimization.py" label="PYTHON · EXTENDED MODEL" title="Periodic-link optimization model" meta="L6 Fourier law, bounds, feasibility, and paired objective" />
          <DownloadItem href="reproducibility/l6_extension/input/optimization_input.json" label="JSON · EXTENDED CONTRACT" title="Periodic-L6 input contract" meta="Frozen configuration used for link screening and reoptimization" />
          <DownloadItem href="reproducibility/l6_extension/results/best_event.json" label="JSON · EXTENDED RESULT" title="Winning L6 event" meta="Replayed design vector and final metrics" />
          <DownloadItem href="reproducibility/l6_extension/results/phasewise_replay.csv" label="CSV · PHASEWISE REPLAY" title="Extended trajectory replay" meta="Target, generated endpoints, and phase-resolved distances" />
          <DownloadItem href="reproducibility/l6_extension/results/link_screening_comparison.csv" label="CSV · LINK SCREENING" title="Five-link comparison" meta="Equal-budget results supporting selection of L6" />
          <DownloadItem href="reproducibility/l6_extension/results/independent_replay_audit.json" label="JSON · AUDIT" title="Independent extension audit" meta="Replay checks for the reported extended result" />
        </div>
        <p className="archive-note"><b>Evidence boundary.</b> Independent replay supports both the selected 54-dimensional event and the reported periodic-L6 extension. The base-run launcher did not complete checkpoint-level finalization, so the archive supports the selected event rather than claiming a finalized terminal checkpoint or universal global optimum. Structural, aerodynamic, and flight validation remain future work.</p>
      </section>

      <footer><div className="brand"><span>FW</span><b>FALCON / SYNTHESIS</b></div><p>Paper, videos, optimization evidence, code, and source data · Khalifa University · 2026</p><a href="#top">BACK TO TOP ↑</a></footer>
    </main>
  );
}
