import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ffmpegPath from "ffmpeg-static";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const output = join(root, "public", "media", "web");
mkdirSync(output, { recursive: true });

const slots = [
  ["flight-front", "source_media/01_flight/front/source.mp4", "0.25"],
  ["flight-rear", "source_media/01_flight/rear/source.mp4", "0.45"],
  ["flight-side", "source_media/01_flight/side/source.mp4", "3.00"],
  ["dlc-front", "source_media/02_dlc/front/source.mp4", "0.25"],
  ["dlc-side", "source_media/02_dlc/side/source.mp4", "0.70"],
  ["mechanism-motion", "source_media/03_mechanism/source.mp4", "0.25"],
  ["optimization-convergence", "source_media/04_optimization/convergence/source.mp4", "30.00"],
  ["design-variables-v2", "source_media/04_optimization/variables/source.mp4", "30.00"],
  ["optimized-mechanism", "source_media/05_final_result/source.mp4", "30.00"],
];

function run(args) {
  const result = spawnSync(ffmpegPath, args, { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

const selected = new Set(process.argv.slice(2));
for (const [name, relative, posterAt] of slots) {
  if (selected.size && !selected.has(name)) continue;
  const input = join(root, relative);
  if (!existsSync(input)) {
    console.log(`SKIP ${name}: add ${relative}`);
    continue;
  }
  const video = join(output, `${name}.mp4`);
  const poster = join(output, `${name}.jpg`);
  run(["-y", "-i", input, "-map_metadata", "-1", "-an", "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x132025,fps=30,setsar=1", "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", video]);
  run(["-y", "-ss", posterAt, "-i", video, "-frames:v", "1", "-update", "1", "-q:v", "2", poster]);
  console.log(`READY ${name}`);
}
