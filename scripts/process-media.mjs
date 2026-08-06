import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ffmpegPath from "ffmpeg-static";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const output = join(root, "public", "media", "web");
mkdirSync(output, { recursive: true });

const slots = [
  ["flight-front", "source_media/01_flight/front/source.mp4"],
  ["flight-side", "source_media/01_flight/side/source.mp4"],
  ["dlc-front", "source_media/02_dlc/front/source.mp4"],
  ["dlc-side", "source_media/02_dlc/side/source.mp4"],
  ["mechanism-motion", "source_media/03_mechanism/source.mp4"],
  ["optimized-mechanism", "source_media/05_final_result/source.mp4"],
];

function run(args) {
  const result = spawnSync(ffmpegPath, args, { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

for (const [name, relative] of slots) {
  const input = join(root, relative);
  if (!existsSync(input)) {
    console.log(`SKIP ${name}: add ${relative}`);
    continue;
  }
  const video = join(output, `${name}.mp4`);
  const poster = join(output, `${name}.jpg`);
  run(["-y", "-i", input, "-map_metadata", "-1", "-an", "-vf", "scale=1280:800:force_original_aspect_ratio=decrease,pad=1280:800:(ow-iw)/2:(oh-ih)/2:color=0x132025,fps=24,setsar=1", "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", video]);
  run(["-y", "-ss", "0.25", "-i", video, "-frames:v", "1", "-update", "1", "-q:v", "2", poster]);
  console.log(`READY ${name}`);
}
