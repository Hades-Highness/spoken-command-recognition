/**
 * Copies the ONNX Runtime Web WASM artifacts into public/ort so the browser can
 * fetch them from a stable URL (ort.env.wasm.wasmPaths = "/ort/").
 *
 * Only the plain WASM execution provider is needed: the app runs single
 * threaded with proxy disabled, so the asyncify/jsep/jspi builds are unused.
 * Model weights are NOT handled here; those are release artifacts placed in
 * public/models/ manually.
 */
import { copyFile, mkdir, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const sourceDir = join(projectRoot, "node_modules", "onnxruntime-web", "dist");
const targetDir = join(projectRoot, "public", "ort");

const ARTIFACTS = ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

async function isCurrent(source, target) {
  try {
    const [sourceStats, targetStats] = await Promise.all([
      stat(source),
      stat(target),
    ]);
    return sourceStats.size === targetStats.size;
  } catch {
    return false;
  }
}

async function main() {
  await mkdir(targetDir, { recursive: true });
  let copied = 0;

  for (const name of ARTIFACTS) {
    const source = join(sourceDir, name);
    const target = join(targetDir, name);

    if (await isCurrent(source, target)) {
      continue;
    }

    await copyFile(source, target);
    copied += 1;
    console.log(`[ort] copied ${name} to public/ort`);
  }

  if (copied === 0) {
    console.log("[ort] public/ort is up to date");
  }
}

main().catch((error) => {
  console.error("[ort] failed to prepare the ONNX Runtime assets:", error);
  process.exitCode = 1;
});
