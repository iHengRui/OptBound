import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = await readFile(path.join(root, "dist/index.html"), "utf8");
const title = "Open Questions · OptBound";
const description = "下载去中心化优化各 setting 的标准 Profile，查看选题人员与待分配问题。";
const page = source
  .replace('<html lang="en"', '<html lang="zh-CN"')
  .replace(/<title>[^<]*<\/title>/, `<title>${title}</title>`)
  .replaceAll("https://optbound.reaslab.io/\"", "https://optbound.reaslab.io/topics/\"")
  .replace(/(<meta\s+(?:name="description"|property="og:description"|name="twitter:description")\s+content=")[^"]*(")/g, `$1${description}$2`)
  .replace(/(<meta\s+(?:property="og:title"|name="twitter:title")\s+content=")[^"]*(")/g, `$1${title}$2`);
await mkdir(path.join(root, "dist/topics"), { recursive: true });
await writeFile(path.join(root, "dist/topics/index.html"), page);
console.log("Built /topics/ with the existing single-file application.");
