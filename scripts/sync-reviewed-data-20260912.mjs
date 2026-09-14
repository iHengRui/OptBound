import { copyFile, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const researchRoot = path.resolve(siteRoot, "..", "..", "vanilla");
const researchData = path.join(researchRoot, "data");
const researchTables = path.join(researchRoot, "outputs", "tables");
const siteData = path.join(siteRoot, "data");

const sources = {
  table: path.join(researchData, "table_cells_manual_review_20260912.yaml"),
  notation: path.join(researchData, "notation_manual_review_20260912.yaml"),
  remarks: path.join(researchData, "remark_notes_manual_review_20260912.yaml"),
  locations: path.join(researchData, "statement_locations_manual_review_20260912.yaml"),
  bibliography: path.join(researchData, "bibliography_manual_review_20260912.yaml"),
  decisions: path.join(researchData, "manual_review_decisions_20260912.yaml"),
  delivery: path.join(researchTables, "decentralized_lb_ub_table_manual_review_20260912.md"),
  pdf: path.join(researchTables, "decentralized_lb_ub_table_manual_review_20260912.pdf"),
};

const destinations = {
  table: path.join(siteData, "table_cells_manual_expanded_v8.yaml"),
  notation: path.join(siteData, "notation_manual_expanded_v8.yaml"),
  remarks: path.join(siteData, "remark_notes_manual_expanded_v8.yaml"),
  locations: path.join(siteData, "statement_locations_manual_expanded_v8.yaml"),
  bibliography: path.join(siteData, "bibliography_manual_expanded_v8.yaml"),
  decisions: path.join(siteData, "manual_review_decisions_20260912.yaml"),
  delivery: path.join(siteData, "decentralized_lb_ub_table_manual_expanded_v8.md"),
  publicPdf: path.join(siteRoot, "public", "decentralized-lb-ub-table-v8.pdf"),
  rootPdf: path.join(siteRoot, "decentralized-lb-ub-table-v8.pdf"),
};

const targetRows = [11, 12, 14, 25, 29, 33, 37, 48, 55, 57, 60, 63, 66, 72];
let tableSource = await readFile(sources.table, "utf8");
tableSource = tableSource
  .replace("source:\n", "source:\n  review_edition: manual_review_20260912\n  review_decisions: data/manual_review_decisions_20260912.yaml\n")
  .replace(
    "mode: agent_reviewed_user_approved_v8_preserving_row_numbers",
    "mode: agent_reviewed_user_approved_web_v8",
  )
  .replaceAll("display: EMPTY", "display: Unknown")
  .replaceAll("status: EMPTY", "status: Unknown")
  .replace("missing_bound_label: EMPTY", "missing_bound_label: Unknown")
  .replace("  - EMPTY\n", "  - Unknown\n")
  .replaceAll("statement_locations_manual_review_20260912.yaml", "statement_locations_manual_expanded_v8.yaml")
  .replaceAll("remark_notes_manual_review_20260912.yaml", "remark_notes_manual_expanded_v8.yaml")
  .replaceAll("notation_manual_review_20260912.yaml", "notation_manual_expanded_v8.yaml")
  .replaceAll("bibliography_manual_review_20260912.yaml", "bibliography_manual_expanded_v8.yaml");
const table = YAML.parse(tableSource);
await writeFile(destinations.table, tableSource);

for (const key of ["notation", "remarks", "decisions"]) {
  await copyFile(sources[key], destinations[key]);
}

for (const key of ["locations", "bibliography"]) {
  const content = (await readFile(sources[key], "utf8"))
    .replaceAll("table_cells_manual_review_20260912.yaml", "table_cells_manual_expanded_v8.yaml");
  await writeFile(destinations[key], content);
}

let delivery = await readFile(sources.delivery, "utf8");
delivery = delivery
  .replace(
    "Every row concerns vanilla decentralized optimization with peer-to-peer gossip, synchronous communication rounds, and no communication compression. These four fixed attributes are therefore omitted from the row labels; the structured problem tags retain them for taxonomy and audit purposes.",
    "Every row in the current Decentralized Optimization release assumes peer-to-peer gossip, synchronous communication rounds, and no communication compression. These fixed attributes are omitted from the row labels; the structured problem tags retain them for taxonomy and audit purposes.",
  )
  .replaceAll("\\(EMPTY\\)", "Unknown")
  .replaceAll("`EMPTY`", "`Unknown`")
  .replace("omits the remaining 49 empty settings", "omits the remaining 49 unknown settings");
await writeFile(destinations.delivery, delivery);

await copyFile(sources.pdf, destinations.publicPdf);
await copyFile(sources.pdf, destinations.rootPdf);

const normalizedRows = table.cells
  .map((cell, index) => ({ row: index + 1, cellId: cell.cell_id }))
  .filter(({ row }) => targetRows.includes(row));
if (normalizedRows.length !== targetRows.length) {
  throw new Error(`Expected ${targetRows.length} reviewed rows, found ${normalizedRows.length}`);
}

console.log(`Synced ${normalizedRows.length} reviewed rows and the 2026-09-12 PDF into the website tree.`);
