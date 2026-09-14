import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const generated = JSON.parse(await readFile(path.join(root, "src/generated-data.json"), "utf8"));
const table = YAML.parse(
  await readFile(path.join(root, "data/table_cells_manual_expanded_v8.yaml"), "utf8"),
);
const reviewAttribution = YAML.parse(
  await readFile(path.join(root, "data/review_attribution_manual_expanded_v8.yaml"), "utf8"),
);
const statementLocations = YAML.parse(
  await readFile(path.join(root, "data/statement_locations_manual_expanded_v8.yaml"), "utf8"),
);
const bibliography = YAML.parse(
  await readFile(path.join(root, "data/bibliography_manual_expanded_v8.yaml"), "utf8"),
);

const expectedStatuses = {
  EXACT: 2,
  COND: 9,
  LOG: 1,
  GAP: 4,
  "UB?": 0,
  "LB?": 48,
  Unknown: 10,
};
const expectedReviewedRows = [11, 12, 14, 25, 29, 33, 37, 48, 55, 57, 60, 63, 66, 72];
const expectedAgentReferences = [
  "ref-arxiv-2402-02356",
  "ref-arxiv-1009-0571",
  "ref-arxiv-1406-0919",
];
const expectedContributors = [
  { name: "Hengrui Zhang", affiliation: "SMS, Peking University" },
  { name: "Zhaojin Gong", affiliation: "SMS, Fudan University" },
  { name: "Benqi Liu", affiliation: "BICMR, Peking University" },
  { name: "Kun Yuan", affiliation: "CMLR, Peking University" },
];
const expectedReviewerNames = [
  "Yutong He",
  "Boao Kong",
  "Jiahao Wang",
  "Hengrui Zhang",
  "Shuchen Zhu",
  "Ming Sun",
  "Mingyu Mo",
  "Wenxuan Wu",
  "Feiyue Ye",
  "Feiming Wang",
];
const errors = [];

function check(condition, message) {
  if (!condition) errors.push(message);
}

function referenceIdsFromCell(value) {
  return [...value.matchAll(/href="#(ref-[^"]+)"/g)].map((match) => match[1]);
}

function unique(values) {
  return [...new Set(values)];
}

const reviewers = reviewAttribution.reviewers || [];
const reviewerById = new Map(reviewers.map((reviewer) => [reviewer.id, reviewer]));
const generatedReviewerById = new Map(generated.reviewers.map((reviewer) => [reviewer.id, reviewer]));
const reviewerIdByReference = new Map();
for (const reviewer of reviewers) {
  check(reviewer.reference_ids.length === 5, `Reviewer ${reviewer.id} must retain exactly five assigned papers`);
  for (const referenceId of reviewer.reference_ids) {
    check(!reviewerIdByReference.has(referenceId), `Reference ${referenceId} has duplicate reviewer assignments`);
    reviewerIdByReference.set(referenceId, reviewer.id);
  }
}

const referenceSet = new Set(generated.referenceIds);
const remarkSet = new Set(generated.remarkIds);
const agentReferenceSet = new Set(reviewAttribution.agent_reviewed_reference_ids || []);
const inactiveHumanReferenceSet = new Set(reviewAttribution.inactive_human_reviewed_reference_ids || []);
const inactiveMetadataById = new Map(
  (reviewAttribution.inactive_human_reviewed_references || [])
    .map((entry) => [entry.reference_id, entry]),
);
const agentReviewByCellId = new Map(
  (reviewAttribution.agent_reviewed_rows || []).map((entry) => [entry.cell_id, entry]),
);

check(generated.rows.length === 74, `Expected 74 rows, found ${generated.rows.length}`);
check(generated.rows.length === table.counts.rows, "Generated rows differ from YAML counts");
check(generated.meta.references === 52, `Expected 52 active cited references, found ${generated.meta.references}`);
check(generated.meta.claimSides === 80, `Expected 80 claim sides, found ${generated.meta.claimSides}`);
check(generated.meta.remarkNotes === 121, `Expected 121 remark notes, found ${generated.meta.remarkNotes}`);
check(generated.meta.notationSymbols === 116, `Expected 116 notation symbols, found ${generated.meta.notationSymbols}`);
check(generated.meta.reviewDate === "2026-09-12", "Website review date is not 2026-09-12");
check(generated.meta.reviewEdition === "manual_review_20260912", "Website review edition is stale");
check(table.presentation.missing_bound_label === "Unknown", "Website missing-bound label must remain Unknown");
check(
  table.source.notation === "data/notation_manual_expanded_v8.yaml",
  "Website table points to a non-site notation filename",
);
check(
  statementLocations.source_table === "data/table_cells_manual_expanded_v8.yaml"
    && bibliography.source_table === "data/table_cells_manual_expanded_v8.yaml",
  "Website provenance registries point to a non-site table filename",
);
check(new Set(generated.rows.map((row) => row.cellId)).size === 74, "Duplicate cell IDs detected");
check(JSON.stringify(generated.meta.statusCounts) === JSON.stringify(expectedStatuses), "Status totals changed");
check(
  table.cells.every((cell) => cell.status !== "EMPTY"
    && cell.lower_bound.display !== "EMPTY"
    && cell.upper_bound.display !== "EMPTY"),
  "Website data still contains an EMPTY status or bound display",
);

check(
  JSON.stringify(reviewAttribution.contributors) === JSON.stringify(expectedContributors)
    && JSON.stringify(generated.contributors) === JSON.stringify(expectedContributors),
  "Contributor metadata or order changed",
);
check(new Set(generated.contributors.map((contributor) => contributor.name)).size === 4, "Duplicate contributor names detected");
check(
  generated.contributors.every((contributor) => contributor.name?.trim() && contributor.affiliation?.trim()),
  "Contributor name or affiliation is missing",
);
check(generated.reviewers.length === 10, `Expected 10 reviewers, found ${generated.reviewers.length}`);
check(new Set(generated.reviewers.map((reviewer) => reviewer.id)).size === 10, "Duplicate reviewer IDs detected");
check(
  JSON.stringify(generated.reviewers.map((reviewer) => reviewer.name)) === JSON.stringify(expectedReviewerNames),
  "Reviewer English names or order changed",
);
check(reviewerIdByReference.size === 50, `Expected 50 historical human assignments, found ${reviewerIdByReference.size}`);
check(
  JSON.stringify([...agentReferenceSet]) === JSON.stringify(expectedAgentReferences),
  "Agent-reviewed reference set changed",
);
check([...agentReferenceSet].every((id) => !reviewerIdByReference.has(id)), "An agent-reviewed reference was assigned to a human reviewer");
check(
  JSON.stringify([...inactiveHumanReferenceSet]) === JSON.stringify(["ref-arxiv-2408-09775"]),
  "Inactive human-review assignment changed",
);
check(
  [...inactiveHumanReferenceSet].every((id) => reviewerIdByReference.has(id) && !referenceSet.has(id) && inactiveMetadataById.has(id)),
  "Inactive human-review assignment is not fully declared or remains in the active bibliography",
);

for (const referenceId of referenceSet) {
  const humanAssigned = reviewerIdByReference.has(referenceId) && !inactiveHumanReferenceSet.has(referenceId);
  const agentReviewed = agentReferenceSet.has(referenceId);
  check(humanAssigned !== agentReviewed, `Active reference ${referenceId} must have exactly one human/agent provenance class`);
}
for (const [referenceId] of reviewerIdByReference) {
  check(referenceSet.has(referenceId) || inactiveHumanReferenceSet.has(referenceId), `Human assignment points to undeclared inactive reference ${referenceId}`);
}
for (const referenceId of agentReferenceSet) {
  check(referenceSet.has(referenceId), `Agent-reviewed reference ${referenceId} is not cited`);
}

for (const sourceReviewer of reviewers) {
  const reviewer = generatedReviewerById.get(sourceReviewer.id);
  check(Boolean(reviewer), `Generated reviewer ${sourceReviewer.id} is missing`);
  if (!reviewer) continue;

  check(reviewer.name === sourceReviewer.name, `Reviewer name changed for ${sourceReviewer.id}`);
  check(reviewer.assignmentCount === 5, `Reviewer ${sourceReviewer.id} lost a historical assignment`);
  check(reviewer.papers.length === 5, `Reviewer ${sourceReviewer.id} does not display all five assignments`);
  check(
    JSON.stringify(reviewer.papers.map((paper) => paper.referenceId)) === JSON.stringify(sourceReviewer.reference_ids),
    `Reviewer paper order changed for ${sourceReviewer.id}`,
  );

  for (const paper of reviewer.papers) {
    check(Boolean(paper.title?.trim()), `Missing paper title for ${paper.referenceId}`);
    check(Boolean(paper.url?.trim()), `Missing paper URL for ${paper.referenceId}`);
    if (paper.active) {
      check(Number.isInteger(paper.number) && paper.number >= 1 && paper.number <= 52, `Invalid paper number for ${paper.referenceId}`);
    } else {
      check(inactiveHumanReferenceSet.has(paper.referenceId), `Unexpected inactive reviewer paper ${paper.referenceId}`);
      check(paper.number === null && paper.occurrences.length === 0, `Inactive paper ${paper.referenceId} still has active occurrences`);
    }

    const expectedOccurrences = generated.rows.flatMap((row) => {
      const lbIds = referenceIdsFromCell(row.cells[6]);
      const ubIds = referenceIdsFromCell(row.cells[7]);
      const appearsInLb = lbIds.includes(paper.referenceId);
      const appearsInUb = ubIds.includes(paper.referenceId);
      if (!appearsInLb && !appearsInUb) return [];
      return [{
        cellId: row.cellId,
        rowNumber: row.number,
        side: appearsInLb && appearsInUb ? "LB + UB" : appearsInLb ? "LB" : "UB",
        setting: row.tags.join(" "),
      }];
    });
    check(
      JSON.stringify(paper.occurrences) === JSON.stringify(expectedOccurrences),
      `Paper-to-row occurrences changed for ${paper.referenceId}`,
    );
  }

  const expectedRows = generated.rows
    .filter((row) => row.review.reviewerIds.includes(sourceReviewer.id))
    .map((row) => row.number);
  check(JSON.stringify(reviewer.rowNumbers) === JSON.stringify(expectedRows), `Reviewer row list changed for ${sourceReviewer.id}`);
  check(
    reviewer.settingCount === new Set(reviewer.papers.flatMap((paper) => paper.occurrences.map((item) => item.setting))).size,
    `Reviewer setting count changed for ${sourceReviewer.id}`,
  );
}

const configuredAgentRows = (reviewAttribution.agent_reviewed_rows || []).map((entry) => entry.row);
check(JSON.stringify(configuredAgentRows) === JSON.stringify(expectedReviewedRows), "Agent-reviewed row list changed");
check(agentReviewByCellId.size === expectedReviewedRows.length, "Duplicate agent-reviewed cell IDs detected");
for (const entry of reviewAttribution.agent_reviewed_rows || []) {
  check(generated.rows[entry.row - 1]?.cellId === entry.cell_id, `Agent-review row/cell mismatch at row ${entry.row}`);
}
check(generated.reviewMeta.agentReviewedRows === 14, `Expected 14 agent-reviewed rows, found ${generated.reviewMeta.agentReviewedRows}`);
check(generated.reviewMeta.agentOnlyRows === 1, `Expected one agent-only evidence row, found ${generated.reviewMeta.agentOnlyRows}`);
check(generated.reviewMeta.attributedRows === 63, `Expected 63 rows with named paper attribution, found ${generated.reviewMeta.attributedRows}`);
check(generated.reviewMeta.pendingCoverageRows === 10, `Expected 10 coverage-pending rows, found ${generated.reviewMeta.pendingCoverageRows}`);

const row11 = generated.rows[10];
check(
  row11.review.lbReviewerIds.includes("reviewer-10")
    && row11.review.ubReviewerIds.includes("reviewer-10")
    && !row11.review.reviewerIds.includes("reviewer-07"),
  "Row 11 does not reflect the DEAREST+ source replacement",
);
check(table.cells[13].upper_bound.display === "\\NFO=\\widetilde O(\\sqrt{LR^2/(\\rho\\eps)})", "Row 14 upper bound was not restored");
check(table.cells[32].lower_bound.display === "\\NSFO=\\Omega(\\sigma^2R^2/(N\\eps^2))", "Row 33 noise-only lower bound is missing");
check(table.cells[59].lower_bound.display === "\\NSFO=\\Omega(\\sigma^2L\\Delta/(N\\eps^2))", "Row 60 noise-only lower bound is missing");
check(table.cells[62].status === "LB?" && table.cells[65].status === "LB?", "Rows 63/66 were not admitted");
check(table.cells[71].status === "Unknown", "Row 72 should remain Unknown after the unsupported draft was removed");
const supportingSourceCount = table.cells.reduce((total, cell) => (
  total
    + (cell.lower_bound.supporting_sources || []).length
    + (cell.upper_bound.supporting_sources || []).length
), 0);
check(supportingSourceCount === 2, `Expected two supporting sources, found ${supportingSourceCount}`);

for (const row of generated.rows) {
  const markup = row.cells.join(" ");
  for (const match of markup.matchAll(/href="\#(ref-[^"]+)"/g)) {
    check(referenceSet.has(match[1]), `Row ${row.number} links to missing reference ${match[1]}`);
  }
  for (const match of markup.matchAll(/href="\#remark-([^"]+)"/g)) {
    check(remarkSet.has(match[1]), `Row ${row.number} links to missing remark ${match[1]}`);
  }

  const lbReferenceIds = referenceIdsFromCell(row.cells[6]);
  const ubReferenceIds = referenceIdsFromCell(row.cells[7]);
  const expectedLbReviewerIds = unique(lbReferenceIds.map((id) => reviewerIdByReference.get(id)).filter(Boolean));
  const expectedUbReviewerIds = unique(ubReferenceIds.map((id) => reviewerIdByReference.get(id)).filter(Boolean));
  check(JSON.stringify(row.review.lbReviewerIds) === JSON.stringify(expectedLbReviewerIds), `Row ${row.number} LB reviewer mapping changed`);
  check(JSON.stringify(row.review.ubReviewerIds) === JSON.stringify(expectedUbReviewerIds), `Row ${row.number} UB reviewer mapping changed`);

  const configuredAgentReview = agentReviewByCellId.get(row.cellId);
  const expectedLbAgentReview = Boolean(configuredAgentReview?.scope?.includes("LB"))
    || lbReferenceIds.some((id) => agentReferenceSet.has(id));
  const expectedUbAgentReview = Boolean(configuredAgentReview?.scope?.includes("UB"))
    || ubReferenceIds.some((id) => agentReferenceSet.has(id));
  const expectedAgentReview = Boolean(configuredAgentReview) || expectedLbAgentReview || expectedUbAgentReview;
  check(row.review.agentReviewed === expectedAgentReview, `Row ${row.number} agent-review provenance changed`);
  check(row.review.lbAgentReviewed === expectedLbAgentReview, `Row ${row.number} LB agent-review provenance changed`);
  check(row.review.ubAgentReviewed === expectedUbAgentReview, `Row ${row.number} UB agent-review provenance changed`);

  for (const reviewerId of row.review.reviewerIds) {
    check(reviewerById.has(reviewerId), `Row ${row.number} has unknown reviewer ${reviewerId}`);
  }
  check(
    !reviewers.some((reviewer) => row.searchText.includes(reviewer.name.toLowerCase())),
    `Row ${row.number} search text unexpectedly includes reviewer data`,
  );
  check(
    !row.searchText.includes(reviewAttribution.unassigned_coverage_label.toLowerCase()),
    `Row ${row.number} search text unexpectedly includes review-status data`,
  );

  if (row.status === "Unknown") {
    check(row.review.state === "coverage-pending", `Unknown row ${row.number} is not coverage-pending`);
    check(row.review.reviewerIds.length === 0, `Unknown row ${row.number} has an inferred reviewer`);
  } else {
    check(["assigned", "agent-reviewed"].includes(row.review.state), `Evidence row ${row.number} has no review provenance`);
    check(row.review.reviewerIds.length > 0 || row.review.agentReviewed, `Evidence row ${row.number} has no human or agent provenance`);
  }
}

if (errors.length) {
  console.error(errors.map((error) => `- ${error}`).join("\n"));
  process.exit(1);
}

console.log("Web parity audit passed: 74 rows, 80 claim sides, 121 remarks, 52 active references, 14 agent-reviewed rows.");
