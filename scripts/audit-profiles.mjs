import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const catalog = JSON.parse(await readFile(path.join(root, "data/profiles-catalog.json"), "utf8"));
const assignments = JSON.parse(await readFile(path.join(root, "data/profile-assignments.json"), "utf8"));
const ids = new Set();
const tags = new Set();
const rows = new Set();
for (const profile of catalog.profiles) {
  assert.match(profile.id, /^\d{2}$/);
  assert(!ids.has(profile.id), `Duplicate profile ${profile.id}`);
  assert(!tags.has(profile.tag), `Duplicate setting ${profile.tag}`);
  assert.equal(profile.file, `ULB-profile-${profile.id}.md`);
  assert(profile.measure.length > 0);
  const content = await readFile(path.join(root, "public/profiles", profile.file));
  assert.equal(createHash("sha256").update(content).digest("hex"), profile.sha256, `Profile ${profile.id} differs from the release`);
  for (const row of profile.sourceRows) {
    assert(!rows.has(row), `Source row ${row} appears twice`);
    rows.add(row);
  }
  ids.add(profile.id);
  tags.add(profile.tag);
}
for (const [id, people] of Object.entries(assignments)) {
  assert(ids.has(id), `Assigned profile ${id} is missing`);
  assert(Array.isArray(people), `Assignments for ${id} must be an array`);
  assert.equal(new Set(people).size, people.length, `Duplicate assignee for ${id}`);
  assert(people.every((name) => typeof name === "string" && name.trim().length > 0));
}
assert.equal(rows.size, catalog.sourceRowsCovered);
assert((await stat(path.join(root, "public/profiles", catalog.archive))).size > 0);
console.log(`Profile audit passed: ${ids.size} settings, ${rows.size} source rows, ${Object.values(assignments).filter((people) => people.length).length} assigned.`);
