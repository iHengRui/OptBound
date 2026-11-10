import { createIcons, Download, Moon, Search, Sun } from "lucide";
import catalog from "../data/profiles-catalog.json";
import assignments from "../data/profile-assignments.json";
import "./topics.css";

// The single-file build uses a relative base; downloads stay rooted at the site.
const base = new URL(import.meta.env.BASE_URL, window.location.origin).pathname;
const geometryLabels = { SC: "Strongly convex", C: "Convex", NC: "Nonconvex", PL: "PL condition" };
const profiles = catalog.profiles.map((profile) => {
  const tags = profile.tag.split("-");
  const people = assignments[profile.id] || [];
  const description = [
    geometryLabels[tags[1]],
    tags.includes("SM") ? "Smooth" : "Nonsmooth",
    tags.includes("TV") ? "Time-varying network" : "Static network",
    ...(tags.includes("COMP") ? ["Composite objective"] : []),
  ].join(" · ");
  return { ...profile, tags, people, description };
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[character]);
}

function profileRow(profile) {
  return `<tr id="setting-${profile.id}" data-profile="${profile.id}">
    <td class="topic-id" data-label="ID"><a href="#setting-${profile.id}" aria-label="Setting ${profile.id}">${profile.id}</a></td>
    <td data-label="Setting">
      <p class="topic-description">${escapeHtml(profile.description)}</p>
      <code class="topic-tag">${escapeHtml(profile.tag)}</code>
      <p class="topic-rows">Table rows ${profile.sourceRows.join(", ")}</p>
    </td>
    <td data-label="Standard measure">${escapeHtml(profile.measure)}</td>
    <td data-label="Assignee"><div class="topic-assignees">${profile.people.length
      ? profile.people.map((name) => `<span class="topic-person"><span class="topic-person-dot" aria-hidden="true"></span>${escapeHtml(name)}</span>`).join("")
      : '<span class="topic-unassigned">Unassigned</span>'}</div></td>
    <td data-label="Profile">${profile.file
      ? `<a class="topic-download" href="${base}profiles/${profile.file}" download="${profile.file}" aria-label="Download the Markdown profile for setting ${profile.id}"><i data-lucide="download" aria-hidden="true"></i><span>Download MD</span></a>`
      : '<span class="topic-pending">Profile pending</span>'}</td>
  </tr>`;
}

export function renderTopics(app) {
  document.body.classList.add("topics-page");
  document.documentElement.lang = "en";
  document.title = "Open Questions · OptBound";
  document.querySelector('link[rel="canonical"]').href = "https://optbound.reaslab.io/topics/";
  document.querySelector('meta[name="description"]').content = "Browse decentralized optimization settings, download available profiles, and view topic assignments.";
  const assigned = profiles.filter((profile) => profile.people.length).length;
  const availableCount = profiles.filter((profile) => profile.file).length;
  const pendingCount = profiles.length - availableCount;
  const people = [...new Set(profiles.flatMap((profile) => profile.people))].sort((a, b) => a.localeCompare(b, "zh-CN"));

  app.innerHTML = `
    <header class="topbar topics-topbar">
      <div class="topbar-inner">
        <a class="brand" href="${base}" aria-label="OptBound home"><span class="brand-mark">Opt</span><span class="brand-name">Bound</span></a>
        <nav class="main-nav" aria-label="Primary navigation">
          <a href="${base}#table">Benchmark</a>
          <a href="${base}topics/" aria-current="page">Open Questions</a>
        </nav>
        <div class="topbar-actions"><button class="icon-button" id="theme-toggle" type="button" aria-label="Switch to dark theme" title="Switch theme"><i data-lucide="sun" class="theme-icon-light"></i><i data-lucide="moon" class="theme-icon-dark"></i></button></div>
      </div>
    </header>
    <main>
      <section class="topics-hero">
        <div class="content-width">
          <p class="topics-breadcrumb"><a href="${base}">OptBound</a><span aria-hidden="true"> / </span>Decentralized Optimization</p>
          <div class="topics-heading-row">
            <div><p class="section-kicker">Profiles & assignments</p><h1>Open Questions</h1></div>
            <a class="command-button secondary" href="${base}profiles/${catalog.archive}" download><i data-lucide="download" aria-hidden="true"></i><span>Download all ${availableCount} profiles</span></a>
          </div>
          <p class="topics-intro">Download a profile and see who is working on each setting. Each profile uses one standard measure; duplicate table rows are grouped.</p>
          ${pendingCount ? `<p class="topics-availability">${availableCount} profiles available; ${pendingCount} pending.</p>` : ""}
          <div class="topics-stats" aria-label="Assignment overview">
            <div class="topic-stat"><strong>${profiles.length}</strong><span>Settings</span></div>
            <div class="topic-stat"><strong>${assigned}</strong><span>Assigned</span></div>
            <div class="topic-stat"><strong>${profiles.length - assigned}</strong><span>Unassigned</span></div>
          </div>
        </div>
      </section>
      <section class="content-width topics-panel" aria-label="Settings">
        <form class="topics-toolbar" role="search" aria-label="Filter settings">
          <label class="search-field topics-search"><i data-lucide="search" aria-hidden="true"></i><span class="sr-only">Search by ID, tag or name</span><input id="topic-search" type="search" placeholder="Search by ID, tag or name…" autocomplete="off"></label>
          <div class="topics-selects">
            <label class="topics-filter"><span>Geometry</span><select id="topic-geometry"><option value="">All</option><option value="SC">Strongly convex (SC)</option><option value="C">Convex (C)</option><option value="NC">Nonconvex (NC)</option><option value="PL">PL condition</option></select></label>
            <label class="topics-filter"><span>Network</span><select id="topic-network"><option value="">All</option><option value="STAT">Static (STAT)</option><option value="TV">Time-varying (TV)</option></select></label>
            <label class="topics-filter"><span>Assignee</span><select id="topic-person"><option value="">All</option>${people.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")}</select></label>
          </div>
          <div class="topics-status-tabs" role="group" aria-label="Assignment status">
            <button type="button" data-status="all" aria-pressed="true">All <span>${profiles.length}</span></button>
            <button type="button" data-status="unassigned" aria-pressed="false">Unassigned <span>${profiles.length - assigned}</span></button>
            <button type="button" data-status="assigned" aria-pressed="false">Assigned <span>${assigned}</span></button>
          </div>
        </form>
        <div class="topics-results-bar"><p id="topic-count" role="status" aria-live="polite"></p><button type="button" id="topic-reset">Clear filters</button></div>
        <div class="topics-table-wrap">
          <table class="topics-table">
            <caption class="sr-only">Decentralized optimization settings, standard measures, assignees and profile downloads</caption>
            <thead><tr><th scope="col">ID</th><th scope="col">Setting</th><th scope="col">Standard measure</th><th scope="col">Assignee</th><th scope="col">Profile</th></tr></thead>
            <tbody id="topics-body"></tbody>
          </table>
          <div class="topics-empty" id="topic-empty" hidden><strong>No matching settings</strong><p>Try another keyword or clear the filters to see all settings.</p><button class="command-button secondary" type="button" id="topic-empty-reset">Clear filters</button></div>
        </div>
      </section>
    </main>
    <footer class="topics-footer"><div class="content-width footer-inner"><span class="optbound-wordmark"><span class="brand-mark">Opt</span><span class="brand-name">Bound</span></span><span>Profiles updated ${catalog.updatedAt} · Markdown</span><a href="${base}#table">Back to benchmark</a></div></footer>`;

  const controls = {
    q: document.querySelector("#topic-search"),
    geometry: document.querySelector("#topic-geometry"),
    network: document.querySelector("#topic-network"),
    person: document.querySelector("#topic-person"),
  };
  const params = new URLSearchParams(window.location.search);
  for (const [key, control] of Object.entries(controls)) {
    control.value = params.get(key) || "";
    if (control instanceof HTMLSelectElement && control.selectedIndex < 0) control.value = "";
  }
  let status = ["all", "unassigned", "assigned"].includes(params.get("status")) ? params.get("status") : "all";

  function filterProfiles() {
    const terms = controls.q.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const visible = profiles.filter((profile) => {
      const text = [profile.id, profile.tag, profile.description, profile.measure, profile.file ? "" : "profile pending", ...profile.people, ...profile.sourceRows].join(" ").toLowerCase();
      return terms.every((term) => text.includes(term))
        && (!controls.geometry.value || profile.tags[1] === controls.geometry.value)
        && (!controls.network.value || profile.tags.includes(controls.network.value))
        && (!controls.person.value || profile.people.includes(controls.person.value))
        && (status === "all" || (status === "assigned" ? profile.people.length > 0 : profile.people.length === 0));
    });
    document.querySelector("#topics-body").innerHTML = visible.map(profileRow).join("");
    document.querySelector("#topic-count").textContent = `Showing ${visible.length} of ${profiles.length} settings`;
    document.querySelector("#topic-empty").hidden = visible.length > 0;
    document.querySelector(".topics-table").hidden = visible.length === 0;
    document.querySelectorAll("[data-status]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.status === status)));
    const query = new URLSearchParams();
    for (const [key, control] of Object.entries(controls)) if (control.value.trim()) query.set(key, control.value.trim());
    if (status !== "all") query.set("status", status);
    const suffix = query.size ? `?${query.toString()}` : "";
    window.history.replaceState(null, "", `${window.location.pathname}${suffix}${window.location.hash}`);
    createIcons({ icons: { Download, Moon, Search, Sun } });
  }

  function resetFilters() {
    Object.values(controls).forEach((control) => { control.value = ""; });
    status = "all";
    filterProfiles();
  }
  controls.q.addEventListener("input", filterProfiles);
  [controls.geometry, controls.network, controls.person].forEach((control) => control.addEventListener("change", filterProfiles));
  document.querySelector(".topics-toolbar").addEventListener("submit", (event) => event.preventDefault());
  document.querySelectorAll("[data-status]").forEach((button) => button.addEventListener("click", () => {
    status = button.dataset.status;
    filterProfiles();
  }));
  document.querySelector("#topic-reset").addEventListener("click", resetFilters);
  document.querySelector("#topic-empty-reset").addEventListener("click", resetFilters);

  const themeToggle = document.querySelector("#theme-toggle");
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').content = theme === "dark" ? "#161817" : "#f7f8f5";
    themeToggle.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
  applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  themeToggle.addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(theme);
    try { localStorage.setItem("optbound-theme", theme); } catch {}
  });
  filterProfiles();
  if (/^#setting-\d+$/.test(window.location.hash)) {
    requestAnimationFrame(() => document.getElementById(window.location.hash.slice(1))?.scrollIntoView({ block: "center" }));
  }
}
