import { createIcons, Download, Moon, Search, Sun } from "lucide";
import catalog from "../data/profiles-catalog.json";
import assignments from "../data/profile-assignments.json";
import "./topics.css";

// The single-file build uses a relative base; downloads stay rooted at the site.
const base = new URL(import.meta.env.BASE_URL, window.location.origin).pathname;
const geometryLabels = { SC: "强凸", C: "凸", NC: "非凸", PL: "PL 条件" };
const profiles = catalog.profiles.map((profile) => {
  const tags = profile.tag.split("-");
  const people = assignments[profile.id] || [];
  const description = [
    geometryLabels[tags[1]],
    tags.includes("SM") ? "光滑" : "非光滑",
    tags.includes("TV") ? "时变网络" : "静态网络",
    ...(tags.includes("COMP") ? ["复合目标"] : []),
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
    <td class="topic-id" data-label="编号"><a href="#setting-${profile.id}" aria-label="Setting ${profile.id}">${profile.id}</a></td>
    <td data-label="Setting">
      <p class="topic-description">${escapeHtml(profile.description)}</p>
      <code class="topic-tag">${escapeHtml(profile.tag)}</code>
      <p class="topic-rows">原表行号 ${profile.sourceRows.join("、")}</p>
    </td>
    <td data-label="标准 measure">${escapeHtml(profile.measure)}</td>
    <td data-label="选题人员"><div class="topic-assignees">${profile.people.length
      ? profile.people.map((name) => `<span class="topic-person"><span class="topic-person-dot" aria-hidden="true"></span>${escapeHtml(name)}</span>`).join("")
      : '<span class="topic-unassigned">待分配</span>'}</div></td>
    <td data-label="Profile"><a class="topic-download" href="${base}profiles/${profile.file}" download="${profile.file}" aria-label="下载 Setting ${profile.id} 的 Markdown Profile"><i data-lucide="download" aria-hidden="true"></i><span>下载 MD</span></a></td>
  </tr>`;
}

export function renderTopics(app) {
  document.body.classList.add("topics-page");
  document.documentElement.lang = "zh-CN";
  document.title = "Open Questions · OptBound";
  document.querySelector('link[rel="canonical"]').href = "https://optbound.reaslab.io/topics/";
  document.querySelector('meta[name="description"]').content = "下载去中心化优化各 setting 的标准 Profile，查看选题人员与待分配问题。";
  const assigned = profiles.filter((profile) => profile.people.length).length;
  const people = [...new Set(profiles.flatMap((profile) => profile.people))].sort((a, b) => a.localeCompare(b, "zh-CN"));

  app.innerHTML = `
    <header class="topbar topics-topbar">
      <div class="topbar-inner">
        <a class="brand" href="${base}" aria-label="OptBound 首页"><span class="brand-mark">Opt</span><span class="brand-name">Bound</span></a>
        <nav class="main-nav" aria-label="主导航">
          <a href="${base}#table">Benchmark</a>
          <a href="${base}topics/" aria-current="page">Open Questions</a>
        </nav>
        <div class="topbar-actions"><button class="icon-button" id="theme-toggle" type="button" aria-label="切换到深色模式" title="切换主题"><i data-lucide="sun" class="theme-icon-light"></i><i data-lucide="moon" class="theme-icon-dark"></i></button></div>
      </div>
    </header>
    <main>
      <section class="topics-hero">
        <div class="content-width">
          <p class="topics-breadcrumb"><a href="${base}">OptBound</a><span aria-hidden="true"> / </span>Decentralized Optimization</p>
          <div class="topics-heading-row">
            <div><p class="section-kicker">选题与 Profile</p><h1>Open Questions</h1></div>
            <a class="command-button secondary" href="${base}profiles/${catalog.archive}" download><i data-lucide="download" aria-hidden="true"></i><span>下载全部 ${profiles.length} 份</span></a>
          </div>
          <p class="topics-intro">查看各 setting 的标准 Profile 与选题人员。每个 setting 保留一个标准 measure，相同 setting 的原表行已合并。</p>
          <div class="topics-stats" aria-label="选题概况">
            <div class="topic-stat"><strong>${profiles.length}</strong><span>Settings</span></div>
            <div class="topic-stat"><strong>${assigned}</strong><span>已分配</span></div>
            <div class="topic-stat"><strong>${profiles.length - assigned}</strong><span>待分配</span></div>
          </div>
        </div>
      </section>
      <section class="content-width topics-panel" aria-label="Setting 列表">
        <form class="topics-toolbar" role="search" aria-label="筛选 Setting">
          <label class="search-field topics-search"><i data-lucide="search" aria-hidden="true"></i><span class="sr-only">搜索编号、标签或人员</span><input id="topic-search" type="search" placeholder="搜索编号、标签或人员…" autocomplete="off"></label>
          <div class="topics-selects">
            <label class="topics-filter"><span>几何条件</span><select id="topic-geometry"><option value="">全部条件</option><option value="SC">强凸 SC</option><option value="C">凸 C</option><option value="NC">非凸 NC</option><option value="PL">PL 条件</option></select></label>
            <label class="topics-filter"><span>通信网络</span><select id="topic-network"><option value="">全部网络</option><option value="STAT">静态 STAT</option><option value="TV">时变 TV</option></select></label>
            <label class="topics-filter"><span>选题人员</span><select id="topic-person"><option value="">全部人员</option>${people.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")}</select></label>
          </div>
          <div class="topics-status-tabs" role="group" aria-label="分配状态">
            <button type="button" data-status="all" aria-pressed="true">全部 <span>${profiles.length}</span></button>
            <button type="button" data-status="unassigned" aria-pressed="false">待分配 <span>${profiles.length - assigned}</span></button>
            <button type="button" data-status="assigned" aria-pressed="false">已分配 <span>${assigned}</span></button>
          </div>
        </form>
        <div class="topics-results-bar"><p id="topic-count" role="status" aria-live="polite"></p><button type="button" id="topic-reset">清除筛选</button></div>
        <div class="topics-table-wrap">
          <table class="topics-table">
            <caption class="sr-only">去中心化优化 Setting、标准 measure、选题人员与 Profile 下载</caption>
            <thead><tr><th scope="col">编号</th><th scope="col">Setting</th><th scope="col">标准 measure</th><th scope="col">选题人员</th><th scope="col">Profile</th></tr></thead>
            <tbody id="topics-body"></tbody>
          </table>
          <div class="topics-empty" id="topic-empty" hidden><strong>没有匹配的 setting</strong><p>试试其他关键词，或清除筛选查看全部问题。</p><button class="command-button secondary" type="button" id="topic-empty-reset">清除筛选</button></div>
        </div>
      </section>
    </main>
    <footer class="topics-footer"><div class="content-width footer-inner"><span class="optbound-wordmark"><span class="brand-mark">Opt</span><span class="brand-name">Bound</span></span><span>Profile 更新于 ${catalog.updatedAt.replaceAll("-", ".")} · Markdown 格式</span><a href="${base}#table">返回 Benchmark</a></div></footer>`;

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
      const text = [profile.id, profile.tag, profile.description, profile.measure, ...profile.people, ...profile.sourceRows].join(" ").toLowerCase();
      return terms.every((term) => text.includes(term))
        && (!controls.geometry.value || profile.tags[1] === controls.geometry.value)
        && (!controls.network.value || profile.tags.includes(controls.network.value))
        && (!controls.person.value || profile.people.includes(controls.person.value))
        && (status === "all" || (status === "assigned" ? profile.people.length > 0 : profile.people.length === 0));
    });
    document.querySelector("#topics-body").innerHTML = visible.map(profileRow).join("");
    document.querySelector("#topic-count").textContent = `显示 ${visible.length} / ${profiles.length} 个 setting`;
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
    themeToggle.setAttribute("aria-label", theme === "dark" ? "切换到浅色模式" : "切换到深色模式");
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
