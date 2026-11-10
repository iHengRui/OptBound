# OptBound

**OptBound: A Unified Benchmark of Lower and Upper Complexity Bounds in Optimization**

OptBound is a static, paper-backed benchmark for comparing lower and upper complexity bounds across optimization settings. It is organized around Distributed Optimization (Decentralized and Centralized Optimization) and Stochastic Optimization (Minimization, Minimax, and Bilevel). The current table covers Decentralized Optimization.

Current data review: `manual_review_20260912` (September 12, 2026). Run `npm run sync:reviewed-data` to refresh the website data and PDF from the adjacent research workspace, then run `npm run audit && npm run build`.

Live site: https://optbound.reaslab.io/

## Open Questions

The selection page lives at `/topics/`. It displays 43 setting profiles from the November 10, 2026 release, with one standard measure per setting. The existing 12 assignments are retained; other settings display “Unassigned”.

- Update names in `data/profile-assignments.json`; each profile ID maps to an array of assignee names.
- `data/profiles-catalog.json` records setting tags, original table rows, measures, and profile checksums.
- Downloadable Markdown files and the complete ZIP live in `public/profiles/`. They are release snapshots copied without editing the profile text.
- `npm run build` verifies the profile files, builds the existing app, and emits `/topics/index.html` for GitHub Pages. The page is a display of maintained assignments, not a self-service signup form.
