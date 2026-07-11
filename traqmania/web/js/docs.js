// Documentation browser (Explain -> Documentation): fetches the repo's
// markdown docs from /api/docs and renders them client-side with md.js.
// Outside a source checkout the API reports no docs and the widget points
// at GitHub instead.

import { renderMarkdown } from "./md.js";

let docList = null; // fetched once per page load

export async function initDocs(root) {
  root.innerHTML = '<p class="doc-note">loading documentation…</p>';
  if (docList === null) {
    try {
      const res = await fetch("/api/docs");
      docList = res.ok ? (await res.json()).docs : [];
    } catch {
      docList = [];
    }
  }
  if (!docList.length) {
    root.innerHTML =
      '<p class="doc-note">The full documentation ships with the source ' +
      'checkout — browse it on <a href="https://github.com/JanLahmann/traQmania" ' +
      'target="_blank" rel="noopener">GitHub</a>.</p>';
    return;
  }

  const picker = document.createElement("select");
  picker.className = "doc-picker";
  picker.setAttribute("aria-label", "Document");
  for (const doc of docList) {
    const opt = document.createElement("option");
    opt.value = doc.id;
    opt.textContent = doc.title;
    picker.append(opt);
  }
  const content = document.createElement("article");
  content.className = "md";

  async function show(id) {
    content.innerHTML = '<p class="doc-note">loading…</p>';
    try {
      const res = await fetch(`/api/docs/${encodeURIComponent(id)}`);
      const doc = await res.json();
      content.innerHTML = renderMarkdown(doc.markdown);
    } catch {
      content.innerHTML = '<p class="doc-note">failed to load this document.</p>';
    }
    if (picker.value !== id) picker.value = id;
    content.scrollTop = 0;
  }

  picker.addEventListener("change", () => show(picker.value));
  content.addEventListener("click", (ev) => {
    const a = ev.target.closest("a");
    if (!a) return;
    if (a.dataset.doc) {
      ev.preventDefault();
      if (docList.some((d) => d.id === a.dataset.doc)) show(a.dataset.doc);
      else window.open(`https://github.com/JanLahmann/traQmania/blob/main/docs/${a.dataset.doc}.md`, "_blank");
    } else if (a.getAttribute("href")?.startsWith("#")) {
      ev.preventDefault(); // keep in-page anchors inside the scrolling panel
      const el = content.querySelector(`[id="${CSS.escape(a.getAttribute("href").slice(1))}"]`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });

  root.replaceChildren(picker, content);
  show(docList[0].id); // server lists SCIENCE (the QML story) first
}
