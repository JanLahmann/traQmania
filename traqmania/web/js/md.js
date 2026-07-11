// Minimal GitHub-flavored markdown renderer for the bundled documentation
// (Explain -> Full documentation). Covers what our docs actually use:
// headings, paragraphs, fenced code, inline code, bold/italics, links,
// images, pipe tables, nested bullet/numbered lists, blockquotes and rules.
// No build step, no CDN — input is our own repo docs, but everything is
// HTML-escaped anyway.

const GITHUB_BASE = "https://github.com/JanLahmann/traQmania/blob/main/";

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const esc = (s) => s.replace(/[&<>"]/g, (c) => ESC[c]);

/** GitHub-style anchor slug for a heading. */
const slug = (text) =>
  text.toLowerCase().replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-");

/** Resolve a markdown href: in-doc anchors stay; other .md files become
 *  data-doc links the browser widget intercepts; other relative paths point
 *  at the GitHub repo; absolute URLs pass through (new tab). */
function linkHtml(text, href) {
  if (href.startsWith("#")) return `<a href="${esc(href)}">${text}</a>`;
  const md = href.match(/^(?:\.\/)?(?:docs\/)?([\w-]+)\.md(#[\w-]*)?$/);
  if (md) return `<a href="#" data-doc="${esc(md[1])}">${text}</a>`;
  if (/^https?:\/\//.test(href)) {
    return `<a href="${esc(href)}" target="_blank" rel="noopener">${text}</a>`;
  }
  return `<a href="${esc(GITHUB_BASE + href)}" target="_blank" rel="noopener">${text}</a>`;
}

/** Relative image sources come from docs/ (served at /docs-assets). */
function imgSrc(src) {
  if (/^https?:\/\//.test(src)) return src;
  return "/docs-assets/" + src.replace(/^(?:\.\/)?docs\//, "");
}

function inline(text) {
  let s = esc(text);
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => `\x00${codes.push(c) - 1}\x00`);
  s = s.replace(
    /!\[([^\]]*)\]\(([^)\s]+)\)/g,
    (_, alt, src) => `<img src="${esc(imgSrc(src))}" alt="${alt}" loading="lazy">`,
  );
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, txt, href) => linkHtml(txt, href));
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  s = s.replace(/\x00(\d+)\x00/g, (_, i) => `<code>${codes[i]}</code>`);
  return s;
}

const LIST_RE = /^(\s*)([-*]|\d+\.)\s+(.*)$/;

function renderList(lines, start) {
  // collect the whole list block (list lines + their indented continuations)
  const items = []; // {indent, ordered, text}
  let i = start;
  while (i < lines.length) {
    const m = lines[i].match(LIST_RE);
    if (m) {
      items.push({
        indent: m[1].length,
        ordered: /\d/.test(m[2]),
        text: m[3],
      });
      i++;
    } else if (/^\s{2,}\S/.test(lines[i]) && items.length) {
      items[items.length - 1].text += " " + lines[i].trim(); // hanging indent
      i++;
    } else {
      break;
    }
  }
  const html = [];
  const stack = []; // open tags at increasing indents
  for (const item of items) {
    while (stack.length && item.indent < stack[stack.length - 1].indent) {
      html.push(`</li></${stack.pop().tag}>`);
    }
    const top = stack[stack.length - 1];
    if (!top || item.indent > top.indent) {
      const tag = item.ordered ? "ol" : "ul";
      html.push(`<${tag}><li>${inline(item.text)}`);
      stack.push({ indent: item.indent, tag });
    } else {
      html.push(`</li><li>${inline(item.text)}`);
    }
  }
  while (stack.length) html.push(`</li></${stack.pop().tag}>`);
  return { html: html.join(""), next: i };
}

function renderTable(lines, start) {
  const rows = [];
  let i = start;
  while (i < lines.length && lines[i].trim().startsWith("|")) {
    rows.push(
      lines[i].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim()),
    );
    i++;
  }
  const [head, , ...body] = rows; // row 1 is the |---| separator
  const tr = (cells, tag) =>
    `<tr>${cells.map((c) => `<${tag}>${inline(c)}</${tag}>`).join("")}</tr>`;
  return {
    html:
      `<div class="md-table-wrap"><table><thead>${tr(head, "th")}</thead>` +
      `<tbody>${body.map((r) => tr(r, "td")).join("")}</tbody></table></div>`,
    next: i,
  };
}

/** Render markdown to an HTML string. */
export function renderMarkdown(md) {
  const lines = md.split("\n");
  const html = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*$/.test(line)) {
      i++;
    } else if (line.startsWith("```")) {
      const code = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++; // closing fence
      html.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
    } else if (/^#{1,6} /.test(line)) {
      const level = line.match(/^#+/)[0].length;
      const text = line.slice(level + 1);
      html.push(`<h${level} id="${slug(text)}">${inline(text)}</h${level}>`);
      i++;
    } else if (/^ {0,3}(-{3,}|\*{3,})\s*$/.test(line)) {
      html.push("<hr>");
      i++;
    } else if (LIST_RE.test(line)) {
      const out = renderList(lines, i);
      html.push(out.html);
      i = out.next;
    } else if (
      line.trim().startsWith("|") &&
      /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1] || "")
    ) {
      const out = renderTable(lines, i);
      html.push(out.html);
      i = out.next;
    } else if (line.startsWith("> ")) {
      const quote = [];
      while (i < lines.length && lines[i].startsWith("> ")) quote.push(lines[i++].slice(2));
      html.push(`<blockquote>${inline(quote.join(" "))}</blockquote>`);
    } else {
      const para = [];
      while (
        i < lines.length &&
        !/^\s*$/.test(lines[i]) &&
        !/^(```|#{1,6} |> )/.test(lines[i]) &&
        !LIST_RE.test(lines[i])
      ) {
        para.push(lines[i++].trim());
      }
      html.push(`<p>${inline(para.join(" "))}</p>`);
    }
  }
  return html.join("\n");
}
