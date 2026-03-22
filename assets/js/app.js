// ===== Shared utilities =====

function stripFrontMatter(text) {
  if (text.startsWith('---')) {
    const end = text.indexOf('\n---', 3);
    if (end !== -1) return text.slice(end + 4).trimStart();
  }
  return text;
}

function extractTitle(markdown) {
  const m = markdown.match(/^#{1,2}\s+(.+)/m);
  return m ? m[1].trim() : null;
}

// ===== Navigation =====

function buildNav() {
  const nav = document.getElementById('nav-series-links');
  if (!nav || typeof SERIES === 'undefined') return;
  SERIES.forEach(s => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = `index.html#${s.id}`;
    a.textContent = s.emoji + ' ' + s.title;
    li.appendChild(a);
    nav.appendChild(li);
  });

  // Mobile toggle
  const toggle = document.getElementById('nav-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => links.classList.toggle('open'));
  }
}

// ===== Back to top + progress =====

function setupScrollFeatures() {
  const btn = document.getElementById('back-to-top');
  const bar = document.getElementById('progress-bar');

  window.addEventListener('scroll', () => {
    const scrolled = window.scrollY;
    const total = document.body.scrollHeight - window.innerHeight;

    if (btn) {
      btn.classList.toggle('visible', scrolled > 400);
    }
    if (bar && total > 0) {
      bar.style.width = Math.min(100, (scrolled / total) * 100) + '%';
    }
  }, { passive: true });

  if (btn) {
    btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
  }
}

// ===== TOC generation =====

function buildTOC(container) {
  const sidebar = document.getElementById('toc-sidebar');
  if (!sidebar) return;

  const headings = container.querySelectorAll('h2, h3');
  if (headings.length < 3) return;

  const title = document.createElement('div');
  title.className = 'toc-title';
  title.textContent = 'On this page';
  sidebar.appendChild(title);

  const ul = document.createElement('ul');

  headings.forEach((h, i) => {
    if (!h.id) h.id = 'heading-' + i;
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    if (h.tagName === 'H3') a.className = 'toc-h3';
    a.addEventListener('click', e => {
      e.preventDefault();
      h.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    li.appendChild(a);
    ul.appendChild(li);
  });

  sidebar.appendChild(ul);

  // Active heading highlight
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        sidebar.querySelectorAll('a').forEach(a => a.classList.remove('active'));
        const active = sidebar.querySelector(`a[href="#${entry.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });

  headings.forEach(h => observer.observe(h));
}

// ===== Copy buttons on code blocks =====

function addCopyButtons(container) {
  container.querySelectorAll('pre').forEach(pre => {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    pre.style.position = 'relative';
    pre.appendChild(btn);

    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.innerText : pre.innerText).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => (btn.textContent = 'Copy'), 2000);
      });
    });
  });
}

// ===== Mermaid =====

function renderMermaid() {
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ startOnLoad: false, theme: 'dark', darkMode: true });
    mermaid.run();
  }
}

// ===== Markdown rendering (used by article.html) =====

async function loadArticle() {
  const params = new URLSearchParams(location.search);
  const path = params.get('path');

  const container = document.getElementById('article-container');
  if (!container) return;

  if (!path) {
    container.innerHTML = '<div class="error-state"><h2>No article specified</h2><p>Add ?path=Series/filename.md to the URL</p></div>';
    return;
  }

  // Show loading
  container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Loading article…</p></div>';

  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw = await res.text();
    const md = stripFrontMatter(raw);

    // Extract series/part info from path
    const parts = path.split('/');
    const seriesId = parts[0];
    const fileName = parts[parts.length - 1].replace('.md', '');
    const partMatch = fileName.match(/part-(\d+)/);
    const partNum = partMatch ? parseInt(partMatch[1]) : 0;

    // Resolve series data
    let seriesData = null;
    let articleData = null;
    if (typeof getSeriesById === 'function') {
      seriesData = getSeriesById(seriesId);
      if (seriesData) articleData = seriesData.articles.find(a => a.num === partNum);
    }

    const title = (articleData && articleData.titles) ? articleData.titles : extractTitle(md) || 'Article';

    // Build breadcrumb
    const breadcrumb = document.getElementById('article-breadcrumb');
    if (breadcrumb && seriesData) {
      breadcrumb.innerHTML = `<a href="index.html">Home</a><span>›</span><a href="series.html?id=${seriesId}">${seriesData.title}</a><span>›</span>Part ${partNum}`;
    }

    // Set page title
    document.title = title + ' — DeepDive';

    // Set article heading
    const titleEl = document.getElementById('article-title');
    if (titleEl) titleEl.textContent = title;

    // Render markdown
    if (typeof marked === 'undefined') throw new Error('marked.js not loaded');

    marked.setOptions({
      breaks: false,
      gfm: true,
    });

    const html = marked.parse(md);

    // Build article body
    const body = document.getElementById('article-body');
    body.innerHTML = html;

    // Syntax highlighting
    if (typeof hljs !== 'undefined') {
      body.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
    }

    // Mermaid
    body.querySelectorAll('code.language-mermaid').forEach(code => {
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;
      code.parentElement.replaceWith(div);
    });
    renderMermaid();

    addCopyButtons(body);
    buildTOC(body);

    // Hide loading, show content
    container.style.display = 'contents';

    // Prev / Next nav
    if (typeof getNeighbors === 'function') {
      const { prev, next } = getNeighbors(seriesId, partNum);
      const navEl = document.getElementById('article-nav');
      if (navEl) {
        if (prev) {
          navEl.querySelector('.nav-prev').innerHTML = `
            <a class="nav-card" href="article.html?path=${prev.file}">
              <div class="nav-direction">← Previous</div>
              <div class="nav-title">Part ${prev.num}: ${prev.titles || ''}</div>
            </a>`;
        }
        if (next) {
          navEl.querySelector('.nav-next').innerHTML = `
            <a class="nav-card next" href="article.html?path=${next.file}">
              <div class="nav-direction">Next →</div>
              <div class="nav-title">Part ${next.num}: ${next.titles || ''}</div>
            </a>`;
        }
        navEl.style.display = 'grid';
      }
    }

  } catch (err) {
    container.innerHTML = `<div class="error-state"><h2>Failed to load article</h2><p>${err.message}</p><p><a href="index.html">← Back to home</a></p></div>`;
  }
}

// ===== Series index page =====

function loadSeriesPage() {
  const params = new URLSearchParams(location.search);
  const id = params.get('id');

  const container = document.getElementById('series-container');
  if (!container || typeof getSeriesById !== 'function') return;

  const s = getSeriesById(id);
  if (!s) {
    container.innerHTML = '<div class="error-state"><h2>Series not found</h2><p><a href="index.html">← Home</a></p></div>';
    return;
  }

  document.title = s.title + ' — DeepDive';

  container.innerHTML = `
    <div class="series-header">
      <div class="series-emoji">${s.emoji}</div>
      <h1>${s.title}</h1>
      <p>${s.description}</p>
      <div class="card-meta">
        <span class="badge badge-parts">${s.parts} parts</span>
        <span class="badge">${s.level}</span>
      </div>
    </div>
    <div class="article-list">
      ${s.articles.map(a => `
        <a class="article-item" href="article.html?path=${a.file}">
          <div class="article-num">${a.num}</div>
          <div class="article-title">Part ${a.num}: ${a.titles || a.file}</div>
        </a>`).join('')}
    </div>`;
}

// ===== Init =====

document.addEventListener('DOMContentLoaded', () => {
  buildNav();
  setupScrollFeatures();

  const page = document.body.dataset.page;
  if (page === 'article') loadArticle();
  if (page === 'series') loadSeriesPage();
});
