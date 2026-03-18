// ============================
// Table of Contents Generator
// ============================
(function() {
  var tocNav = document.getElementById('tocNav');
  if (!tocNav) return;

  var article = document.querySelector('.article-content');
  if (!article) return;

  var headings = article.querySelectorAll('h2, h3');
  if (headings.length < 2) {
    var sidebar = document.getElementById('tocSidebar');
    if (sidebar) sidebar.style.display = 'none';
    return;
  }

  headings.forEach(function(heading, i) {
    if (!heading.id) {
      heading.id = 'heading-' + i;
    }
    var link = document.createElement('a');
    link.href = '#' + heading.id;
    link.textContent = heading.textContent;
    if (heading.tagName === 'H3') link.className = 'toc-h3';
    tocNav.appendChild(link);
  });

  // Active heading highlight on scroll
  var tocLinks = tocNav.querySelectorAll('a');
  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        tocLinks.forEach(function(l) { l.classList.remove('active'); });
        var active = tocNav.querySelector('a[href="#' + entry.target.id + '"]');
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-80px 0px -70% 0px' });

  headings.forEach(function(h) { observer.observe(h); });
})();

// ============================
// Reading Progress Bar
// ============================
(function() {
  var bar = document.getElementById('progressBar');
  if (!bar) return;

  window.addEventListener('scroll', function() {
    var scrollTop = window.scrollY;
    var docHeight = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = docHeight > 0 ? (scrollTop / docHeight * 100) + '%' : '0%';
  });
})();

// ============================
// Code Block Copy Buttons
// ============================
(function() {
  document.querySelectorAll('pre').forEach(function(pre) {
    var code = pre.querySelector('code');
    if (!code) return;

    var btn = document.createElement('button');
    btn.className = 'code-copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', function() {
      navigator.clipboard.writeText(code.textContent).then(function() {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function() {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 2000);
      });
    });
    pre.style.position = 'relative';
    pre.appendChild(btn);
  });
})();

// ============================
// Back to Top Button
// ============================
(function() {
  var btn = document.getElementById('backToTop');
  if (!btn) return;

  window.addEventListener('scroll', function() {
    btn.classList.toggle('visible', window.scrollY > 400);
  });

  btn.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
})();
