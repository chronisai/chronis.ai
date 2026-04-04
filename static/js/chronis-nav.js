/**
 * chronis-nav.js — The Chronis Ecosystem Bridge
 * Injects unified nav + footer + mobile menu + transitions + shared utilities
 * across every static page. One source of truth for the entire site.
 * v3.0 — April 2026
 */
 (function (w, d) {
  'use strict';

  /* ─────────────────────────────────────────────────────────────
     SITE MAP — single source of truth for all URLs
  ───────────────────────────────────────────────────────────── */
  const SITE = {
    home:       { url: '/',                           label: 'Home',       nav: false },
    howItWorks: { url: '/static/how-it-works.html',   label: 'How it works', nav: true },
    locket:     { url: '/static/locket.html',         label: 'The Locket', nav: true },
    compare:    { url: '/static/comparison.html',     label: 'Compare',    nav: true },
    blog:       { url: '/static/blog.html',           label: 'Resources',  nav: true },
    ethics:     { url: '/static/ethics.html',         label: 'Ethics',     nav: true },
    demo:       { url: '/demo',                       label: 'Demo',       nav: true, cta: true },
    privacy:    { url: '/static/privacy.html',        label: 'Privacy',    nav: false },
    investors:  { url: '/static/investors.html',      label: 'Investors',  nav: false },
    thankyou:   { url: '/static/thankyou.html',       label: 'Thank You',  nav: false },
    live:       { url: '/static/live.html',           label: 'Live',       nav: false, skipNav: true, skipFooter: true },
    dashboard:  { url: '/static/dashboard.html',      label: 'Dashboard',  nav: false, skipNav: true, skipFooter: true },
    test:       { url: '/static/test.html',           label: 'Test',       nav: false, skipNav: true },
  };

  /* ─────────────────────────────────────────────────────────────
     DETECT CURRENT PAGE
  ───────────────────────────────────────────────────────────── */
  const path = w.location.pathname.replace(/\/$/, '') || '/';
  let currentKey = 'home';
  for (const [key, page] of Object.entries(SITE)) {
    const u = page.url.replace(/\/$/, '') || '/';
    if (path === u || path.endsWith(u)) { currentKey = key; break; }
  }
  const currentPage = SITE[currentKey];

  /* ─────────────────────────────────────────────────────────────
     SKIP certain app / admin pages
  ───────────────────────────────────────────────────────────── */
  if (currentPage.skipNav && currentPage.skipFooter) return;

  /* ─────────────────────────────────────────────────────────────
     SHARED CSS VARIABLES (injected once, available site-wide)
     These resolve any design-token drift between pages.
  ───────────────────────────────────────────────────────────── */
  if (!d.getElementById('chronis-tokens')) {
    const style = d.createElement('style');
    style.id = 'chronis-tokens';
    style.textContent = `
      :root {
        --c-bg:    #060606;
        --c-bg2:   #0a0a0a;
        --c-sf:    rgba(255,255,255,.033);
        --c-sf2:   rgba(255,255,255,.058);
        --c-bd:    rgba(255,255,255,.076);
        --c-bd2:   rgba(255,255,255,.15);
        --c-tx:    #ede9e2;
        --c-mu:    rgba(237,233,226,.4);
        --c-ch:    rgba(237,233,226,.75);
        --c-ne:    #a78bfa;
        --c-ne2:   #a78bfa;
        --c-gr:    #4ade80;
        --c-rd:    #f87171;
        --c-fd:    'Cormorant Garamond', Georgia, serif;
        --c-fb:    'Plus Jakarta Sans', system-ui, sans-serif;
      }
      /* Nav shared styles */
      #cn-nav {
        position: fixed; top: 0; left: 0; right: 0; z-index: 500;
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 40px; height: 64px;
        background: rgba(6,6,6,.85);
        backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
        border-bottom: 1px solid var(--c-bd);
        font-family: var(--c-fb);
        transition: background .3s;
      }
      #cn-nav .cn-logo {
        font-family: var(--c-fd); font-size: 20px; font-weight: 400;
        letter-spacing: 4px; text-transform: uppercase;
        color: var(--c-ch); text-decoration: none; flex-shrink: 0;
      }
      #cn-nav .cn-links {
        display: flex; gap: 4px; align-items: center;
      }
      #cn-nav .cn-link {
        color: var(--c-mu); text-decoration: none; font-size: 13px;
        padding: 6px 13px; border-radius: 6px; transition: .2s;
        white-space: nowrap; position: relative;
      }
      #cn-nav .cn-link:hover { color: var(--c-tx); background: var(--c-sf); }
      #cn-nav .cn-link.active {
        color: var(--c-tx);
        background: rgba(167,139,250,.10);
        box-shadow: inset 0 0 0 1px rgba(167,139,250,.12);
      }
      #cn-nav .cn-link.active::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 50%;
        transform: translateX(-50%);
        width: 18px;
        height: 1.5px;
        background: var(--c-ne);
        border-radius: 999px;
        box-shadow: 0 0 10px rgba(167,139,250,.75), 0 0 18px rgba(167,139,250,.35);
      }
      #cn-nav .cn-cta {
        background: var(--c-tx); color: #000;
        font-size: 13px; font-weight: 600;
        padding: 8px 18px; border-radius: 8px;
        text-decoration: none; transition: .2s;
        white-space: nowrap; margin-left: 6px;
      }
      #cn-nav .cn-cta:hover { opacity: .87; }
      /* Mobile hamburger */
      #cn-burger {
        display: none; flex-direction: column; gap: 5px;
        width: 36px; height: 36px; align-items: center; justify-content: center;
        cursor: pointer; background: none; border: none; padding: 0;
      }
      #cn-burger span {
        display: block; width: 20px; height: 1.5px;
        background: var(--c-ch); border-radius: 2px;
        transition: .3s;
      }
      #cn-burger.open span:nth-child(1) { transform: translateY(6.5px) rotate(45deg); }
      #cn-burger.open span:nth-child(2) { opacity: 0; }
      #cn-burger.open span:nth-child(3) { transform: translateY(-6.5px) rotate(-45deg); }
      /* Mobile menu drawer */
      #cn-drawer {
        display: none; position: fixed; top: 64px; left: 0; right: 0;
        z-index: 499; background: rgba(6,6,6,.97);
        backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
        border-bottom: 1px solid var(--c-bd);
        flex-direction: column; gap: 0;
        font-family: var(--c-fb);
        transform: translateY(-8px); opacity: 0;
        transition: transform .25s ease, opacity .25s ease;
        pointer-events: none;
      }
      #cn-drawer.open {
        transform: translateY(0); opacity: 1;
        pointer-events: all;
      }
      #cn-drawer a {
        display: block; padding: 16px 28px;
        color: var(--c-mu); text-decoration: none; font-size: 15px;
        border-bottom: 1px solid var(--c-bd); transition: .15s;
      }
      #cn-drawer a:hover {
        color: var(--c-tx);
        background: rgba(167,139,250,.06);
      }
      #cn-drawer a.active {
        color: var(--c-ne);
        background: rgba(167,139,250,.08);
        box-shadow: inset 0 0 0 1px rgba(167,139,250,.10);
      }
      #cn-drawer .cn-drawer-cta {
        margin: 16px 28px 20px;
        display: block; text-align: center;
        background: var(--c-tx); color: #000;
        font-weight: 600; font-size: 14px;
        padding: 13px 20px; border-radius: 10px;
        text-decoration: none; border-bottom: none !important;
      }
      #cn-drawer .cn-drawer-cta:hover { opacity: .88; background: var(--c-tx) !important; }
      /* Footer */
      #cn-footer {
        border-top: 1px solid var(--c-bd);
        padding: 60px 40px 40px;
        font-family: var(--c-fb);
        background: var(--c-bg);
      }
      .cn-footer-grid {
        max-width: 1080px; margin: 0 auto;
        display: grid; grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 48px; margin-bottom: 48px;
      }
      .cn-footer-brand .cn-f-logo {
        font-family: var(--c-fd); font-size: 22px; font-weight: 300;
        letter-spacing: 4px; text-transform: uppercase;
        color: var(--c-ch); text-decoration: none; display: block;
        margin-bottom: 14px;
      }
      .cn-footer-brand p {
        font-size: 13px; color: var(--c-mu); line-height: 1.75;
        font-weight: 300; max-width: 240px;
      }
      .cn-f-col h4 {
        font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
        color: rgba(237,233,226,.3); font-weight: 500; margin-bottom: 16px;
      }
      .cn-f-col a {
        display: block; color: var(--c-mu); text-decoration: none;
        font-size: 13px; margin-bottom: 10px; transition: color .2s;
        font-weight: 300;
      }
      .cn-f-col a:hover { color: var(--c-tx); }
      .cn-f-col a.active { color: var(--c-ne); }
      .cn-footer-bottom {
        max-width: 1080px; margin: 0 auto;
        padding-top: 24px; border-top: 1px solid var(--c-bd);
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 12px;
      }
      .cn-footer-bottom span {
        font-size: 12px; color: rgba(237,233,226,.2);
      }
      .cn-footer-bottom a {
        font-size: 12px; color: rgba(237,233,226,.2);
        text-decoration: none; transition: color .2s;
      }
      .cn-footer-bottom a:hover { color: var(--c-mu); }
      /* Page transition */
      #cn-transition {
        position: fixed; inset: 0; z-index: 9999;
        background: var(--c-bg); pointer-events: none;
        opacity: 0; transition: opacity .22s ease;
      }
      /* Toast */
      #cn-toast {
        position: fixed; bottom: 28px; left: 50%; transform: translateX(-50%) translateY(16px);
        z-index: 9998; background: var(--c-bg2); border: 1px solid var(--c-bd2);
        color: var(--c-ch); font-family: var(--c-fb); font-size: 13px;
        padding: 12px 24px; border-radius: 100px;
        opacity: 0; transition: opacity .3s, transform .3s;
        pointer-events: none; white-space: nowrap;
      }
      #cn-toast.show {
        opacity: 1; transform: translateX(-50%) translateY(0);
      }
      @media (max-width: 900px) {
        #cn-nav { padding: 0 20px; }
        #cn-nav .cn-links { display: none; }
        #cn-burger { display: flex; }
        #cn-drawer { display: flex; }
        .cn-footer-grid { grid-template-columns: 1fr 1fr; gap: 32px; }
        .cn-footer-brand { grid-column: 1 / -1; }
        #cn-footer { padding: 40px 20px 32px; }
      }
      @media (max-width: 480px) {
        .cn-footer-grid { grid-template-columns: 1fr; }
      }
    `;
    d.head.appendChild(style);
  }

  /* ─────────────────────────────────────────────────────────────
     BUILD NAV
  ───────────────────────────────────────────────────────────── */
  if (!currentPage.skipNav) {
    const navLinks = Object.entries(SITE)
      .filter(([, p]) => p.nav && !p.cta)
      .map(([key, p]) => `<a href="${p.url}" class="cn-link${key === currentKey ? ' active' : ''}">${p.label}</a>`)
      .join('');

    const navHTML = `
      <a href="/" class="cn-logo">Chronis</a>
      <nav class="cn-links" aria-label="Main navigation">
        ${navLinks}
        <a href="/demo" class="cn-cta">Try Demo →</a>
      </nav>
      <button id="cn-burger" aria-label="Toggle menu" aria-expanded="false">
        <span></span><span></span><span></span>
      </button>
    `;

    const drawerLinks = Object.entries(SITE)
      .filter(([, p]) => p.nav && !p.cta)
      .map(([key, p]) => `<a href="${p.url}"${key === currentKey ? ' class="active"' : ''}>${p.label}</a>`)
      .join('');

    const drawerHTML = `
      ${drawerLinks}
      <a href="/#waitlist" class="cn-drawer-cta">Get Started →</a>
    `;

    // Replace or create nav
    let nav = d.querySelector('nav');
    if (!nav) {
      nav = d.createElement('nav');
      d.body.insertBefore(nav, d.body.firstChild);
    }
    nav.id = 'cn-nav';
    nav.className = '';
    nav.setAttribute('role', 'navigation');
    nav.innerHTML = navHTML;

    // Mobile drawer
    let drawer = d.getElementById('cn-drawer');
    if (!drawer) {
      drawer = d.createElement('div');
      drawer.id = 'cn-drawer';
      drawer.setAttribute('aria-hidden', 'true');
      d.body.insertBefore(drawer, nav.nextSibling);
    }
    drawer.innerHTML = drawerHTML;

    // Burger toggle
    const burger = d.getElementById('cn-burger');
    if (burger) {
      burger.addEventListener('click', () => {
        const isOpen = drawer.classList.contains('open');
        drawer.classList.toggle('open', !isOpen);
        burger.classList.toggle('open', !isOpen);
        burger.setAttribute('aria-expanded', String(!isOpen));
        drawer.setAttribute('aria-hidden', String(isOpen));
      });

      // Close on outside click
      d.addEventListener('click', (e) => {
        if (!nav.contains(e.target) && !drawer.contains(e.target)) {
          drawer.classList.remove('open');
          burger.classList.remove('open');
          burger.setAttribute('aria-expanded', 'false');
        }
      });
    }

    // Scroll-aware nav shadow
    w.addEventListener('scroll', () => {
      nav.style.background = w.scrollY > 20 ? 'rgba(6,6,6,.95)' : 'rgba(6,6,6,.85)';
    }, { passive: true });
  }

  /* ─────────────────────────────────────────────────────────────
     BUILD FOOTER
  ───────────────────────────────────────────────────────────── */
  if (!currentPage.skipFooter) {
    const isActive = (key) => key === currentKey ? ' class="active"' : '';

    const footerHTML = `
      <div class="cn-footer-grid">
        <div class="cn-footer-brand">
          <a href="/" class="cn-f-logo">Chronis</a>
          <p>World's first real-time AI twin. To remember today, for tommorow.</p>
        </div>
        <div class="cn-f-col">
          <h4>Product</h4>
          <a href="/static/how-it-works.html"${isActive('howItWorks')}>How it works</a>
          <a href="/static/locket.html"${isActive('locket')}>The Locket</a>
          <a href="/demo"${isActive('demo')}>Try Demo</a>
          <a href="/#waitlist">Get Started</a>
        </div>
        <div class="cn-f-col">
          <h4>Learn</h4>
          <a href="/static/blog.html"${isActive('blog')}>Resources</a>
          <a href="/static/comparison.html"${isActive('compare')}>Compare</a>
          <a href="/static/ethics.html"${isActive('ethics')}>Ethics</a>
        </div>
        <div class="cn-f-col">
          <h4>Company</h4>
          <a href="/static/privacy.html"${isActive('privacy')}>Privacy Policy</a>
          <a href="/static/investors.html"${isActive('investors')}>Investors</a>
          <a href="mailto:hello@chronis.in">Contact</a>
        </div>
      </div>
      <div class="cn-footer-bottom">
        <span>© 2026 Chronis · Preserving humanity, one voice at a time</span>
        <div style="display:flex;gap:20px;align-items:center">
          <a href="/static/privacy.html">Privacy</a>
          <a href="/static/ethics.html">Ethics</a>
          <span>Made in India 🇮🇳</span>
        </div>
      </div>
    `;

    let footer = d.querySelector('footer');
    if (!footer) {
      footer = d.createElement('footer');
      d.body.appendChild(footer);
    }
    footer.id = 'cn-footer';
    footer.innerHTML = footerHTML;
  }

  /* ─────────────────────────────────────────────────────────────
     PAGE TRANSITION — smooth fade between pages
  ───────────────────────────────────────────────────────────── */
  const overlay = d.createElement('div');
  overlay.id = 'cn-transition';
  d.body.appendChild(overlay);

  w.addEventListener('load', () => {
    requestAnimationFrame(() => {
      overlay.style.opacity = '0';
    });
  });

  d.addEventListener('click', (e) => {
    const a = e.target.closest('a[href]');
    if (!a) return;
    const href = a.getAttribute('href');
    if (!href || href.startsWith('#') || href.startsWith('mailto:') ||
        href.startsWith('tel:') || a.target === '_blank') return;

    const isSameOrigin = !href.startsWith('http') || href.startsWith(w.location.origin);
    if (!isSameOrigin) return;

    e.preventDefault();
    overlay.style.transition = 'opacity .18s ease';
    overlay.style.opacity = '1';
    setTimeout(() => { w.location.href = href; }, 180);
  });

  /* ─────────────────────────────────────────────────────────────
     TOAST UTILITY — window.Chronis.toast('message')
  ───────────────────────────────────────────────────────────── */
  const toastEl = d.createElement('div');
  toastEl.id = 'cn-toast';
  d.body.appendChild(toastEl);

  function toast(msg, duration = 3000) {
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    clearTimeout(toastEl._t);
    toastEl._t = setTimeout(() => toastEl.classList.remove('show'), duration);
  }

  /* ─────────────────────────────────────────────────────────────
     SCROLL REVEAL — animate elements with data-reveal attribute
     or class .rv (already used in ethics.html)
  ───────────────────────────────────────────────────────────── */
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('vs');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  function initReveal() {
    d.querySelectorAll('.rv, [data-reveal]').forEach(el => {
      if (!el.classList.contains('vs')) revealObserver.observe(el);
    });
  }

  if (d.readyState === 'loading') {
    d.addEventListener('DOMContentLoaded', initReveal);
  } else {
    initReveal();
  }

  /* ─────────────────────────────────────────────────────────────
     GLOBAL NAMESPACE — window.Chronis
  ───────────────────────────────────────────────────────────── */
  w.Chronis = {
    version: '3.0',
    currentPage: currentKey,
    site: SITE,
    toast: toast,

    navigate(url, delay = 180) {
      overlay.style.transition = 'opacity .18s ease';
      overlay.style.opacity = '1';
      setTimeout(() => { w.location.href = url; }, delay);
    },

    event(name, data = {}) {
      try {
        if (w.gtag) w.gtag('event', name, data);
        if (w._paq) w._paq.push(['trackEvent', 'Chronis', name, JSON.stringify(data)]);
      } catch (_) {}
    },

    url(key) {
      return SITE[key]?.url || '/';
    },

    isPage(key) {
      return currentKey === key;
    },

    async share(data) {
      if (navigator.share) {
        try {
          await navigator.share(data);
          return true;
        } catch (_) {}
      }
      if (data.url) {
        await navigator.clipboard.writeText(data.url).catch(() => {});
        toast('Link copied to clipboard');
      }
      return false;
    },
  };

  /* ─────────────────────────────────────────────────────────────
     BREADCRUMB STRUCTURED DATA — improves SEO
  ───────────────────────────────────────────────────────────── */
  if (currentKey !== 'home') {
    const breadcrumb = {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Chronis', item: 'https://chronis.in/' },
        { '@type': 'ListItem', position: 2, name: currentPage.label, item: `https://chronis.in${currentPage.url}` },
      ],
    };
    const s = d.createElement('script');
    s.type = 'application/ld+json';
    s.textContent = JSON.stringify(breadcrumb);
    d.head.appendChild(s);
  }

}(window, document));