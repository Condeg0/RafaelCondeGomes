(function () {
  function applyLang(lang) {
    document.querySelectorAll('.lang-en').forEach(function (el) {
      el.style.display = lang === 'en' ? 'inline' : 'none';
    });
    document.querySelectorAll('.lang-pt').forEach(function (el) {
      el.style.display = lang === 'pt' ? 'inline' : 'none';
    });

    // Swap resume href (match EN filename, avoid double-replacing PT)
    document.querySelectorAll('a').forEach(function (a) {
      if (!a.href) return;
      if (lang === 'pt' && a.href.match(/Resume-RafaelCondeGomes\.pdf/)) {
        a.href = a.href.replace('Resume-RafaelCondeGomes.pdf', 'Resume-RafaelCondeGomes-PT.pdf');
      } else if (lang === 'en' && a.href.match(/Resume-RafaelCondeGomes-PT\.pdf/)) {
        a.href = a.href.replace('Resume-RafaelCondeGomes-PT.pdf', 'Resume-RafaelCondeGomes.pdf');
      }
    });

    var btn = document.getElementById('lang-toggle-btn');
    if (btn) {
      btn.textContent = lang === 'en' ? 'PT' : 'EN';
      btn.setAttribute('aria-label', lang === 'en' ? 'Mudar para Português' : 'Switch to English');
    }

    localStorage.setItem('lang', lang);
  }

  function toggleLang() {
    var current = localStorage.getItem('lang') || 'en';
    applyLang(current === 'en' ? 'pt' : 'en');
  }

  function injectButton() {
    var btn = document.createElement('button');
    btn.id = 'lang-toggle-btn';
    btn.className = 'lang-toggle-btn';
    btn.textContent = 'PT';
    btn.setAttribute('aria-label', 'Mudar para Português');
    btn.addEventListener('click', toggleLang);

    var target = document.querySelector('.navbar-collapse');
    if (target) {
      target.appendChild(btn);
    } else {
      var nav = document.querySelector('.navbar .container-fluid') ||
                document.querySelector('.navbar');
      if (nav) nav.appendChild(btn);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    injectButton();
    var lang = localStorage.getItem('lang') || 'en';
    applyLang(lang);
  });
})();
