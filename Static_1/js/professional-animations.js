// MIXR.ai - Professional Animation System (cleaned branding + safe navigation)

/* eslint-disable no-var */
class ProfessionalAnimations {
  constructor() {
    this.init();
  }

  init() {
    this.setupParticleSystem();
    this.setupAdvancedScrollEffects();
    this.setupProfessionalTransitions(); // only internal links
    this.setupSubtleGlow();
  }

  setupParticleSystem() {
    const canvas = document.createElement('canvas');
    canvas.id = 'particle-canvas';
    canvas.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
      opacity: 0.5;
    `;
    document.body.appendChild(canvas);

    const ctx = canvas.getContext('2d', { alpha: true });
    const particles = [];
    let raf;

    function resizeCanvas() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', () => {
      cancelAnimationFrame(raf);
      resizeCanvas();
      raf = requestAnimationFrame(animateParticles);
    });

    function createParticle() {
      return {
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        size: Math.random() * 1.8 + 0.6,
        opacity: Math.random() * 0.4 + 0.2,
        color: `hsl(${Math.random() * 60 + 210}, 70%, 60%)`
      };
    }

    for (let i = 0; i < 40; i++) particles.push(createParticle());

    function animateParticles() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.globalAlpha = p.opacity;
        ctx.fillStyle = p.color;
        ctx.fill();
      }
      raf = requestAnimationFrame(animateParticles);
    }
    raf = requestAnimationFrame(animateParticles);
  }

  setupAdvancedScrollEffects() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.animation = 'slideInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards';
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });

    document.querySelectorAll('.glass-world-class, .card-world-class').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(50px)';
      observer.observe(el);
    });
  }

  setupProfessionalTransitions() {
    const isInternal = (a) => {
      try {
        const u = new URL(a.href, location.origin);
        const sameOrigin = u.origin === location.origin;
        const isHash = u.pathname === location.pathname && u.hash;
        const isNewTab = a.target && a.target !== '_self';
        const isDownload = a.hasAttribute('download');
        const isMail = u.protocol === 'mailto:';
        return sameOrigin && !isHash && !isNewTab && !isDownload && !isMail;
      } catch { return false; }
    };

    document.addEventListener('click', (e) => {
      const link = e.target.closest && e.target.closest('a[href]');
      if (!link) return;
      if (!isInternal(link)) return;

      e.preventDefault();
      const href = link.getAttribute('href');

      // subtle shrink and fade
      document.body.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
      document.body.style.opacity = '0.9';
      document.body.style.transform = 'scale(0.995)';

      // show minimal loader text "MIXR.ai"
      showLoadingScreen({ brandOnly: true, duration: 900 });

      setTimeout(() => { window.location.href = href; }, 300);
    }, true);
  }

  setupSubtleGlow() {
    document.querySelectorAll('.gradient-text').forEach(el => {
      el.addEventListener('mouseenter', () => {
        el.style.animation = 'quantumGlow 0.9s ease-in-out infinite alternate';
      });
      el.addEventListener('mouseleave', () => {
        el.style.animation = '';
      });
    });
  }
}

// CSS Animations (same keys, no legacy names)
const advancedStyles = document.createElement('style');
advancedStyles.textContent = `
  @keyframes slideInUp {
    from { opacity: 0; transform: translateY(50px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes quantumGlow {
    from { text-shadow: 0 0 20px rgba(251,191,36,0.5); transform: scale(1); }
    to   { text-shadow: 0 0 40px rgba(251,191,36,0.85); transform: scale(1.02); }
  }
`;
document.head.appendChild(advancedStyles);

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new ProfessionalAnimations();
});

// Minimal professional loading screen (brand-only)
function showLoadingScreen(opts) {
  const settings = Object.assign({ brandOnly: true, duration: 1200 }, opts || {});
  const loader = document.createElement('div');
  loader.id = 'professional-loader';
  loader.setAttribute('aria-label', 'Loading');
  loader.innerHTML = `
    <div style="
      position: fixed; inset: 0;
      background: linear-gradient(135deg,#0f0f23 0%,#1a1a3e 25%,#2d1b69 50%,#4c1d95 75%,#7c3aed 100%);
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      z-index: 9999; transition: opacity .4s ease;">
      <div style="
        width: 80px; height: 80px; border: 3px solid rgba(251,191,36,0.3);
        border-top: 3px solid #fbbf24; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 18px;">
      </div>
      <div style="color:#fff; font-size:26px; font-weight:800; letter-spacing:.5px; font-family: 'Inter', sans-serif;">
        MIXR.ai
      </div>
      ${settings.brandOnly ? '' : `<div style="color:rgba(255,255,255,.75); font-size:14px; margin-top:8px;">Loading…</div>`}
    </div>
  `;
  const spinStyle = document.createElement('style');
  spinStyle.textContent = `
    @keyframes spin { 0%{transform:rotate(0)} 100%{transform:rotate(360deg)} }
  `;
  document.head.appendChild(spinStyle);
  document.body.appendChild(loader);

  const remove = () => {
    loader.style.opacity = '0';
    setTimeout(() => loader.remove(), 400);
  };
  // auto remove after duration (safety)
  setTimeout(remove, settings.duration);
}

// Show loader on initial page load only (brand-only)
window.addEventListener('load', () => {
  showLoadingScreen({ brandOnly: true, duration: 900 });
});
