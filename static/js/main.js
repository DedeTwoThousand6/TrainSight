/**
 * TrainSight — main.js
 * Global JavaScript: navbar scroll effect, animations
 */

document.addEventListener('DOMContentLoaded', function () {

    // ── Navbar scroll effect ──────────────────────────────────
    const navbar = document.getElementById('mainNavbar');
    if (navbar) {
        function onScroll() {
            if (window.scrollY > 20) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        }
        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
    }

    // ── Simple fade-in on scroll for [data-aos] elements ─────
    const aosEls = document.querySelectorAll('[data-aos]');
    if (aosEls.length > 0 && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const el = entry.target;
                    const delay = parseInt(el.dataset.aosDelay || 0);
                    setTimeout(() => {
                        el.classList.add('aos-animate');
                    }, delay);
                    observer.unobserve(el);
                }
            });
        }, { threshold: 0.12 });

        aosEls.forEach(el => {
            el.classList.add('aos-init');
            observer.observe(el);
        });
    }

    // ── Number counter animation for stat cards ───────────────
    const statNums = document.querySelectorAll('.stat-num');
    if (statNums.length > 0 && 'IntersectionObserver' in window) {
        const counterObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const el = entry.target;
                    const text = el.textContent.trim();
                    const num = parseFloat(text);
                    if (!isNaN(num) && !el.dataset.animated) {
                        el.dataset.animated = '1';
                        animateCounter(el, num, text);
                    }
                    counterObserver.unobserve(el);
                }
            });
        }, { threshold: 0.5 });

        statNums.forEach(el => counterObserver.observe(el));
    }

    function animateCounter(el, target, originalText) {
        const duration = 900;
        const start = performance.now();
        const isFloat = originalText.includes('.');
        const decimals = isFloat ? (originalText.split('.')[1] || '').length : 0;

        function step(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;
            el.textContent = isFloat ? current.toFixed(decimals) : Math.round(current).toString();
            if (progress < 1) requestAnimationFrame(step);
            else el.textContent = originalText; // restore exact
        }
        requestAnimationFrame(step);
    }

});
