// MasterAI - World-Class UI Effects Library

// Advanced Page Transitions
class PageTransitions {
    static init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.setupPageTransitions();
            this.setupScrollAnimations();
            this.setupParallaxEffects();
            this.setupCounterAnimations();
        });
    }

    static setupPageTransitions() {
        document.querySelectorAll('a[href^="/"]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const href = link.getAttribute('href');
                
                // Fade out effect
                document.body.style.opacity = '0.7';
                document.body.style.transform = 'scale(0.98)';
                document.body.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
                
                setTimeout(() => {
                    window.location.href = href;
                }, 300);
            });
        });
    }

    static setupScrollAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

        document.querySelectorAll('.glass, .hover-lift, .testimonial-premium').forEach(el => {
            observer.observe(el);
        });
    }

    static setupParallaxEffects() {
        let ticking = false;
        
        function updateParallax() {
            const scrolled = window.pageYOffset;
            
            document.querySelectorAll('.float, .float-premium').forEach(element => {
                const speed = element.dataset.speed || 0.5;
                element.style.transform = `translateY(${scrolled * speed}px)`;
            });
            
            ticking = false;
        }

        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(updateParallax);
                ticking = true;
            }
        });
    }

    static setupCounterAnimations() {
        const counters = document.querySelectorAll('[data-count]');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        });

        counters.forEach(counter => observer.observe(counter));
    }

    static animateCounter(element) {
        const target = parseInt(element.dataset.count);
        const duration = 2000;
        const start = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = Math.floor(progress * target);
            element.textContent = current.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        
        requestAnimationFrame(update);
    }
}

// Advanced Button Effects
class ButtonEffects {
    static init() {
        this.setupRippleEffect();
        this.setupHoverEffects();
        this.setupLoadingStates();
    }

    static setupRippleEffect() {
        document.querySelectorAll('button, .btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const ripple = document.createElement('span');
                const rect = button.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.cssText = `
                    position: absolute;
                    width: ${size}px;
                    height: ${size}px;
                    left: ${x}px;
                    top: ${y}px;
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 50%;
                    transform: scale(0);
                    animation: ripple 0.6s linear;
                    pointer-events: none;
                `;
                
                button.style.position = 'relative';
                button.style.overflow = 'hidden';
                button.appendChild(ripple);
                
                setTimeout(() => ripple.remove(), 600);
            });
        });
    }

    static setupHoverEffects() {
        document.querySelectorAll('.hover-lift').forEach(element => {
            element.addEventListener('mouseenter', () => {
                element.style.transform = 'translateY(-12px) scale(1.03)';
                element.style.boxShadow = '0 25px 50px rgba(0, 0, 0, 0.4)';
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.transform = 'translateY(0) scale(1)';
                element.style.boxShadow = '';
            });
        });
    }

    static setupLoadingStates() {
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    const originalText = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';
                    submitBtn.disabled = true;
                    
                    setTimeout(() => {
                        submitBtn.innerHTML = '<i class="fas fa-check mr-2"></i>Success!';
                        submitBtn.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                        
                        setTimeout(() => {
                            submitBtn.innerHTML = originalText;
                            submitBtn.disabled = false;
                            submitBtn.style.background = '';
                        }, 2000);
                    }, 2000);
                }
            });
        });
    }
}

// Advanced Form Enhancements
class FormEnhancements {
    static init() {
        this.setupFloatingLabels();
        this.setupValidation();
        this.setupAutoComplete();
    }

    static setupFloatingLabels() {
        document.querySelectorAll('input, textarea').forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.classList.add('focused');
            });
            
            input.addEventListener('blur', () => {
                if (!input.value) {
                    input.parentElement.classList.remove('focused');
                }
            });
        });
    }

    static setupValidation() {
        document.querySelectorAll('input[required]').forEach(input => {
            input.addEventListener('input', () => {
                if (input.checkValidity()) {
                    input.style.borderColor = '#10b981';
                    input.style.boxShadow = '0 0 0 3px rgba(16, 185, 129, 0.1)';
                } else {
                    input.style.borderColor = '#ef4444';
                    input.style.boxShadow = '0 0 0 3px rgba(239, 68, 68, 0.1)';
                }
            });
        });
    }

    static setupAutoComplete() {
        // Enhanced autocomplete for email domains
        document.querySelectorAll('input[type="email"]').forEach(input => {
            const domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'];
            
            input.addEventListener('input', () => {
                const value = input.value;
                const atIndex = value.indexOf('@');
                
                if (atIndex > 0 && atIndex < value.length - 1) {
                    const domain = value.substring(atIndex + 1);
                    const suggestion = domains.find(d => d.startsWith(domain));
                    
                    if (suggestion && suggestion !== domain) {
                        // Show suggestion UI
                        console.log(`Suggestion: ${value.substring(0, atIndex + 1)}${suggestion}`);
                    }
                }
            });
        });
    }
}

// Performance Optimizations
class PerformanceOptimizer {
    static init() {
        this.setupLazyLoading();
        this.setupImageOptimization();
        this.setupCaching();
    }

    static setupLazyLoading() {
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });

        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }

    static setupImageOptimization() {
        // Preload critical images
        const criticalImages = ['/static/images/hero-bg.jpg'];
        criticalImages.forEach(src => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = 'image';
            link.href = src;
            document.head.appendChild(link);
        });
    }

    static setupCaching() {
        // Service worker registration for caching
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').catch(console.error);
        }
    }
}

// Initialize all effects
document.addEventListener('DOMContentLoaded', () => {
    PageTransitions.init();
    ButtonEffects.init();
    FormEnhancements.init();
    PerformanceOptimizer.init();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to { transform: scale(4); opacity: 0; }
    }
    
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
    
    .focused label {
        transform: translateY(-20px) scale(0.8);
        color: #fbbf24;
    }
`;
document.head.appendChild(style);