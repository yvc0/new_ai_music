// MasterAI - Enhanced UI Effects and Interactions

// Smooth page transitions
document.addEventListener('DOMContentLoaded', function() {
    // Add loading animation to all links
    document.querySelectorAll('a[href^="/"]').forEach(link => {
        link.addEventListener('click', function(e) {
            // Don't interfere with navigation if Ctrl/Cmd is pressed or target is blank
            if (e.ctrlKey || e.metaKey || this.target === '_blank') {
                return;
            }
            
            // Only apply transition if navigating to different page
            if (this.getAttribute('href') !== window.location.pathname) {
                document.body.style.opacity = '0.8';
                document.body.style.transition = 'opacity 0.3s ease';
            }
        });
    });

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all animated elements
    document.querySelectorAll('.hover-lift, .glass').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
        observer.observe(el);
    });

    // Enhanced button interactions
    document.querySelectorAll('button, .btn').forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px) scale(1.02)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Parallax scrolling effect
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const parallaxElements = document.querySelectorAll('.float');
        
        parallaxElements.forEach(element => {
            const speed = 0.5;
            element.style.transform = `translateY(${scrolled * speed}px)`;
        });
    });

    // Dynamic typing effect for hero sections
    const typingElements = document.querySelectorAll('.typing');
    typingElements.forEach(element => {
        const text = element.textContent;
        element.textContent = '';
        
        let i = 0;
        const typeWriter = () => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, 100);
            }
        };
        
        setTimeout(typeWriter, 1000);
    });
});

// Enhanced form interactions
function enhanceForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return;

    const inputs = form.querySelectorAll('input, textarea, select');
    
    inputs.forEach(input => {
        // Add floating label effect
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });
        
        // Add validation styling
        input.addEventListener('input', function() {
            if (this.checkValidity()) {
                this.style.borderColor = '#10b981';
            } else {
                this.style.borderColor = '#ef4444';
            }
        });
    });
}

// Initialize enhanced forms
document.addEventListener('DOMContentLoaded', function() {
    enhanceForm('contact-form');
    enhanceForm('signup-form');
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Loading states for buttons
function addLoadingState(button, loadingText = 'Loading...') {
    const originalText = button.innerHTML;
    const originalClass = button.className;
    
    button.innerHTML = `<i class="fas fa-spinner fa-spin mr-2"></i>${loadingText}`;
    button.disabled = true;
    
    return () => {
        button.innerHTML = originalText;
        button.className = originalClass;
        button.disabled = false;
    };
}

// Enhanced mobile menu
function toggleMobileMenu() {
    const menu = document.getElementById('mobile-menu');
    const body = document.body;
    
    if (menu.classList.contains('hidden')) {
        menu.classList.remove('hidden');
        body.style.overflow = 'hidden';
    } else {
        menu.classList.add('hidden');
        body.style.overflow = 'auto';
    }
}

// Close mobile menu when clicking outside
document.addEventListener('click', function(e) {
    const menu = document.getElementById('mobile-menu');
    const menuButton = document.querySelector('[onclick="toggleMobileMenu()"]');
    
    if (menu && !menu.contains(e.target) && !menuButton.contains(e.target)) {
        menu.classList.add('hidden');
        document.body.style.overflow = 'auto';
    }
});

// Performance optimization
const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

// Debounced scroll handler
const handleScroll = debounce(() => {
    const scrolled = window.pageYOffset;
    const nav = document.querySelector('nav');
    
    if (scrolled > 100) {
        nav.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
    } else {
        nav.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
    }
}, 10);

window.addEventListener('scroll', handleScroll);