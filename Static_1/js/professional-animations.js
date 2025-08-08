// Neural Sonic Genesis - Professional Animation System

class ProfessionalAnimations {
    constructor() {
        this.init();
    }

    init() {
        this.setupParticleSystem();
        this.setupAdvancedScrollEffects();
        this.setupProfessionalTransitions();
        this.setupQuantumEffects();
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
            opacity: 0.6;
        `;
        document.body.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        const particles = [];

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }

        function createParticle() {
            return {
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1,
                opacity: Math.random() * 0.5 + 0.2,
                color: `hsl(${Math.random() * 60 + 200}, 70%, 60%)`
            };
        }

        function animateParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            particles.forEach((particle, index) => {
                particle.x += particle.vx;
                particle.y += particle.vy;
                
                if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
                if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;
                
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                ctx.fillStyle = particle.color;
                ctx.globalAlpha = particle.opacity;
                ctx.fill();
            });
            
            requestAnimationFrame(animateParticles);
        }

        resizeCanvas();
        for (let i = 0; i < 50; i++) {
            particles.push(createParticle());
        }
        animateParticles();

        window.addEventListener('resize', resizeCanvas);
    }

    setupAdvancedScrollEffects() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'slideInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards';
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.glass-world-class, .card-world-class').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(50px)';
            observer.observe(el);
        });
    }

    setupProfessionalTransitions() {
        document.querySelectorAll('a[href^="/"]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const href = link.getAttribute('href');
                
                document.body.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
                document.body.style.opacity = '0.8';
                document.body.style.transform = 'scale(0.98)';
                
                setTimeout(() => {
                    window.location.href = href;
                }, 500);
            });
        });
    }

    setupQuantumEffects() {
        const quantumElements = document.querySelectorAll('.gradient-text');
        
        quantumElements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                element.style.animation = 'quantumGlow 1s ease-in-out infinite alternate';
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.animation = '';
            });
        });
    }
}

// Advanced CSS Animations
const advancedStyles = document.createElement('style');
advancedStyles.textContent = `
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes quantumGlow {
        from {
            text-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
            transform: scale(1);
        }
        to {
            text-shadow: 0 0 40px rgba(251, 191, 36, 0.8);
            transform: scale(1.02);
        }
    }

    @keyframes neuralPulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.3);
        }
        50% {
            box-shadow: 0 0 60px rgba(124, 58, 237, 0.8);
        }
    }

    .neural-pulse {
        animation: neuralPulse 2s ease-in-out infinite;
    }

    .quantum-hover:hover {
        animation: quantumGlow 0.5s ease-in-out;
    }
`;
document.head.appendChild(advancedStyles);

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ProfessionalAnimations();
});

// Professional loading screen
function showLoadingScreen() {
    const loader = document.createElement('div');
    loader.id = 'professional-loader';
    loader.innerHTML = `
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 25%, #2d1b69 50%, #4c1d95 75%, #7c3aed 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            transition: opacity 0.5s ease;
        ">
            <div style="
                width: 80px;
                height: 80px;
                border: 3px solid rgba(251, 191, 36, 0.3);
                border-top: 3px solid #fbbf24;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            "></div>
            <div style="
                color: white;
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                font-family: 'Inter', sans-serif;
            ">Neural Sonic Genesis</div>
            <div style="
                color: rgba(255, 255, 255, 0.7);
                font-size: 14px;
                margin-top: 10px;
                font-family: 'Inter', sans-serif;
            ">Initializing Quantum Audio Intelligence...</div>
        </div>
    `;
    
    const spinStyle = document.createElement('style');
    spinStyle.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(spinStyle);
    document.body.appendChild(loader);
    
    setTimeout(() => {
        loader.style.opacity = '0';
        setTimeout(() => loader.remove(), 500);
    }, 2000);
}

// Show loading screen on page load
window.addEventListener('load', showLoadingScreen);