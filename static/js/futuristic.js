// Futuristic SaaS Interactive JavaScript
class FuturisticUI {
    constructor() {
        this.initParticles();
        this.initScrollAnimations();
        this.initNavigation();
        this.initInteractions();
        this.initTypingAnimation();
        this.initCounterAnimations();
    }

    // Initialize Particle.js background
    initParticles() {
        if (typeof particlesJS !== 'undefined') {
            particlesJS('particles-js', {
                particles: {
                    number: {
                        value: 80,
                        density: {
                            enable: true,
                            value_area: 800
                        }
                    },
                    color: {
                        value: ["#00d9ff", "#8b5cf6", "#f472b6"]
                    },
                    shape: {
                        type: "circle",
                        stroke: {
                            width: 0,
                            color: "#000000"
                        }
                    },
                    opacity: {
                        value: 0.5,
                        random: false,
                        anim: {
                            enable: false,
                            speed: 1,
                            opacity_min: 0.1,
                            sync: false
                        }
                    },
                    size: {
                        value: 3,
                        random: true,
                        anim: {
                            enable: false,
                            speed: 40,
                            size_min: 0.1,
                            sync: false
                        }
                    },
                    line_linked: {
                        enable: true,
                        distance: 150,
                        color: "#00d9ff",
                        opacity: 0.4,
                        width: 1
                    },
                    move: {
                        enable: true,
                        speed: 6,
                        direction: "none",
                        random: false,
                        straight: false,
                        out_mode: "out",
                        bounce: false,
                        attract: {
                            enable: false,
                            rotateX: 600,
                            rotateY: 1200
                        }
                    }
                },
                interactivity: {
                    detect_on: "canvas",
                    events: {
                        onhover: {
                            enable: true,
                            mode: "repulse"
                        },
                        onclick: {
                            enable: true,
                            mode: "push"
                        },
                        resize: true
                    },
                    modes: {
                        grab: {
                            distance: 400,
                            line_linked: {
                                opacity: 1
                            }
                        },
                        bubble: {
                            distance: 400,
                            size: 40,
                            duration: 2,
                            opacity: 8,
                            speed: 3
                        },
                        repulse: {
                            distance: 200,
                            duration: 0.4
                        },
                        push: {
                            particles_nb: 4
                        },
                        remove: {
                            particles_nb: 2
                        }
                    }
                },
                retina_detect: true
            });
        }
    }

    // Initialize scroll-triggered animations
    initScrollAnimations() {
        // Intersection Observer for reveal animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('active');
                }
            });
        }, observerOptions);

        // Observe all elements with reveal class
        document.querySelectorAll('.reveal').forEach(el => {
            observer.observe(el);
        });

        // Parallax effect for floating elements
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            
            document.querySelectorAll('.floating-element').forEach((el, index) => {
                const speed = (index + 1) * 0.3;
                el.style.transform = `translateY(${rate * speed}px) rotate(${scrolled * 0.05}deg)`;
            });

            // Update navbar on scroll
            const navbar = document.querySelector('.navbar-futuristic');
            if (navbar) {
                if (scrolled > 100) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
            }
        });
    }

    // Initialize navigation interactions
    initNavigation() {
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Mobile menu toggle
        const navToggle = document.querySelector('.navbar-toggler');
        const navCollapse = document.querySelector('.navbar-collapse');
        
        if (navToggle && navCollapse) {
            navToggle.addEventListener('click', () => {
                navCollapse.classList.toggle('show');
            });
        }
    }

    // Initialize interactive elements
    initInteractions() {
        // Add magnetic effect to buttons
        document.querySelectorAll('.cta-button, .btn-magnetic').forEach(button => {
            button.addEventListener('mousemove', (e) => {
                const rect = button.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                
                button.style.transform = `translate(${x * 0.3}px, ${y * 0.3}px) scale(1.05)`;
            });

            button.addEventListener('mouseleave', () => {
                button.style.transform = 'translate(0px, 0px) scale(1)';
            });
        });

        // Card hover effects
        document.querySelectorAll('.feature-card, .team-card, .step-card').forEach(card => {
            card.addEventListener('mouseenter', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                card.style.setProperty('--mouse-x', `${x}px`);
                card.style.setProperty('--mouse-y', `${y}px`);
            });
        });

        // Add glow effect on card hover
        document.querySelectorAll('.glass').forEach(element => {
            element.addEventListener('mouseenter', function() {
                this.style.boxShadow = '0 25px 45px rgba(0, 217, 255, 0.2)';
            });
            
            element.addEventListener('mouseleave', function() {
                this.style.boxShadow = '0 25px 45px rgba(0, 0, 0, 0.1)';
            });
        });
    }

    // Initialize typing animation for hero text
    initTypingAnimation() {
        const heroTitle = document.querySelector('.hero-title');
        if (heroTitle) {
            const text = heroTitle.textContent;
            heroTitle.textContent = '';
            
            let i = 0;
            const typeWriter = () => {
                if (i < text.length) {
                    heroTitle.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 100);
                } else {
                    // Add cursor blink effect
                    heroTitle.style.borderRight = '3px solid #00d9ff';
                    heroTitle.style.animation = 'blink-cursor 1s infinite';
                }
            };
            
            setTimeout(typeWriter, 1000);
        }
    }

    // Initialize counter animations
    initCounterAnimations() {
        const counters = document.querySelectorAll('.counter');
        const observerOptions = {
            threshold: 0.5
        };

        const counterObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const counter = entry.target;
                    const target = parseInt(counter.getAttribute('data-target'));
                    const increment = target / 200;
                    let current = 0;

                    const updateCounter = () => {
                        if (current < target) {
                            current += increment;
                            counter.textContent = Math.ceil(current);
                            setTimeout(updateCounter, 10);
                        } else {
                            counter.textContent = target;
                        }
                    };

                    updateCounter();
                    counterObserver.unobserve(counter);
                }
            });
        }, observerOptions);

        counters.forEach(counter => counterObserver.observe(counter));
    }

    // Utility method to create floating particles
    createFloatingParticle(x, y) {
        const particle = document.createElement('div');
        particle.className = 'floating-particle';
        particle.style.cssText = `
            position: fixed;
            width: 4px;
            height: 4px;
            background: #00d9ff;
            border-radius: 50%;
            pointer-events: none;
            z-index: 1000;
            left: ${x}px;
            top: ${y}px;
            animation: particleFloat 2s ease-out forwards;
        `;
        
        document.body.appendChild(particle);
        
        setTimeout(() => {
            particle.remove();
        }, 2000);
    }

    // Initialize cursor trail effect
    initCursorTrail() {
        let mouseX = 0, mouseY = 0;
        let trail = [];

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
            
            trail.push({x: mouseX, y: mouseY, opacity: 1});
            
            if (trail.length > 10) {
                trail.shift();
            }
        });

        const animateTrail = () => {
            trail.forEach((point, index) => {
                point.opacity *= 0.9;
                if (point.opacity < 0.1) {
                    trail.splice(index, 1);
                }
            });
            
            requestAnimationFrame(animateTrail);
        };
        
        animateTrail();
    }

    // Initialize loading states
    initLoadingStates() {
        // Add loading animation to forms
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function() {
                const submitBtn = this.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.innerHTML = '<span class="loading-dots">Processing</span>';
                    submitBtn.disabled = true;
                }
            });
        });
    }

    // GSAP animations (if GSAP is loaded)
    initGSAPAnimations() {
        if (typeof gsap !== 'undefined') {
            // Timeline for hero section
            const heroTl = gsap.timeline();
            heroTl.from('.hero-title', {duration: 1, y: 100, opacity: 0, ease: 'power3.out'})
                  .from('.hero-subtitle', {duration: 1, y: 50, opacity: 0, ease: 'power3.out'}, '-=0.5')
                  .from('.cta-button', {duration: 1, y: 30, opacity: 0, ease: 'power3.out'}, '-=0.3');

            // ScrollTrigger animations
            if (typeof ScrollTrigger !== 'undefined') {
                gsap.registerPlugin(ScrollTrigger);

                // Animate feature cards
                gsap.from('.feature-card', {
                    scrollTrigger: {
                        trigger: '.features-grid',
                        start: 'top 80%',
                    },
                    y: 50,
                    opacity: 0,
                    duration: 0.8,
                    stagger: 0.2,
                    ease: 'power3.out'
                });

                // Animate team cards
                gsap.from('.team-card', {
                    scrollTrigger: {
                        trigger: '.team-grid',
                        start: 'top 80%',
                    },
                    y: 50,
                    opacity: 0,
                    duration: 0.8,
                    stagger: 0.2,
                    ease: 'power3.out'
                });

                // Animate timeline items
                gsap.from('.timeline-item', {
                    scrollTrigger: {
                        trigger: '.timeline',
                        start: 'top 80%',
                    },
                    x: (index) => index % 2 === 0 ? -100 : 100,
                    opacity: 0,
                    duration: 0.8,
                    stagger: 0.3,
                    ease: 'power3.out'
                });
            }
        }
    }
}

// Additional CSS for animations (injected via JavaScript)
const additionalStyles = `
    @keyframes blink-cursor {
        0%, 50% { border-color: #00d9ff; }
        51%, 100% { border-color: transparent; }
    }
    
    @keyframes particleFloat {
        0% {
            transform: translateY(0) scale(1);
            opacity: 1;
        }
        100% {
            transform: translateY(-100px) scale(0);
            opacity: 0;
        }
    }
    
    .floating-particle {
        box-shadow: 0 0 10px #00d9ff;
    }
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new FuturisticUI();
    
    // Add click effect to buttons
    document.addEventListener('click', (e) => {
        if (e.target.matches('.cta-button, .btn')) {
            const ui = new FuturisticUI();
            ui.createFloatingParticle(e.clientX, e.clientY);
        }
    });
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FuturisticUI;
}