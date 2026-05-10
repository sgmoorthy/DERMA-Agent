/**
 * DERMA-Agent GitHub Pages - Main JavaScript
 * Handles interactive elements, charts, and UI functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initNavigation();
    initTutorialTabs();
    initCopyButtons();
    initSmoothScroll();
    initKaplanMeierChart();
    initScrollAnimations();
    initMobileMenu();
});

/**
 * Navigation - Active state on scroll
 */
function initNavigation() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    window.addEventListener('scroll', () => {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (window.scrollY >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

/**
 * Tutorial Tabs
 */
function initTutorialTabs() {
    const tabs = document.querySelectorAll('.tutorial-tab');
    const contents = document.querySelectorAll('.tutorial-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.getAttribute('data-tab');
            
            // Remove active from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            // Add active to clicked tab
            tab.classList.add('active');
            document.getElementById(targetId).classList.add('active');
            
            // Track analytics
            console.log('Tutorial tab clicked:', targetId);
        });
    });
}

/**
 * Copy Code Buttons
 */
function initCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeBlock = this.previousElementSibling;
            const code = codeBlock.querySelector('code').textContent;
            
            navigator.clipboard.writeText(code).then(() => {
                // Show feedback
                const originalText = this.textContent;
                this.textContent = '✓ Copied!';
                this.style.background = 'rgba(16, 185, 129, 0.3)';
                
                setTimeout(() => {
                    this.textContent = originalText;
                    this.style.background = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy:', err);
            });
        });
    });
}

/**
 * Smooth Scroll for Anchor Links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
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
}

/**
 * Kaplan-Meier Survival Chart
 */
function initKaplanMeierChart() {
    const ctx = document.getElementById('kaplanMeierChart');
    
    if (!ctx) return;
    
    // Sample survival data (simulated)
    const months = Array.from({length: 61}, (_, i) => i);
    
    // Low lymphocyte group (worse survival)
    const lowGroup = months.map(t => {
        const survival = Math.exp(-0.02 * t) * (1 + 0.1 * Math.sin(t/10));
        return survival;
    });
    
    // High lymphocyte group (better survival)
    const highGroup = months.map(t => {
        const survival = Math.exp(-0.01 * t) * (1 + 0.05 * Math.sin(t/10));
        return survival;
    });
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Low Lymphocyte Density (n=150)',
                    data: lowGroup,
                    borderColor: 'rgba(59, 130, 246, 0.8)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: 'High Lymphocyte Density (n=180)',
                    data: highGroup,
                    borderColor: 'rgba(245, 158, 11, 0.8)',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Kaplan-Meier Survival Curves by Lymphocyte Density',
                    color: '#fff',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    labels: { color: '#fff' }
                },
                annotation: {
                    annotations: {
                        pvalue: {
                            type: 'label',
                            xValue: 30,
                            yValue: 0.6,
                            content: ['Log-rank p = 0.023', 'HR = 0.67 (95% CI: 0.48-0.93)'],
                            color: '#fff',
                            font: { size: 12 }
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Months',
                        color: '#9ca3af'
                    },
                    ticks: { color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Survival Probability',
                        color: '#9ca3af'
                    },
                    ticks: { color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

/**
 * Scroll Animations
 */
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    document.querySelectorAll('.overview-card, .app-card, .result-card, .step-block').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.5s, transform 0.5s';
        observer.observe(el);
    });
}

// Add animation class styles
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);

/**
 * Mobile Menu Toggle
 */
function initMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger) {
        hamburger.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            hamburger.classList.toggle('active');
        });
    }
}

/**
 * Analytics Tracking (placeholder)
 */
function trackEvent(eventName, properties = {}) {
    // Placeholder for analytics tracking
    // Could integrate with Google Analytics, Mixpanel, etc.
    console.log('Event tracked:', eventName, properties);
}

// Track page load
trackEvent('page_load', {
    page: window.location.pathname,
    referrer: document.referrer
});

// Track GitHub button clicks
document.querySelectorAll('a[href*="github.com"]').forEach(link => {
    link.addEventListener('click', () => {
        trackEvent('github_click', { href: link.href });
    });
});

/**
 * Performance Monitoring
 */
if ('PerformanceObserver' in window) {
    const perfObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
            console.log('Performance entry:', entry.name, entry.duration);
        }
    });
    
    perfObserver.observe({ entryTypes: ['measure', 'navigation'] });
}

// Mark page load performance
performance.mark('page_load_start');
window.addEventListener('load', () => {
    performance.mark('page_load_end');
    performance.measure('page_load', 'page_load_start', 'page_load_end');
});
