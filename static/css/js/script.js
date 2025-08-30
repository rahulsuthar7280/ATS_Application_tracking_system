// script.js

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. Hamburger Menu and Navbar Toggle ---
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');
    const navActions = document.querySelector('.nav-actions');

    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
        navActions.classList.toggle('active');
        document.body.classList.toggle('menu-open'); // To prevent background scrolling
    });

    // Close menu when a link is clicked
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
            navActions.classList.remove('active');
            document.body.classList.remove('menu-open');
        });
    });

    // --- 2. Theme Switcher (Light/Dark Mode) ---
    const themeToggleButton = document.getElementById('theme-toggle-btn');
    const body = document.body;

    // Check for user's preferred theme in local storage
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme) {
        body.classList.add(storedTheme);
    }

    // Set the initial icon based on the current theme
    function updateThemeIcon() {
        if (body.classList.contains('dark-mode')) {
            themeToggleButton.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            themeToggleButton.innerHTML = '<i class="fas fa-moon"></i>';
        }
    }
    updateThemeIcon();

    themeToggleButton.addEventListener('click', () => {
        if (body.classList.contains('dark-mode')) {
            body.classList.remove('dark-mode');
            body.classList.add('light-mode');
            localStorage.setItem('theme', 'light-mode');
        } else {
            body.classList.remove('light-mode');
            body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark-mode');
        }
        updateThemeIcon();
    });

    // --- 3. Counter Animation for Hero Section ---
    const counters = document.querySelectorAll('.counter');
    const heroSection = document.querySelector('.hero-section');

    const observerOptions = {
        root: null,
        threshold: 0.5 // Trigger when 50% of the element is visible
    };

    const counterObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                counters.forEach(counter => {
                    const target = parseInt(counter.textContent);
                    let count = 0;
                    const increment = target / 100; // Adjust for smoother animation

                    const updateCounter = () => {
                        count += increment;
                        if (count < target) {
                            counter.textContent = Math.ceil(count);
                            requestAnimationFrame(updateCounter);
                        } else {
                            counter.textContent = target;
                        }
                    };

                    updateCounter();
                });
                observer.unobserve(entry.target); // Stop observing after animation
            }
        });
    }, observerOptions);

    if (heroSection) {
        counterObserver.observe(heroSection);
    }


    // --- 4. Scroll Animations ---
    const animateOnScrollElements = document.querySelectorAll('.animate-on-scroll');

    const scrollObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                observer.unobserve(entry.target);
            }
        });
    }, {
        root: null,
        threshold: 0.1 // Trigger when 10% of the element is visible
    });

    animateOnScrollElements.forEach(el => {
        scrollObserver.observe(el);
    });

    // --- 5. Chart.js Integration ---
    // Bar Chart for Performance Metrics
    const performanceCtx = document.getElementById('performanceChart');
    if (performanceCtx) {
        new Chart(performanceCtx, {
            type: 'bar',
            data: {
                labels: ['Q1', 'Q2', 'Q3', 'Q4'],
                datasets: [{
                    label: 'New Hires',
                    data: [12, 19, 15, 22],
                    backgroundColor: [
                        'rgba(106, 17, 203, 0.7)',
                        'rgba(37, 117, 252, 0.7)',
                        'rgba(106, 17, 203, 0.7)',
                        'rgba(37, 117, 252, 0.7)'
                    ],
                    borderColor: [
                        'rgba(106, 17, 203, 1)',
                        'rgba(37, 117, 252, 1)',
                        'rgba(106, 17, 203, 1)',
                        'rgba(37, 117, 252, 1)'
                    ],
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Pie Chart for Candidate Source Breakdown
    const sourceCtx = document.getElementById('sourceChart');
    if (sourceCtx) {
        new Chart(sourceCtx, {
            type: 'pie',
            data: {
                labels: ['LinkedIn', 'Company Website', 'Referrals', 'Job Boards'],
                datasets: [{
                    label: 'Candidate Sources',
                    data: [300, 150, 100, 150],
                    backgroundColor: [
                        'rgba(106, 17, 203, 0.8)',
                        'rgba(37, 117, 252, 0.8)',
                        'rgba(255, 159, 64, 0.8)',
                        'rgba(75, 192, 192, 0.8)'
                    ],
                    borderColor: [
                        'rgba(106, 17, 203, 1)',
                        'rgba(37, 117, 252, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
});