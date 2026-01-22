document.addEventListener('DOMContentLoaded', () => {
    const toggleButton = document.getElementById('theme-toggle');
    const root = document.documentElement;
    const urlParams = new URLSearchParams(window.location.search);

    // Prioritize cookie as source of truth
    let theme = getThemeCookie();
    if (theme === null) {
        theme = urlParams.get('theme') || 'dark'; // Default to dark if neither
    }
    let isDarkMode = theme === 'dark';

    // Apply theme without animation
    root.classList.toggle('light', !isDarkMode);

    // Sync cookie and URL to match the prioritized theme
    setThemeCookie(theme);
    updateUrl(theme);

    toggleButton.addEventListener('click', async () => {
        // Check for View Transitions API support and user motion preferences
        if (!document.startViewTransition || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            // Fallback: Toggle without animation
            isDarkMode = !isDarkMode;
            root.classList.toggle('light', !isDarkMode);
            const newTheme = isDarkMode ? 'dark' : 'light';
            setThemeCookie(newTheme);
            updateUrl(newTheme);
            return;
        }

        // Start View Transition
        const transition = document.startViewTransition(() => {
            isDarkMode = !isDarkMode;
            root.classList.toggle('light', !isDarkMode);
        });

        // Wait for transition to be ready, then animate the reveal
        await transition.ready;

        // Calculate position from button center
        const { top, left, width, height } = toggleButton.getBoundingClientRect();
        const x = left + width / 2;
        const y = top + height / 2;

        // Calculate max radius to cover the entire viewport
        const right = window.innerWidth - left;
        const bottom = window.innerHeight - top;
        const maxRadius = Math.hypot(Math.max(left, right), Math.max(top, bottom));

        // Animate the circular reveal
        root.animate(
            {
                clipPath: [
                    `circle(0px at ${x}px ${y}px)`,
                    `circle(${maxRadius}px at ${x}px ${y}px)`
                ]
            },
            {
                duration: 500,
                easing: 'ease-in-out',
                pseudoElement: '::view-transition-new(root)'
            }
        );

        // Update cookie and URL after transition
        transition.finished.then(() => {
            const newTheme = isDarkMode ? 'dark' : 'light';
            setThemeCookie(newTheme);
            updateUrl(newTheme);
        });
    });

    function updateUrl(theme) {
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('theme', theme);
        window.history.pushState({}, '', newUrl);
    }

    function setThemeCookie(theme) {
        document.cookie = `theme=${theme}; path=/; max-age=31536000`; // 1 year expiration
    }

    function getThemeCookie() {
        const cookies = document.cookie.split('; ');
        for (let cookie of cookies) {
            const [name, value] = cookie.split('=');
            if (name === 'theme') {
                return value;
            }
        }
        return null;
    }
});