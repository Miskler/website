(function() {
    const root = document.documentElement;
    const urlParams = new URLSearchParams(window.location.search);

    // Prioritize cookie as source of truth
    let theme = getThemeCookie();
    if (theme === null) {
        theme = urlParams.get('theme') || 'dark'; // Default to dark if neither
    }
    let isDarkMode = theme === 'dark';

    // Apply theme synchronously as early as possible to avoid FOUC
    root.classList.toggle('light', !isDarkMode);

    // Sync cookie and URL immediately
    setThemeCookie(theme);
    if (window.history && window.history.pushState) {
        updateUrl(theme);
    }

    // Function to get theme from cookie (synchronous)
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

    // Function to set theme cookie (synchronous)
    function setThemeCookie(theme) {
        document.cookie = `theme=${theme}; path=/; max-age=31536000`; // 1 year expiration
    }

    // Function to update URL (uses history.pushState)
    function updateUrl(theme) {
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('theme', theme);
        window.history.pushState({}, '', newUrl);
    }

    // The rest of the script waits for DOMContentLoaded
    document.addEventListener('DOMContentLoaded', () => {
        const toggleButton = document.getElementById('theme-toggle');

        // Set checkbox state to match (assuming checked means light mode)
        toggleButton.checked = !isDarkMode;

        // Function to apply initial theme to iframes (without animation)
        function initializeIframe(iframe) {
            if (iframe.contentDocument && iframe.contentDocument.readyState === 'complete') {
                applyInitialThemeToIframe(iframe);
            } else {
                iframe.addEventListener('load', () => applyInitialThemeToIframe(iframe));
            }
        }

        function applyInitialThemeToIframe(iframe) {
            try {
                const iframeRoot = iframe.contentDocument.documentElement;
                iframeRoot.classList.toggle('light', theme === 'light');
            } catch (e) {
                // Skip if access denied (though same-origin assumed)
            }
        }

        // Initialize all existing iframes
        const iframes = document.querySelectorAll('iframe');
        iframes.forEach(initializeIframe);

        // Observe for dynamically added iframes (optional)
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'IFRAME') {
                        initializeIframe(node);
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });

        toggleButton.addEventListener('change', async (e) => {
            const wantsLight = e.target.checked;
            const newIsDarkMode = !wantsLight;
            const newTheme = newIsDarkMode ? 'dark' : 'light';

            // Check for View Transitions API support and user motion preferences
            if (!document.startViewTransition || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                // Fallback: Toggle without animation
                root.classList.toggle('light', wantsLight);
                setThemeCookie(newTheme);
                updateUrl(newTheme);
                applyToIframes(newTheme);
                return;
            }

            // Calculate global position from button center
            const { top, left, width, height } = toggleButton.getBoundingClientRect();
            const globalX = left + width / 2;
            const globalY = top + height / 2;

            // Start View Transition for main document
            const transition = document.startViewTransition(() => {
                root.classList.toggle('light', wantsLight);
            });

            // Wait for transition to be ready, then animate the reveal in main
            await transition.ready;

            // Calculate max radius for main document
            const right = window.innerWidth - globalX;
            const bottom = window.innerHeight - globalY;
            const maxRadius = Math.hypot(Math.max(globalX, right), Math.max(globalY, bottom));

            // Animate the circular reveal in main
            root.animate(
                {
                    clipPath: [
                        `circle(0px at ${globalX}px ${globalY}px)`,
                        `circle(${maxRadius}px at ${globalX}px ${globalY}px)`
                    ]
                },
                {
                    duration: 500,
                    easing: 'ease-in-out',
                    pseudoElement: '::view-transition-new(root)'
                }
            );

            // Update cookie, URL after transition
            setThemeCookie(newTheme);
            updateUrl(newTheme);

            // Trigger theme toggle in iframes via postMessage (no animation in iframe)
            applyToIframes(newTheme);
        });

        // Function to apply theme to same-origin iframes (direct toggle, no API)
        function applyToIframes(theme) {
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                try {
                    iframe.contentWindow.postMessage({
                        action: 'toggleTheme',
                        theme: theme
                    }, '*'); // Use '*' for same-origin
                } catch (e) {
                    // Skip if access denied
                }
            });
        }
    });
})();