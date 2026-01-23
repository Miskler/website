(function() {
    const iframeRoot = document.documentElement;

    // Apply initial theme synchronously using shared cookie
    const getThemeCookie = () => {
        const cookies = document.cookie.split('; ');
        for (let cookie of cookies) {
            const [name, value] = cookie.split('=');
            if (name === 'theme') {
                return value;
            }
        }
        return null;
    };

    let theme = getThemeCookie() || 'dark';
    iframeRoot.classList.toggle('light', theme === 'light');

    // Listen for messages after DOMContentLoaded
    document.addEventListener('DOMContentLoaded', () => {
        window.addEventListener('message', (event) => {
            // Verify origin if needed; assuming same-origin, skip for simplicity
            if (event.data.action === 'toggleTheme') {
                const { theme } = event.data;
                const wantsLight = theme === 'light';
                iframeRoot.classList.toggle('light', wantsLight);
            }
        });
    });
})();