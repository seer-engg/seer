"""Browser stealth configuration for Browserless connections.

Provides user agent strings, BrowserProfile kwargs, and stealth JavaScript
scripts for remote browser sessions connected via CDP to a Browserless
service. The stealth scripts mask common automation detection vectors
such as navigator.webdriver, missing chrome.runtime, and WebGL fingerprints.
"""
from __future__ import annotations

import platform
from typing import Any, Dict

# Realistic Chrome user agents — keep aligned with current stable Chrome
# Last updated: 2026-04 (Chrome 134)
CHROME_USER_AGENTS: Dict[str, str] = {
    "linux": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "darwin": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
}


def get_platform_user_agent() -> str:
    """Get a realistic user agent for the current platform.

    Returns:
        User agent string appropriate for the current OS.
    """
    system = platform.system().lower()
    return CHROME_USER_AGENTS.get(system, CHROME_USER_AGENTS["linux"])


def get_remote_profile_kwargs() -> Dict[str, Any]:
    """Get BrowserUseProfile kwargs for remote Browserless connections.

    When connecting to a remote Browserless service via CDP, Chrome
    launch args are managed by Browserless itself. Only user_agent
    and viewport are forwarded via BrowserProfile.

    Returns:
        Dict of kwargs to pass to BrowserUseProfile constructor.
    """
    return {
        "user_agent": get_platform_user_agent(),
        "viewport": {"width": 1280, "height": 800},
    }


# ---------------------------------------------------------------------------
# Stealth JavaScript scripts injected via CDP Page.addScriptToEvaluateOnNewDocument
# These run before any page JS and mask common automation fingerprints.
# ---------------------------------------------------------------------------

STEALTH_SCRIPTS: list[str] = [
    # 1. navigator.webdriver — delete then redefine as undefined
    #    Chrome sets this at the C++ level; simple defineProperty may not stick.
    #    We delete first, then redefine on the prototype chain.
    """
    (function() {
        // Delete the property from Navigator.prototype if it exists
        if ('webdriver' in navigator) {
            try { delete Object.getPrototypeOf(navigator).webdriver; } catch(e) {}
        }
        // Redefine on Navigator.prototype to return undefined
        Object.defineProperty(Object.getPrototypeOf(navigator), 'webdriver', {
            get: () => undefined,
            configurable: true,
        });
    })();
    """,

    # 2. chrome.runtime stub — real Chrome exposes this; headless/automated doesn't
    """
    if (!window.chrome) { window.chrome = {}; }
    if (!window.chrome.runtime) {
        window.chrome.runtime = {
            connect: function() {},
            sendMessage: function() {},
            onMessage: { addListener: function() {}, removeListener: function() {} },
            id: undefined,
        };
    }
    """,

    # 3. navigator.plugins — must pass `instanceof PluginArray` check
    #    We build a proper PluginArray by modifying the existing one's prototype chain.
    """
    (function() {
        const makeMimeType = (type, suffixes, description, plugin) => {
            const mt = Object.create(MimeType.prototype);
            Object.defineProperties(mt, {
                type: { get: () => type },
                suffixes: { get: () => suffixes },
                description: { get: () => description },
                enabledPlugin: { get: () => plugin },
            });
            return mt;
        };
        const makePlugin = (name, description, filename, mimeTypes) => {
            const p = Object.create(Plugin.prototype);
            const mts = mimeTypes.map(m => makeMimeType(m.type, m.suffixes, m.description, p));
            Object.defineProperties(p, {
                name: { get: () => name },
                description: { get: () => description },
                filename: { get: () => filename },
                length: { get: () => mts.length },
            });
            mts.forEach((mt, i) => { Object.defineProperty(p, i, { get: () => mt }); });
            p.item = (i) => mts[i] || null;
            p.namedItem = (n) => mts.find(m => m.type === n) || null;
            return p;
        };
        const pluginData = [
            makePlugin('Chrome PDF Plugin', 'Portable Document Format', 'internal-pdf-viewer',
                [{type: 'application/x-google-chrome-pdf', suffixes: 'pdf', description: 'Portable Document Format'}]),
            makePlugin('Chrome PDF Viewer', '', 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                [{type: 'application/pdf', suffixes: 'pdf', description: ''}]),
            makePlugin('Chromium PDF Plugin', 'Portable Document Format', 'internal-pdf-viewer',
                [{type: 'application/x-google-chrome-pdf', suffixes: 'pdf', description: 'Portable Document Format'}]),
            makePlugin('Chromium PDF Viewer', '', 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                [{type: 'application/pdf', suffixes: 'pdf', description: ''}]),
            makePlugin('Native Client', '', 'internal-nacl-plugin', []),
        ];
        const fakePlugins = Object.create(PluginArray.prototype);
        pluginData.forEach((p, i) => { Object.defineProperty(fakePlugins, i, { get: () => p }); });
        Object.defineProperties(fakePlugins, { length: { get: () => 5 } });
        fakePlugins.item = (i) => pluginData[i] || null;
        fakePlugins.namedItem = (name) => pluginData.find(p => p.name === name) || null;
        fakePlugins.refresh = () => {};
        Object.defineProperty(navigator, 'plugins', { get: () => fakePlugins, configurable: true });
    })();
    """,

    # 4. navigator.languages — ensure consistent with Accept-Language
    """
    Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
        configurable: true,
    });
    """,

    # 5. WebGL vendor/renderer — spoof to common Intel GPU (most common worldwide)
    """
    (function() {
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(param) {
            if (param === 37445) return 'Google Inc. (Intel)';  // UNMASKED_VENDOR_WEBGL
            if (param === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics 630, OpenGL 4.5)';  // UNMASKED_RENDERER_WEBGL
            return getParameter.call(this, param);
        };
        if (typeof WebGL2RenderingContext !== 'undefined') {
            const getParameter2 = WebGL2RenderingContext.prototype.getParameter;
            WebGL2RenderingContext.prototype.getParameter = function(param) {
                if (param === 37445) return 'Google Inc. (Intel)';
                if (param === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics 630, OpenGL 4.5)';
                return getParameter2.call(this, param);
            };
        }
    })();
    """,

    # 6. Permissions API — make "notifications" return "denied" like real Chrome
    """
    if (navigator.permissions) {
        const origQuery = navigator.permissions.query.bind(navigator.permissions);
        navigator.permissions.query = function(params) {
            if (params.name === 'notifications') {
                return Promise.resolve({ state: 'denied', onchange: null });
            }
            return origQuery(params);
        };
    }
    """,

    # 7. Remove CDP artifacts — sourceURL patterns left by automation frameworks
    """
    (function() {
        const origToString = Function.prototype.toString;
        Function.prototype.toString = function() {
            const result = origToString.call(this);
            if (result.includes('native code')) return result;
            return result.replace(/\\/\\/# sourceURL=pptr:.*$/gm, '')
                         .replace(/\\/\\/# sourceURL=__playwright.*$/gm, '');
        };
    })();
    """,

    # 8. window.chrome.csi and chrome.loadTimes — expected by detection scripts
    """
    if (window.chrome) {
        if (!window.chrome.csi) {
            window.chrome.csi = function() {
                return {
                    startE: Date.now(),
                    onloadT: Date.now(),
                    pageT: Math.random() * 1000 + 500,
                    tran: 15,
                };
            };
        }
        if (!window.chrome.loadTimes) {
            window.chrome.loadTimes = function() {
                return {
                    commitLoadTime: Date.now() / 1000,
                    connectionInfo: 'h2',
                    finishDocumentLoadTime: Date.now() / 1000,
                    finishLoadTime: Date.now() / 1000,
                    firstPaintAfterLoadTime: 0,
                    firstPaintTime: Date.now() / 1000,
                    navigationType: 'Other',
                    npnNegotiatedProtocol: 'h2',
                    requestTime: Date.now() / 1000 - 0.5,
                    startLoadTime: Date.now() / 1000 - 0.5,
                    wasAlternateProtocolAvailable: false,
                    wasFetchedViaSpdy: true,
                    wasNpnNegotiated: true,
                };
            };
        }
    }
    """,
]


def get_stealth_scripts_combined() -> str:
    """Return all stealth scripts combined into a single JS string.

    Used with CDP Page.addScriptToEvaluateOnNewDocument to inject
    before any page JavaScript runs.
    """
    return "\n".join(STEALTH_SCRIPTS)
