# Browser Stealth Anti-Detection System

## Overview

Seer injects stealth measures into every browser session to reduce bot detection (CAPTCHAs, blocks) when users log into sites like LinkedIn, Google, etc. via interactive sessions or workflow browser nodes.

The stealth system has two layers:
1. **CDP-level UA override** — replaces Browserless's default `HeadlessChrome/...` user-agent
2. **JS stealth scripts** — injected before any page JS runs, masking automation fingerprints

## Architecture

```
pool_manager.py                    stealth_config.py
─────────────────                  ─────────────────
create_session()                   CHROME_USER_AGENTS (Chrome 134)
    │                              STEALTH_SCRIPTS (8 scripts)
    ├─ browser_session.start()     get_stealth_scripts_combined()
    │                              get_platform_user_agent()
    ├─ _inject_stealth_scripts()
    │   ├─ Emulation.setUserAgentOverride   ← forces UA at CDP protocol level
    │   ├─ Page.addScriptToEvaluateOnNewDocument  ← persists across navigations
    │   └─ Runtime.evaluate                 ← applies to current page
    │
    └─ _apply_cookies_via_cdp()
```

Stealth injection is gated by `config.browser_stealth_enabled` (default: `True`).

## Detection Vectors Addressed

| # | Vector | How we address it | File |
|---|--------|-------------------|------|
| 1 | **User-Agent says HeadlessChrome** | `Emulation.setUserAgentOverride` with Chrome/134 UA | `pool_manager.py` |
| 2 | **navigator.webdriver = true** | Delete from `Navigator.prototype`, redefine as `undefined` | `stealth_config.py` script #1 |
| 3 | **Missing chrome.runtime** | Stub `chrome.runtime` with connect/sendMessage/onMessage | `stealth_config.py` script #2 |
| 4 | **navigator.plugins not PluginArray** | Build plugins via `Object.create(PluginArray.prototype)` | `stealth_config.py` script #3 |
| 5 | **navigator.languages inconsistent** | Override to `['en-US', 'en']` matching Accept-Language | `stealth_config.py` script #4 |
| 6 | **WebGL fingerprint reveals headless** | Spoof vendor/renderer to Intel UHD 630 | `stealth_config.py` script #5 |
| 7 | **Permissions API anomalies** | Notifications return `denied` (like real Chrome) | `stealth_config.py` script #6 |
| 8 | **CDP sourceURL artifacts** | Strip `pptr:` and `__playwright` from `Function.toString` | `stealth_config.py` script #7 |
| 9 | **Missing chrome.csi/loadTimes** | Stub both with realistic return values | `stealth_config.py` script #8 |

## Detection Vectors NOT Yet Addressed

| Vector | Why | Potential fix |
|--------|-----|---------------|
| **Runtime.Enable CDP leak** | browser-use calls `Runtime.enable` via cdp_use (actor/page.py:66). Anti-bots detect the side effects. | Would require patching browser-use or cdp_use upstream |
| **Canvas fingerprint** | Deterministic canvas rendering reveals headless at binary level | CloakBrowser (C++ Chromium patches) |
| **TLS fingerprint** | Client Hello differs from real Chrome | CloakBrowser or custom Chromium build |
| **AudioContext fingerprint** | Not spoofed | CloakBrowser |
| **IP reputation** | Datacenter IPs are flagged | Residential proxy (deployment concern) |

## Key Implementation Details

### Why CDP-level UA override?
Browserless launches Chrome with `--headless=new`, which sets the internal UA to `HeadlessChrome/...`. The `BrowserUseProfile.user_agent` parameter is NOT applied by browser-use for remote CDP connections. We must use `Emulation.setUserAgentOverride` to force it at the protocol level.

### Why delete+redefine for webdriver?
Chrome sets `navigator.webdriver` at the C++ level on `Navigator.prototype`. A simple `Object.defineProperty(navigator, 'webdriver', ...)` creates a property on the instance that may not override the prototype getter. We delete from the prototype first, then redefine there.

### Why PluginArray.prototype?
Detection scripts check `navigator.plugins instanceof PluginArray`. A plain array with plugin-like objects fails this check. We use `Object.create(PluginArray.prototype)` to build objects that pass the prototype chain instanceof check.

### Why addScriptToEvaluateOnNewDocument?
Unlike `Runtime.evaluate` (one-shot), `Page.addScriptToEvaluateOnNewDocument` runs the script before any page JS on every navigation. This ensures stealth overrides are in place even when the user navigates to new pages during an interactive session.

## Maintenance

### Updating Chrome UA version
When Chrome releases a new major version, update `CHROME_USER_AGENTS` in `stealth_config.py`. The comment at the top tracks when it was last updated.

### Adding new stealth scripts
Append to the `STEALTH_SCRIPTS` list in `stealth_config.py`. Each script is an IIFE or short snippet. Add a corresponding test in `tests/unit/services/browser/test_stealth_config.py`.

## Files

| File | Role |
|------|------|
| `src/seer/services/browser/stealth_config.py` | UA strings, stealth JS scripts, config helpers |
| `src/seer/services/browser/pool_manager.py` | `_inject_stealth_scripts()` — CDP injection at session start |
| `src/seer/config.py` | `browser_stealth_enabled` flag |
| `tests/unit/services/browser/test_stealth_config.py` | Unit tests for UA, scripts, config |

## Future Improvements (Tier 3)

If JS-level stealth is insufficient for specific sites, consider:
- **CloakBrowser** — OSS custom Chromium with 48 C++ patches (canvas, WebGL, audio, TLS). CDP-compatible, can replace the Browserless image.
- **Camoufox** — Firefox fork with best anti-detection, but requires architecture change (Juggler protocol, not CDP).
- **Residential proxies** — for IP reputation issues.
