"""
Test cookie persistence round-trip without requiring Google login.
Uses httpbin.org which sets simple test cookies.

Run with: uv run python tests/debug/test_cookie_restore.py
"""
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_cookie_round_trip():
    """Test that cookies are saved and restored correctly."""
    # Import here to avoid issues when file is imported but not run
    from seer.services.browser.pool_manager import BrowserPoolManager

    logger.info("=== Starting Cookie Round-Trip Test ===")

    # 1. Get pool instance
    pool = await BrowserPoolManager.get_instance()
    logger.info("Pool instance acquired")

    # 2. Create first session WITHOUT storage_state
    logger.info("Creating session 1 (no cookies)...")
    session1 = await pool.create_session(
        user_id="test-user",
        session_type="interactive",
        storage_state=None,
        stealth_mode=True,
    )
    logger.info(f"Session 1 created: {session1.id}")

    # 3. Navigate to cookie-setting page
    page = await session1.session.must_get_current_page()
    logger.info("Navigating to httpbin.org/cookies/set...")
    await page.goto("https://httpbin.org/cookies/set/test_cookie/test_value_12345")
    await asyncio.sleep(1)  # Wait for cookie to be set

    # 4. Verify cookie was set by checking cookies page
    logger.info("Verifying cookie was set...")
    await page.goto("https://httpbin.org/cookies")
    await asyncio.sleep(0.5)
    content1 = await page.content()
    logger.info(f"Session 1 - Cookies page content: {content1[:500]}...")

    if "test_cookie" not in content1:
        logger.error("Cookie was NOT set in session 1!")
        await pool.shutdown()
        return False

    logger.info("Cookie verified in session 1")

    # 5. Export state and release session
    logger.info("Releasing session 1 and exporting storage state...")
    state = await pool.release_session(session1.id)

    if not state:
        logger.error("No storage state returned from release_session!")
        await pool.shutdown()
        return False

    cookies = state.get("cookies", [])
    logger.info(f"Exported {len(cookies)} cookies from session 1")

    # Log the actual cookies for debugging
    for cookie in cookies:
        logger.info(f"  Cookie: {cookie.get('name')}={cookie.get('value')[:20]}... (domain: {cookie.get('domain')})")

    test_cookie = next((c for c in cookies if c.get("name") == "test_cookie"), None)
    if not test_cookie:
        logger.error("test_cookie not found in exported cookies!")
        await pool.shutdown()
        return False

    logger.info(f"Found test_cookie in exported state: {test_cookie.get('value')}")

    # 6. Create NEW session WITH the saved storage_state
    logger.info("Creating session 2 WITH storage_state (cookies should be restored)...")
    session2 = await pool.create_session(
        user_id="test-user",
        session_type="interactive",
        storage_state=state,
        stealth_mode=True,
    )
    logger.info(f"Session 2 created: {session2.id}")

    # 7. Check if cookies are present in the new session
    page2 = await session2.session.must_get_current_page()
    logger.info("Navigating to httpbin.org/cookies in session 2...")
    await page2.goto("https://httpbin.org/cookies")
    await asyncio.sleep(0.5)
    content2 = await page2.content()
    logger.info(f"Session 2 - Cookies page content: {content2[:500]}...")

    # 8. Verify the cookie persisted
    if "test_cookie" in content2 and "test_value_12345" in content2:
        logger.info("SUCCESS! Cookie was restored in session 2!")
        success = True
    else:
        logger.error("FAILURE! Cookie was NOT restored in session 2!")
        success = False

    # 9. Cleanup
    logger.info("Cleaning up...")
    await pool.release_session(session2.id)
    await pool.shutdown()

    logger.info("=== Test Complete ===")
    return success


if __name__ == "__main__":
    result = asyncio.run(test_cookie_round_trip())
    exit(0 if result else 1)
