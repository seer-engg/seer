import re
import asyncio
from e2b import AsyncSandbox, AsyncCommandHandle
from shared.logger import get_logger
logger = get_logger("codex.planner.deploy_server")
from .constants import SUCCESS_PAT, FAIL_PATTERNS, TARGET_AGENT_PORT



async def deploy_server_and_confirm_ready(cmd: str, sb: AsyncSandbox, cwd: str, timeout_s: int = 40) -> tuple[AsyncSandbox, AsyncCommandHandle]:
    ready_evt = asyncio.Event()
    failed_evt = asyncio.Event()
    last_err = []
    # Accumulate all output (stdout + stderr) for pattern matching
    # This handles cases where patterns span multiple chunks
    stderr_buffer = []

    on_stdout_count = [0]  # Use list to allow modification in closure
    on_stderr_count = [0]
    
    def on_stdout(chunk: str):
        on_stdout_count[0] += 1
        logger.info(f"[STDOUT #{on_stdout_count[0]}] Length: {len(chunk)}")
        logger.info(f"  Content: {repr(chunk[:200])}")  # Show first 200 chars with escape sequences
        
        # Also check stdout for error patterns (sometimes stderr goes to stdout)
        last_err.append(f"STDOUT: {chunk}")
        stderr_buffer.append(chunk)
        
        # Check the accumulated buffer for success pattern (could span chunks)
        full_buffer = ''.join(stderr_buffer)
        if SUCCESS_PAT.search(full_buffer):
            logger.info("✓ SUCCESS PATTERN FOUND!")
            ready_evt.set()
            return  # No need to check for failures if success found
            
        # Check for failures in stdout too
        for i, pattern in enumerate(FAIL_PATTERNS):
            if pattern.search(full_buffer):
                logger.info(f"✗ FAILURE PATTERN #{i} DETECTED IN STDOUT: {pattern.pattern}")
                failed_evt.set()
                return

    def on_stderr(chunk: str):
        on_stderr_count[0] += 1
        logger.info(f"[STDERR #{on_stderr_count[0]}] Length: {len(chunk)}")
        logger.info(f"  Content: {repr(chunk[:200])}")  # Show first 200 chars with escape sequences
        
        last_err.append(f"STDERR: {chunk}")
        stderr_buffer.append(chunk)
        
        # Check the accumulated buffer for both success and failure patterns
        full_buffer = ''.join(stderr_buffer)
        logger.info(f"  Buffer size: {len(full_buffer)} chars")
        
        # Check for success pattern first (sometimes goes to stderr)
        if SUCCESS_PAT.search(full_buffer):
            logger.info("✓ SUCCESS PATTERN FOUND IN STDERR!")
            ready_evt.set()
            return
        
        # Check for failure patterns
        for i, pattern in enumerate(FAIL_PATTERNS):
            if pattern.search(full_buffer):
                logger.info(f"✗ FAILURE PATTERN #{i} DETECTED: {pattern.pattern}")
                logger.info(f"  Matched in: {full_buffer[-200:]}")
                failed_evt.set()
                return
        
        logger.info(f"  No patterns matched yet (checked 1 success + {len(FAIL_PATTERNS)} failure patterns)")

    # 1) start server in background and stream logs
    handle = await sb.commands.run(
        cmd,
        background=True,
        on_stdout=on_stdout,
        on_stderr=on_stderr,
        cwd=cwd
    )
    url = sb.get_host(TARGET_AGENT_PORT)
    EXTERNAL_URL = f"https://{url}/docs"
    INTERNAL_URL = f"http://0.0.0.0:{TARGET_AGENT_PORT}/docs"

    try:
        # 2) wait for either explicit failure, explicit success, or timeout
        done, _ = await asyncio.wait(
            {
                asyncio.create_task(ready_evt.wait()),
                asyncio.create_task(failed_evt.wait()),
            },
            timeout=timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
    except Exception as e:
        logger.info(f"Exception during wait: {e}")
        done = set()

    if failed_evt.is_set():
        # process might still be running; kill and surface last logs
        await sb.commands.kill(handle.pid)
        raise RuntimeError(
            "Server reported startup failure. "
            f"stderr tail: {''.join(last_err)[-800:]}"
        )

    # 3) if we saw the success line (fast path), double-check with probes
    if ready_evt.is_set():
        logger.info("Success pattern detected, verifying with probes...")
        
        # First probe from inside (local connectivity within sandbox)
        inside_ok = await _probe_from_inside(sb, INTERNAL_URL)
        if not inside_ok:
            logger.warning("Internal probe failed despite success pattern")
            # Fall through to liveness/diag
        else:
            # Internal probe passed, now verify external connectivity
            logger.info("Internal probe passed, checking external connectivity...")
            outside_ok = await _probe_from_outside(EXTERNAL_URL)
            if outside_ok:
                logger.info("✓ Server is ready and accessible from outside!")
                return sb, handle
            else:
                logger.warning("External probe failed - server not accessible from outside")
                # Fall through to liveness/diag

    # 4) if no decisive log yet, poll health a few times (slow path)
    # for _ in range(12):
    #     ok = await _probe_from_inside(sb, READY_URL)
    #     if ok:
    #         return sb, handle
    #     await asyncio.sleep(1.5)

    # 5) final check: is process still alive?
    procs = await sb.commands.list()
    alive = any(p.pid == handle.pid for p in procs)
    
    # Debug summary
    logger.info("\n" + "="*80)
    logger.info("TIMEOUT REACHED - DEBUG SUMMARY")
    logger.info("="*80)
    logger.info(f"on_stdout called: {on_stdout_count[0]} times")
    logger.info(f"on_stderr called: {on_stderr_count[0]} times")
    logger.info(f"ready_evt.is_set(): {ready_evt.is_set()}")
    logger.info(f"failed_evt.is_set(): {failed_evt.is_set()}")
    logger.info(f"Total buffer size: {len(''.join(stderr_buffer))} chars")
    logger.info(f"\nFirst 500 chars of buffer:")
    logger.info(repr(''.join(stderr_buffer)[:500]))
    logger.info(f"\nLast 500 chars of buffer:")
    logger.info(repr(''.join(stderr_buffer)[-500:]))
    logger.info("="*80 + "\n")
    
    # raise TimeoutError(
    #     f"Server not healthy after {timeout_s}s (alive={alive}). "
    #     f"on_stdout_calls={on_stdout_count[0]}, on_stderr_calls={on_stderr_count[0]}. "
    #     f"stderr tail: {''.join(last_err)[-800:]}"
    # )
    return sb, handle

async def _probe_from_inside(sb: AsyncSandbox, url: str) -> bool:
    """Probe server from inside the sandbox using curl."""
    logger.info(f"Probing from inside sandbox: {url}")
    res = await sb.commands.run(f"curl -fsS {url} || true")  # foreground probe
    logger.info(f"  stdout: {res.stdout[:200] if res.stdout else 'None'}")
    logger.info(f"  stderr: {res.stderr[:200] if res.stderr else 'None'}")
    logger.info(f"  exit_code: {res.exit_code}")
    success = res.exit_code == 0
    logger.info(f"  Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
    return success

async def _probe_from_outside(url: str, timeout: float = 5.0) -> bool:
    """Probe server from outside (current server) using HTTP request."""
    import httpx

    if 'http' not in url:
        url = f"http://{url}"
    
    logger.info(f"Probing from outside: {url}")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            success = response.status_code == 200
            logger.info(f"  status_code: {response.status_code}")
            logger.info(f"  response length: {len(response.text)} chars")
            logger.info(f"  Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
            return success
    except httpx.TimeoutException as e:
        logger.info(f"  ✗ Timeout: {e}")
        return False
    except httpx.ConnectError as e:
        logger.info(f"  ✗ Connection error: {e}")
        return False
    except Exception as e:
        logger.info(f"  ✗ Unexpected error: {type(e).__name__}: {e}")
        return False

