from e2b import AsyncSandbox


async def kill_process_on_port(sb: AsyncSandbox, port: int):
    kill_cmd = f"lsof -ti :{port} | xargs -r kill -9"
    kill_proc = await sb.commands.run(kill_cmd)
    if "not found" in kill_proc.stderr:
        raise RuntimeError(f"Cannot clear port {port}: lsof command not found.")
    return True
