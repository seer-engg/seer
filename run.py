#!/usr/bin/env python3
"""
Seer Launcher
Launches: Agents + UI
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from seer.shared.config import get_seer_config
from seer.shared.logger import get_logger

load_dotenv()

# Get logger for launcher
logger = get_logger('launcher')

class Launcher:
    """Manages all Seer processes"""
    
    def __init__(self):
        self.processes = []
        self.project_root = Path(__file__).parent
        self.python_exe = self._find_python_executable()
        self.langgraph_exe = self._find_langgraph_executable()
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def _find_python_executable(self):
        """Find the correct Python executable to use"""
        # Try virtual environment first
        venv_python = self.project_root / "venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        
        # Fall back to system python
        return "python"
    
    def _find_langgraph_executable(self):
        """Find the correct LangGraph CLI to use"""
        # Prefer venv langgraph if present; else search PATH
        venv_langgraph = self.project_root / "venv" / "bin" / "langgraph"
        if venv_langgraph.exists():
            return str(venv_langgraph)
        # Use system langgraph (installed via pip in PyEnv/system Python)
        # Get the full path to avoid PATH issues in subprocesses
        import shutil
        langgraph_path = shutil.which("langgraph")
        if langgraph_path:
            return langgraph_path
        return "langgraph"  # Fallback
        
    def start_process(self, name: str, command: list, cwd: str = None, env: dict = None):
        """Start a process and track it"""
        logger.info(f"▶️  Starting {name}...")
        
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        # Ensure child processes can import the seer package even when run from subdirs
        try:
            py_paths = []
            # Add project root and its parent (so `import seer` resolves when not installed)
            py_paths.append(str(self.project_root))
            py_paths.append(str(self.project_root.parent))
            existing_py_path = process_env.get("PYTHONPATH", "")
            if existing_py_path:
                py_paths.append(existing_py_path)
            process_env["PYTHONPATH"] = os.pathsep.join(py_paths)
        except Exception:
            pass
        
        # Create log file for this process
        log_file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        log_file = self.logs_dir / f"{log_file_name}.log"
        
        # Open log file
        log_handle = open(log_file, "w")
        
        # Start process with PIPE to capture output and strip ANSI codes
        process = subprocess.Popen(
            command,
            cwd=cwd or str(self.project_root),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            preexec_fn=os.setsid  # Start a new process group so we can terminate children
        )
        
        # Start a thread to read output and strip ANSI codes
        import threading
        def strip_ansi_and_log():
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # Strip ANSI escape codes
                clean_line = ansi_escape.sub('', line)
                log_handle.write(clean_line)
                log_handle.flush()
        
        log_thread = threading.Thread(target=strip_ansi_and_log, daemon=True)
        log_thread.start()
        
        self.processes.append((name, process, log_handle))
        logger.info(f"✅ {name} started (PID: {process.pid})")
        logger.info(f"   📝 Logs: {log_file}")
        return process
    
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            config = get_seer_config()
            return config._is_port_available(port)
        except Exception:
            # Fallback to original implementation
            import socket
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return True
            except OSError:
                return False
    
    def check_port_listening(self, port: int, timeout: int = 15) -> bool:
        """Check if a port is listening (health check)"""
        import socket
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect(('127.0.0.1', port))
                    return True
            except (socket.error, socket.timeout):
                time.sleep(0.5)
        
        return False
    
    def cleanup_existing_processes(self):
        """Kill any existing Seer processes"""
        logger.info("🧹 Cleaning up any existing Seer processes...")

        # Kill langgraph processes
        try:
            subprocess.run(["pkill", "-f", "langgraph.*dev"], check=False)
        except:
            pass

        # Kill streamlit processes
        try:
            subprocess.run(["pkill", "-f", "streamlit.*ui"], check=False)
        except:
            pass

        time.sleep(2)  # Give processes time to die

    async def start_all(self):
        """Start all components in order"""
        logger.info("🔮 Starting Seer\n")
        logger.info("=" * 60)
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("❌ Error: OPENAI_API_KEY not found in environment")
            logger.error("   Set it in .env file or export it")
            sys.exit(1)
        
        # Check if we're using the virtual environment
        if not self.python_exe.endswith("venv/bin/python"):
            logger.warning("⚠️  Warning: Not using virtual environment Python")
            logger.warning(f"   Using: {self.python_exe}")
            logger.warning("   Consider running: source venv/bin/activate")
        
        # Clean up any existing processes first
        self.cleanup_existing_processes()
        
        try:
            # Get configuration
            config = get_seer_config()
            
            # 1. Start Orchestrator Agent (langgraph dev)
            logger.info("\n1️⃣  Orchestrator Agent (LangGraph)")

            # Try to find an available port starting from configured port
            orchestrator_port = config.orchestrator_port
            if not self.check_port_available(orchestrator_port):
                orchestrator_port = config.get_available_port(config.orchestrator_port, config.orchestrator_port + 10)

            if orchestrator_port != config.orchestrator_port:
                logger.warning(f"⚠️  Port {config.orchestrator_port} in use, using port {orchestrator_port} instead")
                logger.warning(f"   Update your UI to use: http://127.0.0.1:{orchestrator_port}")

            self.start_process(
                "Orchestrator (LangGraph)",
                [self.langgraph_exe, "dev", "--port", str(orchestrator_port), "--host", "127.0.0.1"],
                cwd=str(self.project_root / "agents" / "orchestrator")
            )

            # Wait for port to be listening
            if not self.check_port_listening(orchestrator_port, timeout=15):
                logger.error(f"❌ Orchestrator agent failed to start on port {orchestrator_port}")
                logger.error(f"   Check logs: {self.logs_dir}/orchestrator_langgraph.log")
                self.stop_all()
                sys.exit(1)
            
            # 2. Start Eval Agent (langgraph dev)
            logger.info("\n2️⃣  Eval Agent (LangGraph)")
            eval_port = config.eval_agent_port
            if not self.check_port_available(eval_port) or eval_port == orchestrator_port:
                eval_port = config.get_available_port(config.eval_agent_port, config.eval_agent_port + 10)
            
            self.start_process(
                "Eval Agent (LangGraph)",
                [self.langgraph_exe, "dev", "--port", str(eval_port), "--host", "127.0.0.1"],
                cwd=str(self.project_root / "agents" / "eval_agent")
            )
            
            # Wait for port to be listening
            if not self.check_port_listening(eval_port, timeout=15):
                logger.error(f"❌ Eval agent failed to start on port {eval_port}")
                logger.error(f"   Check logs: {self.logs_dir}/eval_agent_langgraph.log")
                self.stop_all()
                sys.exit(1)
            
            # 3. Start Coding Agent (langgraph dev)
            logger.info("\n3️⃣  Coding Agent (LangGraph)")
            try:
                from seer.shared.config import get_config as get_agent_config
                agent_config = get_agent_config()
                coding_port = agent_config.get_agent_config("coding_agent").get("port", 8003)
            except Exception:
                coding_port = 8003  # Fallback to default
            if not self.check_port_available(coding_port) or coding_port in [orchestrator_port, eval_port]:
                coding_port = config.get_available_port(coding_port, coding_port + 10)
            
            self.start_process(
                "Coding Agent (LangGraph)",
                [self.langgraph_exe, "dev", "--port", str(coding_port), "--host", "127.0.0.1"],
                cwd=str(self.project_root / "agents" / "coding_agent")
            )
            
            # Wait for port to be listening
            if not self.check_port_listening(coding_port, timeout=15):
                logger.error(f"❌ Coding agent failed to start on port {coding_port}")
                logger.error(f"   Check logs: {self.logs_dir}/coding_agent_langgraph.log")
                self.stop_all()
                sys.exit(1)
            # 4. Start Streamlit UI
            logger.info("\n4️⃣  Streamlit UI")
            ui_port = config.ui_port
            if not self.check_port_available(ui_port):
                ui_port = config.get_available_port(config.ui_port, config.ui_port + 10)
            
            ui_env = {
                "ORCHESTRATOR_URL": f"http://127.0.0.1:{orchestrator_port}"
            }
            self.start_process(
                "Streamlit UI",
                ["streamlit", "run", "ui/streamlit_app.py", "--server.headless", "true", "--server.port", str(ui_port)],
                env=ui_env
            )
            time.sleep(3)
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ All components started!\n")
            logger.info("🔮 Seer Agents are running:")
            logger.info(f"   - UI:                http://localhost:{ui_port}")
            logger.info("=" * 60)
            logger.info("Press Ctrl+C to stop all components\n")
            
        except Exception as e:
            logger.error(f"\n❌ Error starting components: {e}")
            self.stop_all()
            sys.exit(1)
    
    def stop_all(self):
        """Stop all processes"""
        logger.info("\n🛑 Stopping all components...")
        for item in reversed(self.processes):
            name = item[0]
            process = item[1]
            log_handle = item[2] if len(item) > 2 else None
            
            try:
                logger.info(f"   Stopping {name}...")
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except Exception:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        process.kill()
            except Exception as e:
                logger.error(f"   Error stopping {name}: {e}")
            finally:
                if log_handle:
                    try:
                        log_handle.close()
                    except:
                        pass
        logger.info("✅ All components stopped")
    
    def wait(self):
        """Wait for all processes and monitor health"""
        try:
            while True:
                time.sleep(1)
                
                # Check if any process died
                for item in self.processes:
                    name = item[0]
                    process = item[1]
                    log_handle = item[2] if len(item) > 2 else None
                    
                    if process.poll() is not None:
                        logger.warning(f"\n⚠️  {name} process died unexpectedly (exit code: {process.returncode})")
                        
                        # Read last lines from log file
                        if log_handle:
                            log_file = log_handle.name
                            logger.warning(f"   📝 Check logs: {log_file}")
                            try:
                                with open(log_file, "r") as f:
                                    lines = f.readlines()
                                    logger.warning(f"   Last 20 lines of log:")
                                    for line in lines[-20:]:
                                        logger.warning(f"     {line.rstrip()}")
                            except:
                                pass
                        
                        self.stop_all()
                        return
                        
        except KeyboardInterrupt:
            self.stop_all()


async def main():
    """Main entry point"""
    launcher = Launcher()
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        launcher.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    await launcher.start_all()
    launcher.wait()

def cli():
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    cli()
