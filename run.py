#!/usr/bin/env python3
"""
Seer Launcher
Launches: LangGraph A2A Agents + UI
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from seer.shared.config import get_seer_config

load_dotenv()

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
        print(f"▶️  Starting {name}...")
        
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
            universal_newlines=True
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
        print(f"✅ {name} started (PID: {process.pid})")
        print(f"   📝 Logs: {log_file}")
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
        print("🧹 Cleaning up any existing Seer processes...")

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
        print("🔮 Starting Seer\n")
        print("=" * 60)
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not found in environment")
            print("   Set it in .env file or export it")
            sys.exit(1)
        
        # Check if we're using the virtual environment
        if not self.python_exe.endswith("venv/bin/python"):
            print("⚠️  Warning: Not using virtual environment Python")
            print(f"   Using: {self.python_exe}")
            print("   Consider running: source venv/bin/activate")
        
        # Clean up any existing processes first
        self.cleanup_existing_processes()
        
        try:
            # Get configuration
            config = get_seer_config()
            
            # 1. Start Orchestrator Agent (langgraph dev)
            print("\n1️⃣  Orchestrator Agent (LangGraph)")

            # Try to find an available port starting from configured port
            orchestrator_port = config.orchestrator_port
            if not self.check_port_available(orchestrator_port):
                orchestrator_port = config.get_available_port(config.orchestrator_port, config.orchestrator_port + 10)

            if orchestrator_port != config.orchestrator_port:
                print(f"⚠️  Port {config.orchestrator_port} in use, using port {orchestrator_port} instead")
                print(f"   Update your UI to use: http://127.0.0.1:{orchestrator_port}")

            self.start_process(
                "Orchestrator (LangGraph)",
                [self.langgraph_exe, "dev", "--port", str(orchestrator_port), "--host", "127.0.0.1"],
                cwd=str(self.project_root / "agents" / "orchestrator")
            )

            # Wait for port to be listening
            if not self.check_port_listening(orchestrator_port, timeout=15):
                print(f"❌ Orchestrator agent failed to start on port {orchestrator_port}")
                print(f"   Check logs: {self.logs_dir}/orchestrator_langgraph.log")
                self.stop_all()
                sys.exit(1)
            
            # 2. Start Customer Success Agent (langgraph dev)
            print("\n2️⃣  Customer Success Agent (LangGraph)")
            cs_port = config.customer_success_port
            if not self.check_port_available(cs_port) or cs_port == orchestrator_port:
                cs_port = config.get_available_port(config.customer_success_port, config.customer_success_port + 10)
            
            self.start_process(
                "Customer Success (LangGraph)",
                [self.langgraph_exe, "dev", "--port", str(cs_port), "--host", "127.0.0.1"],
                cwd=str(self.project_root / "agents" / "customer_success")
            )
            
            # Wait for port to be listening
            if not self.check_port_listening(cs_port, timeout=15):
                print(f"❌ Customer Success agent failed to start on port {cs_port}")
                print(f"   Check logs: {self.logs_dir}/customer_success_langgraph.log")
                self.stop_all()
                sys.exit(1)
            
            # 3. Start Eval Agent (langgraph dev)
            print("\n3️⃣  Eval Agent (LangGraph)")
            eval_port = config.eval_agent_port
            if not self.check_port_available(eval_port) or eval_port in [orchestrator_port, cs_port]:
                eval_port = config.get_available_port(config.eval_agent_port, config.eval_agent_port + 10)
            
            self.start_process(
                "Eval Agent (LangGraph)",
                [self.langgraph_exe, "dev", "--port", str(eval_port), "--host", "127.0.0.1"],
                cwd=str(self.project_root / "agents" / "eval_agent")
            )
            
            # Wait for port to be listening
            if not self.check_port_listening(eval_port, timeout=15):
                print(f"❌ Eval agent failed to start on port {eval_port}")
                print(f"   Check logs: {self.logs_dir}/eval_agent_langgraph.log")
                self.stop_all()
                sys.exit(1)
            
            # Start Streamlit UI
            print("\n4️⃣  Streamlit UI")
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
            
            print("\n" + "=" * 60)
            print("✅ All components started!\n")
            print("🔮 Seer is running (A2A Hub-and-Spoke Architecture):")
            print(f"   - UI:                http://localhost:{ui_port}")
            print(f"   - Orchestrator:      http://127.0.0.1:{orchestrator_port}")
            print(f"   - CS Agent API:      http://127.0.0.1:{cs_port}")
            print(f"   - Eval Agent API:    http://127.0.0.1:{eval_port}")
            print("=" * 60)
            print("\n💡 Agents communicate through the Orchestrator agent (hub-and-spoke)")
            print("   All messages are routed through the central orchestrator")
            print("   Use the 'Orchestrator Monitor' tab to see message flow\n")
            print("Press Ctrl+C to stop all components\n")
            
        except Exception as e:
            print(f"\n❌ Error starting components: {e}")
            self.stop_all()
            sys.exit(1)
    
    def stop_all(self):
        """Stop all processes"""
        print("\n🛑 Stopping all components...")
        for item in reversed(self.processes):
            name = item[0]
            process = item[1]
            log_handle = item[2] if len(item) > 2 else None
            
            try:
                print(f"   Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"   Error stopping {name}: {e}")
            finally:
                if log_handle:
                    try:
                        log_handle.close()
                    except:
                        pass
        print("✅ All components stopped")
    
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
                        print(f"\n⚠️  {name} process died unexpectedly (exit code: {process.returncode})")
                        
                        # Read last lines from log file
                        if log_handle:
                            log_file = log_handle.name
                            print(f"   📝 Check logs: {log_file}")
                            try:
                                with open(log_file, "r") as f:
                                    lines = f.readlines()
                                    print(f"   Last 20 lines of log:")
                                    for line in lines[-20:]:
                                        print(f"     {line.rstrip()}")
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
