#!/usr/bin/env python3
"""
Seer Launcher
Launches: Event Bus + LangGraph Dev Agents + Wrappers + UI
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

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
        
        # Create log file for this process
        log_file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        log_file = self.logs_dir / f"{log_file_name}.log"
        
        # Open log file
        log_handle = open(log_file, "w")
        
        process = subprocess.Popen(
            command,
            cwd=cwd or str(self.project_root),
            env=process_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        self.processes.append((name, process, log_handle))
        print(f"✅ {name} started (PID: {process.pid})")
        print(f"   📝 Logs: {log_file}")
        return process
    
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available"""
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
        
        # Kill uvicorn processes (Event Bus)
        try:
            subprocess.run(["pkill", "-f", "uvicorn.*event_bus"], check=False)
        except:
            pass
        
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
        
        # Kill bridge processes
        try:
            subprocess.run(["pkill", "-f", "eventbus_bridge\\.py"], check=False)
        except:
            pass
        
        time.sleep(2)  # Give processes time to die
    
    def start_all(self):
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
            # 1. Start Event Bus
            print("\n1️⃣  Event Bus")
            
            # Try to find an available port starting from 8000
            event_bus_port = 8000
            while not self.check_port_available(event_bus_port) and event_bus_port < 8010:
                event_bus_port += 1
            
            if event_bus_port >= 8010:
                print("❌ No available ports found in range 8000-8009!")
                print("   Please stop any existing Seer processes:")
                print("   - Kill any running uvicorn processes")
                print("   - Or restart your terminal")
                print("   - Or use: pkill -f uvicorn")
                sys.exit(1)
            
            if event_bus_port != 8000:
                print(f"⚠️  Port 8000 in use, using port {event_bus_port} instead")
                print(f"   Update your UI to use: http://127.0.0.1:{event_bus_port}")
            
            self.start_process(
                "Event Bus",
                [self.python_exe, "-m", "uvicorn", "event_bus.server:app", "--host", "127.0.0.1", "--port", str(event_bus_port)]
            )
            
            # Wait for Event Bus to be ready
            if not self.check_port_listening(event_bus_port, timeout=10):
                print(f"❌ Event Bus failed to start on port {event_bus_port}")
                print(f"   Check logs: {self.logs_dir}/event_bus.log")
                self.stop_all()
                sys.exit(1)
            
            # 2. Start Customer Success Agent (langgraph dev)
            print("\n2️⃣  Customer Success Agent (LangGraph)")
            cs_port = 8001 if event_bus_port != 8001 else 8002
            self.start_process(
                "Customer Success (LangGraph)",
                [self.langgraph_exe, "dev", "--port", str(cs_port), "--host", "127.0.0.1", "--no-browser"],
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
            eval_port = 8002 if event_bus_port != 8002 and cs_port != 8002 else 8003
            self.start_process(
                "Eval Agent (LangGraph)",
                [self.langgraph_exe, "dev", "--port", str(eval_port), "--host", "127.0.0.1", "--no-browser"],
                cwd=str(self.project_root / "agents" / "eval_agent")
            )
            
            # Wait for port to be listening
            if not self.check_port_listening(eval_port, timeout=15):
                print(f"❌ Eval agent failed to start on port {eval_port}")
                print(f"   Check logs: {self.logs_dir}/eval_agent_langgraph.log")
                self.stop_all()
                sys.exit(1)
            
            # 4. Start Customer Success Bridge
            print("\n4️⃣  Customer Success Bridge")
            bridge_env = {
                "LANGGRAPH_URL": f"http://localhost:{cs_port}",
                "EVENT_BUS_URL": f"http://127.0.0.1:{event_bus_port}"
            }
            self.start_process(
                "CS Bridge",
                [self.python_exe, "agents/customer_success/eventbus_bridge.py"],
                env=bridge_env
            )
            time.sleep(2)
            
            # 5. Start Eval Agent Bridge
            print("\n5️⃣  Eval Agent Bridge")
            bridge_env = {
                "LANGGRAPH_URL": f"http://localhost:{eval_port}",
                "EVENT_BUS_URL": f"http://127.0.0.1:{event_bus_port}"
            }
            self.start_process(
                "Eval Bridge",
                [self.python_exe, "agents/eval_agent/eventbus_bridge.py"],
                env=bridge_env
            )
            time.sleep(2)
            
            # 6. Start Streamlit UI
            print("\n6️⃣  Streamlit UI")
            ui_env = {
                "EVENT_BUS_URL": f"http://127.0.0.1:{event_bus_port}"
            }
            self.start_process(
                "Streamlit UI",
                ["streamlit", "run", "ui/streamlit_app.py", "--server.headless", "true"],
                env=ui_env
            )
            time.sleep(3)
            
            print("\n" + "=" * 60)
            print("✅ All components started!\n")
            print("🔮 Seer is running:")
            print(f"   - UI:                http://localhost:8501")
            print(f"   - Event Bus:         http://127.0.0.1:{event_bus_port}")
            print(f"   - Bus API Docs:      http://127.0.0.1:{event_bus_port}/docs")
            print(f"   - CS Agent API:      http://127.0.0.1:{cs_port}")
            print(f"   - Eval Agent API:    http://127.0.0.1:{eval_port}")
            print("=" * 60)
            print("\n💡 Open the UI to interact with agents and debug conversations")
            print("   Use the 'Agent Threads' tab to see what each agent is doing\n")
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


def main():
    """Main entry point"""
    launcher = Launcher()
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        launcher.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    launcher.start_all()
    launcher.wait()


if __name__ == "__main__":
    main()

