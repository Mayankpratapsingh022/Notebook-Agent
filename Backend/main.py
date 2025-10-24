#!/usr/bin/env python3
"""
FastAPI Backend for Modal Notebook
Handles notebook operations and Modal sandbox execution
"""

import json
import base64
import io
import time
import uuid
from typing import Dict, List, Any, Optional
import modal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback
import logging
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class CellCreate(BaseModel):
    cell_type: str
    source: List[str]
    metadata: Optional[Dict[str, Any]] = {}

class CellUpdate(BaseModel):
    source: List[str]
    metadata: Optional[Dict[str, Any]] = {}

class CellExecute(BaseModel):
    cell_id: str
    code: str

class NotebookData(BaseModel):
    cells: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    nbformat: int
    nbformat_minor: int

class ModalNotebookKernel:
    """Modal-based notebook kernel for code execution"""
    
    def __init__(self, gpu_config: str = "cpu", cpu_cores: float = 4.0, memory_mb: int = 8192, timeout: int = 300):
        self.gpu_config = gpu_config
        self.cpu_cores = cpu_cores
        self.memory_mb = memory_mb
        self.timeout = timeout
        self._sandbox = None
        self._persistent_session = None
        self.execution_count = 0
        self.session_id = str(uuid.uuid4())
        
        # Create Modal image
        self.image = self._create_image()
        self._setup_sandbox()
    
    def _create_image(self):
        """Create Modal image with required packages"""
        packages = [
            "numpy", "matplotlib", "pandas", "scipy", "seaborn", 
            "plotly", "requests", "pillow", "jupyter", "ipykernel",
            "nbformat", "nbconvert"
        ]
        packages_str = " ".join(packages)
        
        return (modal.Image.debian_slim()
                .apt_install("git", "build-essential")
                .run_commands("pip install --upgrade pip")
                .run_commands(f"pip install {packages_str}"))
    
    def _setup_sandbox(self):
        """Setup Modal sandbox"""
        try:
            app = modal.App.lookup(f"notebook-kernel-{self.session_id}", create_if_missing=True)
            
            sandbox_kwargs = {
                "image": self.image,
                "cpu": self.cpu_cores,
                "memory": self.memory_mb,
                "timeout": self.timeout,
                "app": app
            }
            
            if self.gpu_config != "cpu":
                if self.gpu_config == "T4":
                    sandbox_kwargs["gpu"] = modal.gpu.T4()
                elif self.gpu_config == "A100":
                    sandbox_kwargs["gpu"] = modal.gpu.A100()
            
            self._sandbox = modal.Sandbox.create(**sandbox_kwargs)
            logger.info(f"Modal sandbox created with {self.gpu_config} GPU")
            
        except Exception as e:
            logger.error(f"Error creating Modal sandbox: {e}")
            raise
    
    def _start_persistent_session(self):
        """Start persistent Python session"""
        if self._persistent_session is not None:
            return
        
        logger.info("Starting persistent Python session...")
        
        session_script = '''
import sys
import json
import traceback
import base64
import io
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global namespace
_global_namespace = {
    '__builtins__': __builtins__,
    '__name__': '__main__',
    '__doc__': None,
    '__package__': None
}

_captured_figures = []

def _capture_show(*args, **kwargs):
    global _captured_figures
    try:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            _captured_figures.append(img_base64)
            buf.close()
            plt.close(fig)
    except Exception as e:
        print(f"Error capturing plot: {e}", file=sys.stderr)

plt.show = _capture_show

with open("/tmp/session_ready", "w") as f:
    f.write("READY")

print("Persistent Python session started", flush=True)

while True:
    try:
        if os.path.exists("/tmp/execute_command"):
            with open("/tmp/execute_command", "r") as f:
                content = f.read().strip()
                if not content:
                    continue
                try:
                    command = json.loads(content)
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {content[:100]}...", file=sys.stderr)
                    continue
            
            os.remove("/tmp/execute_command")
            
            if command.get("action") == "execute":
                code = command.get("code", "")
                _captured_figures = []
                
                try:
                    # Check if code contains shell commands (lines starting with !)
                    lines = code.strip().split('\\n')
                    shell_commands = []
                    python_code_lines = []
                    
                    for line in lines:
                        stripped_line = line.strip()
                        if stripped_line.startswith('!'):
                            # This is a shell command
                            shell_cmd = stripped_line[1:].strip()  # Remove the !
                            shell_commands.append(shell_cmd)
                        else:
                            # This is Python code
                            python_code_lines.append(line)
                    
                    stdout_parts = []
                    stderr_parts = []
                    
                    # Execute shell commands first
                    for shell_cmd in shell_commands:
                        try:
                            import subprocess
                            result = subprocess.run(
                                shell_cmd,
                                shell=True,
                                capture_output=True,
                                text=True,
                                timeout=60  # 60 second timeout for shell commands
                            )
                            
                            if result.stdout:
                                stdout_parts.append(f"$ {shell_cmd}")
                                stdout_parts.append(result.stdout.rstrip())
                            
                            if result.stderr:
                                stderr_parts.append(f"$ {shell_cmd}")
                                stderr_parts.append(result.stderr.rstrip())
                            
                            # If command failed, add error info
                            if result.returncode != 0:
                                stderr_parts.append(f"Command exited with code {result.returncode}")
                                
                        except subprocess.TimeoutExpired:
                            stderr_parts.append(f"$ {shell_cmd}")
                            stderr_parts.append("Command timed out after 60 seconds")
                        except Exception as e:
                            stderr_parts.append(f"$ {shell_cmd}")
                            stderr_parts.append(f"Error executing shell command: {str(e)}")
                    
                    # Execute Python code if present
                    python_stdout = ""
                    if python_code_lines and any(line.strip() for line in python_code_lines):
                        python_code = '\\n'.join(python_code_lines)
                        
                        # Capture stdout during Python execution
                        import io
                        from contextlib import redirect_stdout
                        
                        stdout_buffer = io.StringIO()
                        
                        with redirect_stdout(stdout_buffer):
                            # Execute code in the persistent namespace
                            exec(python_code, _global_namespace)
                        
                        python_stdout = stdout_buffer.getvalue()
                    
                    # Combine all stdout
                    all_stdout_parts = stdout_parts.copy()
                    if python_stdout:
                        all_stdout_parts.append(python_stdout.rstrip())
                    
                    stdout_output = '\\n'.join(all_stdout_parts) if all_stdout_parts else ""
                    stderr_output = '\\n'.join(stderr_parts) if stderr_parts else ""
                    
                    # Send results back
                    result = {
                        "status": "success",
                        "stdout": stdout_output,
                        "stderr": stderr_output,
                        "plots": _captured_figures.copy()
                    }
                    
                    with open("/tmp/execute_result", "w") as f:
                        f.write(json.dumps(result))
                    
                except Exception as e:
                    error_result = {
                        "status": "error",
                        "error": {
                            "name": type(e).__name__,
                            "value": str(e),
                            "traceback": traceback.format_exc()
                        }
                    }
                    
                    with open("/tmp/execute_result", "w") as f:
                        f.write(json.dumps(error_result))
            
            elif command.get("action") == "terminate":
                break
        
        else:
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Session error: {e}", file=sys.stderr)
        error_result = {
            "status": "error",
            "error": {
                "name": type(e).__name__,
                "value": str(e),
                "traceback": traceback.format_exc()
            }
        }
        with open("/tmp/execute_result", "w") as f:
            f.write(json.dumps(error_result))
'''
        
        self._persistent_session = self._sandbox.exec(
            "python3", "-c", session_script,
            timeout=None
        )
        
        # Wait for session to be ready
        max_wait = 10
        for _ in range(max_wait * 10):
            try:
                with self._sandbox.open("/tmp/session_ready", "r") as f:
                    if f.read().strip() == "READY":
                        logger.info("Persistent session ready")
                        return
            except Exception:
                pass
            time.sleep(0.1)
        
        raise RuntimeError("Failed to initialize persistent session")
    
    def execute_cell(self, code: str) -> Dict[str, Any]:
        """Execute a code cell"""
        if not self._sandbox:
            raise RuntimeError("Sandbox not initialized")
        
        if self._persistent_session is None:
            self._start_persistent_session()
        
        self.execution_count += 1
        logger.info(f"Executing cell #{self.execution_count}")
        
        # Clean up result files
        try:
            self._sandbox.exec("rm", "-f", "/tmp/execute_command", "/tmp/execute_result").wait()
        except Exception:
            pass
        
        # Send execution command
        command = {
            "action": "execute",
            "code": code
        }
        
        with self._sandbox.open("/tmp/execute_command", "w") as f:
            f.write(json.dumps(command))
        
        # Wait for result
        max_wait = 60
        result = None
        
        for _ in range(max_wait * 10):
            try:
                with self._sandbox.open("/tmp/execute_result", "r") as f:
                    result_json = f.read().strip()
                    if result_json:
                        try:
                            result = json.loads(result_json)
                            break
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
            time.sleep(0.1)
        
        if result is None:
            raise RuntimeError("Timeout waiting for code execution result")
        
        # Clean up
        try:
            self._sandbox.exec("rm", "-f", "/tmp/execute_result").wait()
        except Exception:
            pass
        
        return result
    
    def shutdown(self):
        """Shutdown the sandbox"""
        try:
            if self._persistent_session:
                terminate_command = {"action": "terminate"}
                with self._sandbox.open("/tmp/execute_command", "w") as f:
                    f.write(json.dumps(terminate_command))
                
                self._persistent_session.terminate()
                self._persistent_session = None
                logger.info("Persistent session terminated")
            
            if self._sandbox:
                self._sandbox.terminate()
                self._sandbox = None
                logger.info("Modal sandbox terminated")
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global notebook instance
notebook_data = {
    "cells": [
        {
            "id": "1",
            "cell_type": "markdown",
            "source": ["# Welcome to Modal notebooks!\n\nWrite Python code and collaborate in real time. Your code runs in Modal's **serverless cloud**, and anyone in the same workspace can join.\n\nThis notebook comes with some common Python libraries installed. Run cells with Shift+Enter."],
            "metadata": {},
            "execution_count": None
        },
        {
            "id": "2", 
            "cell_type": "code",
            "source": ["import matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(-6, 6, 1000)\ny = np.sinc(x)\n\nplt.plot(x, y, color=\"darkblue\")\nplt.axhline(0, color=\"black\", linewidth=0.5)\nplt.axvline(0, color=\"black\", linewidth=0.5)\nplt.grid(True, alpha=0.3)"],
            "metadata": {},
            "execution_count": None
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

kernel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI backend for Modal Notebook")
    yield
    # Shutdown
    if kernel:
        kernel.shutdown()
        logger.info("Modal kernel shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Modal Notebook API",
    description="FastAPI backend for Modal notebook with sandbox execution",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Modal Notebook API", "status": "running"}

@app.get("/api/notebook")
async def get_notebook():
    """Get the current notebook data"""
    return notebook_data

@app.post("/api/notebook")
async def update_notebook(notebook: NotebookData):
    """Update the entire notebook"""
    global notebook_data
    notebook_data = notebook.dict()
    return {"status": "success", "message": "Notebook updated"}

@app.post("/api/cells")
async def create_cell(cell: CellCreate):
    """Create a new cell"""
    cell_id = str(uuid.uuid4())
    new_cell = {
        "id": cell_id,
        "cell_type": cell.cell_type,
        "source": cell.source,
        "metadata": cell.metadata,
        "execution_count": None
    }
    notebook_data["cells"].append(new_cell)
    return {"status": "success", "cell_id": cell_id, "cell": new_cell}

@app.put("/api/cells/{cell_id}")
async def update_cell(cell_id: str, cell_update: CellUpdate):
    """Update a cell"""
    for i, cell in enumerate(notebook_data["cells"]):
        if cell["id"] == cell_id:
            notebook_data["cells"][i]["source"] = cell_update.source
            notebook_data["cells"][i]["metadata"] = cell_update.metadata
            return {"status": "success", "cell": notebook_data["cells"][i]}
    
    raise HTTPException(status_code=404, detail="Cell not found")

@app.delete("/api/cells/{cell_id}")
async def delete_cell(cell_id: str):
    """Delete a cell"""
    global notebook_data
    notebook_data["cells"] = [cell for cell in notebook_data["cells"] if cell["id"] != cell_id]
    return {"status": "success", "message": "Cell deleted"}

@app.post("/api/kernel/initialize")
async def initialize_kernel(
    gpu_config: str = "cpu",
    cpu_cores: float = 4.0,
    memory_mb: int = 8192,
    timeout: int = 300
):
    """Initialize the Modal kernel"""
    global kernel
    try:
        if kernel is None:
            kernel = ModalNotebookKernel(
                gpu_config=gpu_config, 
                cpu_cores=cpu_cores, 
                memory_mb=memory_mb, 
                timeout=timeout
            )
            return {"status": "success", "message": f"✅ Kernel initialized with {gpu_config} GPU, {cpu_cores} CPU cores, {memory_mb}MB RAM"}
        else:
            return {"status": "success", "message": "✅ Kernel already initialized"}
    except Exception as e:
        return {"status": "error", "message": f"❌ Error initializing kernel: {str(e)}"}

@app.post("/api/cells/{cell_id}/execute")
async def execute_cell(cell_id: str, execute_data: CellExecute):
    """Execute a code cell"""
    global kernel
    
    if kernel is None:
        raise HTTPException(status_code=400, detail="Kernel not initialized")
    
    try:
        # Update cell content
        for cell in notebook_data["cells"]:
            if cell["id"] == cell_id:
                cell["source"] = execute_data.code.split('\n')
                break
        
        # Execute code
        result = kernel.execute_cell(execute_data.code)
        
        # Create outputs
        outputs = []
        if result["status"] == "success":
            if result.get("stdout"):
                outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": result["stdout"]
                })
            
            # Add plots
            for plot_base64 in result.get("plots", []):
                outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": plot_base64
                    },
                    "metadata": {}
                })
        
        elif result["status"] == "error":
            error_info = result["error"]
            outputs.append({
                "output_type": "error",
                "ename": error_info["name"],
                "evalue": error_info["value"],
                "traceback": error_info["traceback"].split('\n')
            })
        
        # Update cell outputs and execution count
        for cell in notebook_data["cells"]:
            if cell["id"] == cell_id:
                cell["outputs"] = outputs
                cell["execution_count"] = kernel.execution_count
                break
        
        return {"status": "success", "outputs": outputs}
        
    except Exception as e:
        logger.error(f"Error executing cell: {e}")
        return {"status": "error", "message": f"❌ Execution error: {str(e)}"}

@app.get("/api/kernel/status")
async def get_kernel_status():
    """Get kernel status"""
    return {
        "initialized": kernel is not None,
        "execution_count": kernel.execution_count if kernel else 0
    }

@app.post("/api/kernel/shutdown")
async def shutdown_kernel():
    """Shutdown the kernel"""
    global kernel
    if kernel:
        kernel.shutdown()
        kernel = None
        return {"status": "success", "message": "Kernel shutdown complete"}
    else:
        return {"status": "success", "message": "Kernel was not running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)