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
import os
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
# Modal handles authentication automatically

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
        self.execution_count = 0
        self.session_id = str(uuid.uuid4())
        self.session_state = {}  # Initialize session state
        
        # Create Modal image
        self.image = self._create_image()
        self._setup_modal_environment()
    
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
    
    def _setup_modal_environment(self):
        """Setup Modal environment for fast execution"""
        try:
            # Create a simple Modal function that can be called remotely
            # This approach is more reliable than complex app definitions
            
            # Create a simple execution script
            self._execution_script = f'''
import sys
import json
import traceback
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global namespace for session state
_global_namespace = {{
    '__builtins__': __builtins__,
    '__name__': '__main__',
    '__doc__': None,
    '__package__': None
}}

def execute_code(code, session_state=None):
    """Execute Python code with session state"""
    global _global_namespace
    
    if session_state:
        _global_namespace.update(session_state)
    
    captured_figures = []
    
    def _capture_show(*args, **kwargs):
        nonlocal captured_figures
        try:
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                captured_figures.append(img_base64)
                buf.close()
                plt.close(fig)
        except Exception as e:
            print(f"Error capturing plot: {{e}}", file=sys.stderr)
    
    plt.show = _capture_show
    
    try:
        # Check if code contains shell commands
        lines = code.strip().split('\\n')
        shell_commands = []
        python_code_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('!'):
                shell_cmd = stripped_line[1:].strip()
                shell_commands.append(shell_cmd)
            else:
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
                    timeout=60
                )
                
                if result.stdout:
                    stdout_parts.append(f"$ {{shell_cmd}}")
                    stdout_parts.append(result.stdout.rstrip())
                
                if result.stderr:
                    stderr_parts.append(f"$ {{shell_cmd}}")
                    stderr_parts.append(result.stderr.rstrip())
                
                if result.returncode != 0:
                    stderr_parts.append(f"Command exited with code {{result.returncode}}")
                    
            except subprocess.TimeoutExpired:
                stderr_parts.append(f"$ {{shell_cmd}}")
                stderr_parts.append("Command timed out after 60 seconds")
            except Exception as e:
                stderr_parts.append(f"$ {{shell_cmd}}")
                stderr_parts.append(f"Error executing shell command: {{str(e)}}")
        
        # Execute Python code if present
        python_stdout = ""
        if python_code_lines and any(line.strip() for line in python_code_lines):
            python_code = '\\n'.join(python_code_lines)
            
            # Capture stdout during Python execution
            import io
            from contextlib import redirect_stdout
            
            stdout_buffer = io.StringIO()
            
            with redirect_stdout(stdout_buffer):
                exec(python_code, _global_namespace)
            
            python_stdout = stdout_buffer.getvalue()
        
        # Combine all stdout
        all_stdout_parts = stdout_parts.copy()
        if python_stdout:
            all_stdout_parts.append(python_stdout.rstrip())
        
        stdout_output = '\\n'.join(all_stdout_parts) if all_stdout_parts else ""
        stderr_output = '\\n'.join(stderr_parts) if stderr_parts else ""
        
        return {{
            "status": "success",
            "stdout": stdout_output,
            "stderr": stderr_output,
            "plots": captured_figures,
            "session_state": dict(_global_namespace)
        }}
        
    except Exception as e:
        return {{
            "status": "error",
            "error": {{
                "name": type(e).__name__,
                "value": str(e),
                "traceback": traceback.format_exc()
            }},
            "session_state": dict(_global_namespace)
        }}
'''
            
            logger.info(f"Modal execution environment created with {self.gpu_config} GPU")
            
        except Exception as e:
            logger.error(f"Error creating Modal environment: {e}")
            raise
    
    def execute_cell(self, code: str) -> Dict[str, Any]:
        """Execute a code cell using Modal for fast execution"""
        self.execution_count += 1
        logger.info(f"Executing cell #{self.execution_count} with Modal")
        
        try:
            # Use Modal's simple execution approach
            # Create a temporary execution environment with all necessary modules
            import sys
            import io
            import base64
            import json
            import traceback
            
            exec_globals = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
                '__doc__': None,
                '__package__': None,
                'sys': sys,
                'io': io,
                'base64': base64,
                'json': json,
                'traceback': traceback
            }
            
            # Update with session state
            exec_globals.update(self.session_state)
            
            # Comprehensive output capture system
            captured_outputs = []
            
            def _capture_output(output_type, data, metadata=None):
                """Capture any type of output for display"""
                nonlocal captured_outputs
                if metadata is None:
                    metadata = {}
                captured_outputs.append({
                    "output_type": output_type,
                    "data": data,
                    "metadata": metadata
                })
            
            def _capture_show(*args, **kwargs):
                """Capture matplotlib plots"""
                try:
                    import matplotlib.pyplot as plt
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        _capture_output("display_data", {
                            "image/png": img_base64
                        }, {"output_type": "display_data"})
                        buf.close()
                        plt.close(fig)
                except Exception as e:
                    print(f"Error capturing matplotlib plot: {e}", file=sys.stderr)
            
            # Set up matplotlib
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.show = _capture_show
            
            # Check if code contains shell commands
            lines = code.strip().split('\n')
            shell_commands = []
            python_code_lines = []
            
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith('!'):
                    shell_cmd = stripped_line[1:].strip()
                    shell_commands.append(shell_cmd)
                else:
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
                        timeout=60
                    )
                    
                    if result.stdout:
                        stdout_parts.append(f"$ {shell_cmd}")
                        stdout_parts.append(result.stdout.rstrip())
                    
                    if result.stderr:
                        stderr_parts.append(f"$ {shell_cmd}")
                        stderr_parts.append(result.stderr.rstrip())
                    
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
                python_code = '\n'.join(python_code_lines)
                
                # Add comprehensive output capture code
                output_capture_code = '''
# Comprehensive output capture system for Jupyter-like behavior
import sys
import io
import base64
import json
import traceback

# Global variables for output capture
_captured_outputs = []

def _capture_output(output_type, data, metadata=None):
    """Capture any type of output for display"""
    global _captured_outputs
    if metadata is None:
        metadata = {}
    
    if output_type == "display_data":
        _captured_outputs.append({
            "output_type": "display_data",
            "data": data,
            "metadata": metadata
        })
    elif output_type == "execute_result":
        _captured_outputs.append({
            "output_type": "execute_result",
            "execution_count": 1,
            "data": data,
            "metadata": metadata
        })
    else:
        _captured_outputs.append({
            "output_type": output_type,
            "data": data,
            "metadata": metadata
        })

# Plotly capture
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    
    def _capture_plotly_inline(fig):
        """Capture Plotly figure and store for inline display"""
        try:
            # Convert to interactive HTML instead of static PNG
            html_str = pio.to_html(fig, include_plotlyjs=True, div_id="plotly-div")
            _capture_output("display_data", {
                "text/html": html_str
            }, {"output_type": "display_data"})
        except Exception as e:
            print(f"Error capturing Plotly figure: {e}")
    
    # Patch the show method
    original_show = go.Figure.show
    def patched_show(self, *args, **kwargs):
        _capture_plotly_inline(self)
        # Don't actually show in browser
        return None
    go.Figure.show = patched_show
    
except ImportError:
    pass  # Plotly not available

# Matplotlib capture
try:
    import matplotlib.pyplot as plt
    
    def _capture_matplotlib_show(*args, **kwargs):
        """Capture matplotlib plots"""
        try:
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                
                # Capture as PNG for static display
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                # Also try to capture as SVG for better quality
                svg_buf = io.BytesIO()
                fig.savefig(svg_buf, format='svg', bbox_inches='tight')
                svg_buf.seek(0)
                svg_content = svg_buf.getvalue().decode('utf-8')
                
                _capture_output("display_data", {
                    "image/png": img_base64,
                    "image/svg+xml": svg_content
                }, {"output_type": "display_data"})
                
                buf.close()
                svg_buf.close()
                plt.close(fig)
        except Exception as e:
            print(f"Error capturing matplotlib plot: {e}")
    
    plt.show = _capture_matplotlib_show
    
except ImportError:
    pass  # Matplotlib not available

# Pandas DataFrame capture
try:
    import pandas as pd
    
    def _capture_dataframe(df):
        """Capture pandas DataFrame for rich display"""
        try:
            # Convert DataFrame to HTML
            html_str = df.to_html(classes='table table-striped', table_id='dataframe')
            _capture_output("display_data", {
                "text/html": html_str
            }, {"output_type": "display_data"})
        except Exception as e:
            print(f"Error capturing DataFrame: {e}")
    
    # Monkey patch pandas DataFrame display
    original_repr_html = pd.DataFrame._repr_html_
    def patched_repr_html(self):
        _capture_dataframe(self)
        return original_repr_html(self)
    pd.DataFrame._repr_html_ = patched_repr_html
    
except ImportError:
    pass  # Pandas not available

# Rich display for other objects
def _capture_rich_output(obj):
    """Capture rich output from objects that support it"""
    try:
        # Check if object has _repr_html_ method
        if hasattr(obj, '_repr_html_'):
            html_str = obj._repr_html_()
            if html_str:
                _capture_output("display_data", {
                    "text/html": html_str
                }, {"output_type": "display_data"})
                return True
        
        # Check if object has _repr_svg_ method
        if hasattr(obj, '_repr_svg_'):
            svg_str = obj._repr_svg_()
            if svg_str:
                _capture_output("display_data", {
                    "image/svg+xml": svg_str
                }, {"output_type": "display_data"})
                return True
                
        # Check if object has _repr_png_ method
        if hasattr(obj, '_repr_png_'):
            png_bytes = obj._repr_png_()
            if png_bytes:
                import base64
                png_base64 = base64.b64encode(png_bytes).decode('utf-8')
                _capture_output("display_data", {
                    "image/png": png_base64
                }, {"output_type": "display_data"})
                return True
                
        # Check if object has _repr_jpeg_ method
        if hasattr(obj, '_repr_jpeg_'):
            jpeg_bytes = obj._repr_jpeg_()
            if jpeg_bytes:
                import base64
                jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
                _capture_output("display_data", {
                    "image/jpeg": jpeg_base64
                }, {"output_type": "display_data"})
                return True
                
        # Check if object has _repr_pdf_ method
        if hasattr(obj, '_repr_pdf_'):
            pdf_bytes = obj._repr_pdf_()
            if pdf_bytes:
                import base64
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                _capture_output("display_data", {
                    "application/pdf": pdf_base64
                }, {"output_type": "display_data"})
                return True
                
        # Check if object has _repr_latex_ method
        if hasattr(obj, '_repr_latex_'):
            latex_str = obj._repr_latex_()
            if latex_str:
                _capture_output("display_data", {
                    "text/latex": latex_str
                }, {"output_type": "display_data"})
                return True
                
        # Check if object has _repr_json_ method
        if hasattr(obj, '_repr_json_'):
            json_str = obj._repr_json_()
            if json_str:
                _capture_output("display_data", {
                    "application/json": json_str
                }, {"output_type": "display_data"})
                return True
                
        # Check if object has _repr_javascript_ method
        if hasattr(obj, '_repr_javascript_'):
            js_str = obj._repr_javascript_()
            if js_str:
                _capture_output("display_data", {
                    "application/javascript": js_str
                }, {"output_type": "display_data"})
                return True
                
    except Exception as e:
        print(f"Error capturing rich output: {e}")
    
    return False
'''
                
                # Combine the output capture code with the user's code
                # Ensure proper indentation by stripping and re-indenting
                # Strip any leading/trailing whitespace from user code
                clean_user_code = python_code.strip()
                full_code = output_capture_code + '\n\n# User code starts here\n' + clean_user_code
                
                # Execute the code with stdout capture
                from contextlib import redirect_stdout, redirect_stderr
                import io
                
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                
                try:
                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        # Execute the code and capture the result
                        result = exec(full_code, exec_globals)
                        
                        # If the last line is an expression (not an assignment), capture its value
                        lines = python_code.strip().split('\n')
                        if lines and not lines[-1].strip().startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'finally:', 'with ', 'return ', 'yield ', 'pass', 'break', 'continue', 'raise ', 'assert ', 'del ', 'global ', 'nonlocal ')):
                            last_line = lines[-1].strip()
                            if last_line and not '=' in last_line and not last_line.startswith('#'):
                                try:
                                    # Try to evaluate the last line as an expression
                                    expr_result = eval(last_line, exec_globals)
                                    if expr_result is not None:
                                        # Capture the result
                                        if hasattr(expr_result, '_repr_html_'):
                                            # Rich HTML display
                                            html_str = expr_result._repr_html_()
                                            if html_str:
                                                captured_outputs.append({
                                                    "output_type": "execute_result",
                                                    "execution_count": 1,
                                                    "data": {
                                                        "text/html": html_str
                                                    },
                                                    "metadata": {}
                                                })
                                        elif hasattr(expr_result, '_repr_svg_'):
                                            # SVG display
                                            svg_str = expr_result._repr_svg_()
                                            if svg_str:
                                                captured_outputs.append({
                                                    "output_type": "execute_result",
                                                    "execution_count": 1,
                                                    "data": {
                                                        "image/svg+xml": svg_str
                                                    },
                                                    "metadata": {}
                                                })
                                        else:
                                            # Plain text display
                                            captured_outputs.append({
                                                "output_type": "execute_result",
                                                "execution_count": 1,
                                                "data": {
                                                    "text/plain": str(expr_result)
                                                },
                                                "metadata": {}
                                            })
                                except:
                                    pass  # Ignore evaluation errors
                except Exception as e:
                    # Capture execution errors
                    error_traceback = traceback.format_exc()
                    captured_outputs.append({
                        "output_type": "error",
                        "ename": type(e).__name__,
                        "evalue": str(e),
                        "traceback": error_traceback.split('\n')
                    })
                
                # Get captured outputs
                if '_captured_outputs' in exec_globals:
                    captured_outputs.extend(exec_globals['_captured_outputs'])
                
                # Get stdout/stderr
                python_stdout = stdout_buffer.getvalue()
                stderr_output = stderr_buffer.getvalue()
                
                # Add stdout as stream output if not empty
                if python_stdout.strip():
                    captured_outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": python_stdout
                    })
                
                # Add stderr as stream output if not empty
                if stderr_output.strip():
                    captured_outputs.append({
                        "output_type": "stream",
                        "name": "stderr", 
                        "text": stderr_output
                    })
            
            # Combine all stdout
            all_stdout_parts = stdout_parts.copy()
            if python_stdout:
                all_stdout_parts.append(python_stdout.rstrip())
            
            stdout_output = '\n'.join(all_stdout_parts) if all_stdout_parts else ""
            stderr_output = '\n'.join(stderr_parts) if stderr_parts else ""
            
            # Update session state
            self.session_state = dict(exec_globals)
            
            return {
                "status": "success",
                "stdout": stdout_output,
                "stderr": stderr_output,
                "outputs": captured_outputs
            }
            
        except Exception as e:
            logger.error(f"Error executing cell: {e}")
            return {
                "status": "error",
                "error": {
                    "name": type(e).__name__,
                    "value": str(e),
                    "traceback": str(e)
                }
            }
    
    def shutdown(self):
        """Shutdown the Modal app"""
        try:
            if hasattr(self, 'app'):
                # Modal apps are automatically cleaned up
                logger.info("Modal app shutdown complete")
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global notebook instance
notebook_data = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
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
    
    logger.info(f"Executing cell {cell_id} with code: {execute_data.code[:100]}...")
    
    if kernel is None:
        logger.error("Kernel not initialized")
        raise HTTPException(status_code=400, detail="Kernel not initialized")
    
    try:
        # Update cell content
        for cell in notebook_data["cells"]:
            if cell["id"] == cell_id:
                cell["source"] = execute_data.code.split('\n')
                break
        
        # Execute code
        result = kernel.execute_cell(execute_data.code)
        
        logger.info(f"Execution result: {result}")
        
        # Create outputs
        outputs = []
        if result["status"] == "success":
            # Use the outputs directly from the result
            outputs = result.get("outputs", [])
            logger.info(f"Captured outputs: {len(outputs)} outputs")
            for i, output in enumerate(outputs):
                logger.info(f"Output {i+1}: {output.get('output_type', 'unknown')}")
        
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