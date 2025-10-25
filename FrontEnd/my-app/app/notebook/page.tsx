'use client';

import React, {
    Dispatch,
    SetStateAction,
    useState,
    DragEvent,
    FormEvent,
    useRef,
  } from "react";
  import { FiPlus, FiTrash, FiPlay, FiMoreVertical, FiMove, FiChevronUp, FiChevronDown } from "react-icons/fi";
  import { motion } from "framer-motion";
  import { FaFire } from "react-icons/fa";
  
  // Notebook cell types and nbformat structure
  type CellType = "code" | "markdown";
  
  interface NotebookCell {
    id: string;
    cell_type: CellType;
    source: string[];
    metadata: {
      collapsed?: boolean;
      execution_count?: number;
      [key: string]: any;
    };
    outputs?: any[];
    execution_count?: number;
  }
  
  interface NotebookData {
    cells: NotebookCell[];
    metadata: {
      kernelspec: {
        display_name: string;
        language: string;
        name: string;
      };
      language_info: {
        name: string;
        version: string;
      };
    };
    nbformat: number;
    nbformat_minor: number;
  }
  
  export default function NotebookPage() {
    return (
      <div className="h-screen w-full bg-neutral-900 text-neutral-50">
        <Notebook />
      </div>
    );
  }
  
  const Notebook = () => {
    const [notebookData, setNotebookData] = useState<NotebookData>({
      cells: [],
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3"
        },
        language_info: {
          name: "python",
          version: "3.8.0"
        }
      },
      nbformat: 4,
      nbformat_minor: 4
    });

    const [activeCell, setActiveCell] = useState<string | null>(null);
    const [kernelStatus, setKernelStatus] = useState<{initialized: boolean, execution_count: number}>({
      initialized: false,
      execution_count: 0
    });
    const [isLoading, setIsLoading] = useState(false);

    // API base URL
    const API_BASE = process.env.NODE_ENV === 'production' 
      ? 'https://your-backend-url.com' 
      : 'http://localhost:8000';

    // Load notebook data on mount
    React.useEffect(() => {
      loadNotebook();
      checkKernelStatus();
    }, []);

    const loadNotebook = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/notebook`);
        if (response.ok) {
          const data = await response.json();
          setNotebookData(data);
        }
      } catch (error) {
        console.error('Error loading notebook:', error);
      }
    };

    const checkKernelStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/kernel/status`);
        if (response.ok) {
          const status = await response.json();
          setKernelStatus(status);
        }
      } catch (error) {
        console.error('Error checking kernel status:', error);
      }
    };

    const initializeKernel = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_BASE}/api/kernel/initialize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            gpu_config: 'cpu',
            cpu_cores: 4.0,
            memory_mb: 8192,
            timeout: 300
          })
        });
        
        if (response.ok) {
          const result = await response.json();
          console.log(result.message);
          await checkKernelStatus();
        }
      } catch (error) {
        console.error('Error initializing kernel:', error);
      } finally {
        setIsLoading(false);
      }
    };

    const addCell = async (index: number, cellType: CellType) => {
      try {
        const response = await fetch(`${API_BASE}/api/cells`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            cell_type: cellType,
            source: cellType === "code" ? ["# Write your code here"] : ["# Write your markdown here"],
            metadata: {}
          })
        });

        if (response.ok) {
          const result = await response.json();
          const newCell: NotebookCell = {
            id: result.cell_id,
            cell_type: cellType,
            source: cellType === "code" ? ["# Write your code here"] : ["# Write your markdown here"],
            metadata: {},
            execution_count: undefined
          };

          const newCells = [...notebookData.cells];
          newCells.splice(index, 0, newCell);
          
          setNotebookData({
            ...notebookData,
            cells: newCells
          });
        }
      } catch (error) {
        console.error('Error adding cell:', error);
      }
    };

    const moveCellUp = (cellId: string) => {
      const cellIndex = notebookData.cells.findIndex(cell => cell.id === cellId);
      if (cellIndex > 0) {
        const newCells = [...notebookData.cells];
        [newCells[cellIndex - 1], newCells[cellIndex]] = [newCells[cellIndex], newCells[cellIndex - 1]];
        setNotebookData({
          ...notebookData,
          cells: newCells
        });
      }
    };

    const moveCellDown = (cellId: string) => {
      const cellIndex = notebookData.cells.findIndex(cell => cell.id === cellId);
      if (cellIndex < notebookData.cells.length - 1) {
        const newCells = [...notebookData.cells];
        [newCells[cellIndex], newCells[cellIndex + 1]] = [newCells[cellIndex + 1], newCells[cellIndex]];
        setNotebookData({
          ...notebookData,
          cells: newCells
        });
      }
    };

    const deleteCell = async (cellId: string) => {
      try {
        const response = await fetch(`${API_BASE}/api/cells/${cellId}`, {
          method: 'DELETE'
        });

        if (response.ok) {
          setNotebookData({
            ...notebookData,
            cells: notebookData.cells.filter(cell => cell.id !== cellId)
          });
        }
      } catch (error) {
        console.error('Error deleting cell:', error);
      }
    };

    const updateCellSource = async (cellId: string, newSource: string[]) => {
      try {
        const response = await fetch(`${API_BASE}/api/cells/${cellId}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            source: newSource,
            metadata: {}
          })
        });

        if (response.ok) {
          setNotebookData({
            ...notebookData,
            cells: notebookData.cells.map(cell => 
              cell.id === cellId ? { ...cell, source: newSource } : cell
            )
          });
        }
      } catch (error) {
        console.error('Error updating cell:', error);
      }
    };

    const handleDragStart = (e: DragEvent, cellId: string) => {
      e.dataTransfer.setData("cellId", cellId);
      setActiveCell(cellId);
    };

    const handleDragEnd = (e: DragEvent) => {
      const cellId = e.dataTransfer.getData("cellId");
      setActiveCell(null);
      clearHighlights();

      const indicators = getIndicators();
      const { element } = getNearestIndicator(e, indicators);

      const before = element.dataset.before || "-1";

      if (before !== cellId) {
        let copy = [...notebookData.cells];
        let cellToTransfer = copy.find((c) => c.id === cellId);
        if (!cellToTransfer) return;

        copy = copy.filter((c) => c.id !== cellId);

        const moveToBack = before === "-1";

        if (moveToBack) {
          copy.push(cellToTransfer);
        } else {
          const insertAtIndex = copy.findIndex((el) => el.id === before);
          if (insertAtIndex === undefined) return;
          copy.splice(insertAtIndex, 0, cellToTransfer);
        }

        setNotebookData({
          ...notebookData,
          cells: copy
        });
      }
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      highlightIndicator(e);
    };

    const clearHighlights = (els?: HTMLElement[]) => {
      const indicators = els || getIndicators();
      indicators.forEach((i) => {
        i.style.opacity = "0";
      });
    };

    const highlightIndicator = (e: DragEvent) => {
      const indicators = getIndicators();
      clearHighlights(indicators);
      const el = getNearestIndicator(e, indicators);
      el.element.style.opacity = "1";
    };

    const getNearestIndicator = (e: DragEvent, indicators: HTMLElement[]) => {
      const DISTANCE_OFFSET = 50;
      const el = indicators.reduce(
        (closest, child) => {
          const box = child.getBoundingClientRect();
          const offset = e.clientY - (box.top + DISTANCE_OFFSET);
          if (offset < 0 && offset > closest.offset) {
            return { offset: offset, element: child };
          } else {
            return closest;
          }
        },
        {
          offset: Number.NEGATIVE_INFINITY,
          element: indicators[indicators.length - 1],
        }
      );
      return el;
    };

    const getIndicators = () => {
      return Array.from(
        document.querySelectorAll("[data-cell-indicator]") as unknown as HTMLElement[]
      );
    };

    const executeCell = async (cellId: string, code: string) => {
      if (!kernelStatus.initialized) {
        console.log('Kernel not initialized. Please initialize kernel first.');
        return;
      }

      try {
        const response = await fetch(`${API_BASE}/api/cells/${cellId}/execute`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            cell_id: cellId,
            code: code
          })
        });

        if (response.ok) {
          const result = await response.json();
          console.log('Execution result:', result);
          console.log('Outputs:', result.outputs);
          console.log('Number of outputs:', result.outputs?.length);
          
          // Update cell with outputs
          setNotebookData(prev => {
            const updated = {
              ...prev,
              cells: prev.cells.map(cell => 
                cell.id === cellId 
                  ? { 
                      ...cell, 
                      outputs: result.outputs,
                      execution_count: kernelStatus.execution_count + 1
                    } 
                  : cell
              )
            };
            console.log('Updated notebook data:', updated);
            return updated;
          });

          // Update kernel status
          await checkKernelStatus();
        }
      } catch (error) {
        console.error('Error executing cell:', error);
      }
    };

    return (
      <div className="flex h-full w-full overflow-y-auto">
        <div className="w-full max-w-4xl mx-auto p-6">
          {/* Kernel Status and Controls */}
          <div className="mb-6 p-4 bg-neutral-800 rounded-lg border border-neutral-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <span className="text-sm text-neutral-400">
                  Kernel Status: {kernelStatus.initialized ? '✅ Initialized' : '❌ Not Initialized'}
                </span>
                <span className="text-sm text-neutral-400">
                  Execution Count: {kernelStatus.execution_count}
                </span>
              </div>
              <button
                onClick={initializeKernel}
                disabled={isLoading || kernelStatus.initialized}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-600 disabled:cursor-not-allowed text-white rounded-lg text-sm transition-colors"
              >
                {isLoading ? 'Initializing...' : kernelStatus.initialized ? 'Kernel Ready' : 'Initialize Kernel'}
              </button>
            </div>
          </div>

          <div className="space-y-1">
            {notebookData.cells.map((cell, index) => (
              <div key={cell.id}>
                {/* Hover zone for inserting cells above */}
                <CellInsertionZone 
                  onAddCode={() => addCell(index, "code")}
                  onAddMarkdown={() => addCell(index, "markdown")}
                />
                
                <CellContainer
                  cell={cell}
                  index={index}
                  activeCell={activeCell}
                  kernelInitialized={kernelStatus.initialized}
                  onDelete={() => deleteCell(cell.id)}
                  onUpdateSource={(newSource) => updateCellSource(cell.id, newSource)}
                  onExecute={(code) => executeCell(cell.id, code)}
                  onMoveUp={() => moveCellUp(cell.id)}
                  onMoveDown={() => moveCellDown(cell.id)}
                  onDragStart={handleDragStart}
                  onDragEnd={handleDragEnd}
                  onDragOver={handleDragOver}
                />
              </div>
            ))}
            <DropIndicator beforeId={null} />
            <CellInsertionZone 
              onAddCode={() => addCell(notebookData.cells.length, "code")}
              onAddMarkdown={() => addCell(notebookData.cells.length, "markdown")}
              isEndZone={true}
            />
          </div>
        </div>
      </div>
    );
  };
  
  // Cell Container Component
  interface CellContainerProps {
    cell: NotebookCell;
    index: number;
    activeCell: string | null;
    kernelInitialized: boolean;
    onDelete: () => void;
    onUpdateSource: (newSource: string[]) => void;
    onExecute: (code: string) => void;
    onMoveUp: () => void;
    onMoveDown: () => void;
    onDragStart: (e: DragEvent, cellId: string) => void;
    onDragEnd: (e: DragEvent) => void;
    onDragOver: (e: DragEvent) => void;
  }

  const CellContainer = ({ 
    cell, 
    index, 
    activeCell,
    kernelInitialized,
    onDelete, 
    onUpdateSource, 
    onExecute,
    onMoveUp,
    onMoveDown,
    onDragStart,
    onDragEnd,
    onDragOver
  }: CellContainerProps) => {
    const [isHovered, setIsHovered] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [localSource, setLocalSource] = useState(cell.source.join('\n'));
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleSourceChange = (newSource: string) => {
      setLocalSource(newSource);
      onUpdateSource(newSource.split('\n'));
      
      // Auto-resize textarea
      if (textareaRef.current) {
        const textarea = textareaRef.current;
        textarea.style.height = 'auto';
        textarea.style.height = `${Math.max(cell.cell_type === 'code' ? 40 : 20, textarea.scrollHeight)}px`;
      }
    };


    const handleRun = () => {
      if (kernelInitialized) {
        onExecute(localSource);
      } else {
        console.log('Kernel not initialized. Please initialize kernel first.');
      }
    };

    const isDragging = activeCell === cell.id;

    return (
      <>
        {/* Drop Indicator */}
        <DropIndicator beforeId={cell.id} />
        
        <motion.div 
          layout
          layoutId={cell.id}
          className="relative group"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          onDrop={onDragEnd}
          onDragOver={onDragOver}
          style={{
            opacity: isDragging ? 0.5 : 1,
            transform: isDragging ? 'rotate(5deg)' : 'rotate(0deg)',
          }}
        >
          {/* Drag Handle */}
          <div className="absolute left-0 top-0 h-full w-6 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <div 
              draggable
              onDragStart={(e) => onDragStart(e, cell.id)}
              className="cursor-grab active:cursor-grabbing p-1 hover:bg-neutral-700 rounded"
              title="Drag to reorder"
            >
              <FiMove className="w-3 h-3 text-neutral-400" />
            </div>
          </div>

          {/* Cell Content */}
          <div className="ml-6 border border-neutral-700 bg-neutral-800 rounded-lg overflow-hidden">
          {/* Cell Header */}
          <div className="flex items-center justify-between px-3 py-1 bg-neutral-700/50 border-b border-neutral-600">
              <div className="flex items-center gap-2">
                <span className="text-xs text-neutral-400">
                  {cell.cell_type === 'code' ? 'Code' : 'Markdown'}
                </span>
                {cell.execution_count && (
                  <span className="text-xs text-neutral-400">
                    [{cell.execution_count}]
                  </span>
                )}
              </div>
              
              {/* Cell Controls */}
              <div className={`flex items-center gap-1 transition-opacity ${isHovered ? 'opacity-100' : 'opacity-0'}`}>
                <button
                  onClick={onMoveUp}
                  className="p-1 hover:bg-neutral-600 rounded text-blue-400"
                  title="Move cell up"
                >
                  <FiChevronUp className="w-4 h-4" />
                </button>
                <button
                  onClick={onMoveDown}
                  className="p-1 hover:bg-neutral-600 rounded text-blue-400"
                  title="Move cell down"
                >
                  <FiChevronDown className="w-4 h-4" />
                </button>
                {cell.cell_type === 'code' && (
                  <button
                    onClick={handleRun}
                    className="p-1 hover:bg-neutral-600 rounded text-green-400"
                    title="Run cell"
                  >
                    <FiPlay className="w-4 h-4" />
                  </button>
                )}
                <button
                  onClick={onDelete}
                  className="p-1 hover:bg-neutral-600 rounded text-red-400"
                  title="Delete cell"
                >
                  <FiTrash className="w-4 h-4" />
                </button>
                <button
                  className="p-1 hover:bg-neutral-600 rounded text-neutral-400"
                  title="More options"
                >
                  <FiMoreVertical className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Cell Body */}
            <div className="p-2">
              {cell.cell_type === 'markdown' && !isEditing ? (
                <div 
                  className="prose prose-invert max-w-none overflow-visible"
                  onClick={() => setIsEditing(true)}
                  style={{ height: 'auto', minHeight: 'auto' }}
                >
                  <div dangerouslySetInnerHTML={{ 
                    __html: cell.source.join('\n').replace(/\n/g, '<br>')
                  }} />
                </div>
              ) : (
                <div className="relative">
                  {/* Line Numbers */}
                  {cell.cell_type === 'code' && (
                    <div className="absolute left-0 top-0 h-full w-6 bg-neutral-700/30 border-r border-neutral-600 flex flex-col text-xs text-neutral-400 select-none">
                      {localSource.split('\n').map((_, index) => (
                        <div key={index} className="h-5 flex items-center justify-center">
                          {index + 1}
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {/* Code/Markdown Editor */}
                  <textarea
                    ref={textareaRef}
                    value={localSource}
                    onChange={(e) => handleSourceChange(e.target.value)}
                    onKeyDown={(e) => {
                      // Handle keyboard shortcuts
                      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                        e.preventDefault();
                        setIsEditing(false);
                        onUpdateSource(localSource.split('\n'));
                      }
                      if (e.key === 'Escape') {
                        setIsEditing(false);
                        setLocalSource(cell.source.join('\n'));
                      }
                      
                      // Auto-resize on Enter key for markdown
                      if (e.key === 'Enter') {
                        setTimeout(() => {
                          if (textareaRef.current) {
                            const textarea = textareaRef.current;
                            textarea.style.height = 'auto';
                            textarea.style.height = `${Math.max(cell.cell_type === 'code' ? 40 : 20, textarea.scrollHeight)}px`;
                          }
                        }, 0);
                      }
                    }}
                    onBlur={() => {
                      setIsEditing(false);
                      onUpdateSource(localSource.split('\n'));
                    }}
                    className={`w-full bg-transparent text-neutral-100 resize-none focus:outline-none overflow-hidden ${
                      cell.cell_type === 'code' ? 'pl-8' : ''
                    }`}
                    style={{ 
                      height: 'auto',
                      minHeight: `${Math.max(cell.cell_type === 'code' ? 40 : 20, localSource.split('\n').length * 20 + 10)}px`,
                      fontFamily: cell.cell_type === 'code' ? 'monospace' : 'inherit',
                      lineHeight: '20px'
                    }}
                    placeholder={cell.cell_type === 'code' ? 'Write your code here...' : 'Write your markdown here...'}
                    onInput={(e) => {
                      const target = e.target as HTMLTextAreaElement;
                      target.style.height = 'auto';
                      target.style.height = `${Math.max(cell.cell_type === 'code' ? 40 : 20, target.scrollHeight)}px`;
                    }}
                  />
                </div>
              )}
            </div>

            {/* Cell Outputs */}
            {cell.outputs && cell.outputs.length > 0 ? (
              <div className="mt-2 p-3 bg-neutral-900 border-t border-neutral-600">
                <div className="text-xs text-blue-400 mb-2">
                  📊 Outputs ({cell.outputs.length}): {cell.outputs.map(o => o.output_type).join(', ')}
                </div>
                {(() => {
                  console.log('Rendering outputs for cell:', cell.id, cell.outputs);
                  console.log('Cell has outputs:', cell.outputs.length, 'outputs');
                  console.log('Output types:', cell.outputs.map(o => o.output_type));
                  return null;
                })()}
                {cell.outputs.map((output, outputIndex) => (
                  <div key={outputIndex} className="mb-2 last:mb-0">
                    {/* Stream output (stdout/stderr) */}
                    {output.output_type === 'stream' && (
                      <div className="bg-neutral-800 p-2 rounded text-sm font-mono text-green-400 whitespace-pre-wrap">
                        {output.data?.text || output.text}
                      </div>
                    )}
                    
                    {/* Display data (images, plots, etc.) */}
                    {output.output_type === 'display_data' && output.data && (
                      <div className="bg-neutral-800 p-2 rounded">
                        {(() => {
                          console.log('Display data output:', output);
                          return null;
                        })()}
                        {/* Interactive HTML content (Plotly, etc.) */}
                        {output.data['text/html'] && (
                          <div>
                            {(() => {
                              console.log('Rendering HTML output:', output.data['text/html'].substring(0, 200) + '...');
                              return null;
                            })()}
                            <div 
                              dangerouslySetInnerHTML={{ 
                                __html: output.data['text/html'] 
                              }}
                              className="max-w-full"
                              style={{ width: '100%', height: 'auto' }}
                            />
                          </div>
                        )}
                        {/* SVG images (high quality vector graphics) */}
                        {output.data['image/svg+xml'] && !output.data['text/html'] && (
                          <div 
                            dangerouslySetInnerHTML={{ 
                              __html: output.data['image/svg+xml'] 
                            }}
                            className="max-w-full"
                          />
                        )}
                        {/* PNG images (fallback for static images) */}
                        {output.data['image/png'] && !output.data['text/html'] && !output.data['image/svg+xml'] && (
                          <img 
                            src={`data:image/png;base64,${output.data['image/png']}`} 
                            alt="Plot output" 
                            className="max-w-full h-auto rounded"
                          />
                        )}
                        {/* JPEG images */}
                        {output.data['image/jpeg'] && !output.data['text/html'] && !output.data['image/svg+xml'] && !output.data['image/png'] && (
                          <img 
                            src={`data:image/jpeg;base64,${output.data['image/jpeg']}`} 
                            alt="Image output" 
                            className="max-w-full h-auto rounded"
                          />
                        )}
                        {/* PDF documents */}
                        {output.data['application/pdf'] && (
                          <div className="bg-neutral-700 p-4 rounded border border-neutral-600">
                            <div className="flex items-center gap-2 mb-2">
                              <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                              </svg>
                              <span className="text-red-400 font-medium">PDF Document</span>
                            </div>
                            <p className="text-neutral-300 text-sm">PDF content is available but cannot be displayed inline. Download the file to view.</p>
                          </div>
                        )}
                        {/* LaTeX content */}
                        {output.data['text/latex'] && (
                          <div className="bg-neutral-700 p-4 rounded border border-neutral-600">
                            <div className="flex items-center gap-2 mb-2">
                              <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                              </svg>
                              <span className="text-blue-400 font-medium">LaTeX Content</span>
                            </div>
                            <pre className="text-neutral-300 font-mono text-sm whitespace-pre-wrap bg-neutral-800 p-2 rounded">
                              {output.data['text/latex']}
                            </pre>
                          </div>
                        )}
                        {/* JavaScript content */}
                        {output.data['application/javascript'] && (
                          <div className="bg-neutral-700 p-4 rounded border border-neutral-600">
                            <div className="flex items-center gap-2 mb-2">
                              <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                              </svg>
                              <span className="text-yellow-400 font-medium">JavaScript Code</span>
                            </div>
                            <pre className="text-neutral-300 font-mono text-sm whitespace-pre-wrap bg-neutral-800 p-2 rounded">
                              {output.data['application/javascript']}
                            </pre>
                          </div>
                        )}
                        {/* Plain text */}
                        {output.data['text/plain'] && (
                          <div className="text-green-400 font-mono text-sm whitespace-pre-wrap">
                            {output.data['text/plain']}
                          </div>
                        )}
                        {/* JSON data */}
                        {output.data['application/json'] && (
                          <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap bg-neutral-900 p-2 rounded">
                            {JSON.stringify(JSON.parse(output.data['application/json']), null, 2)}
                          </pre>
                        )}
                      </div>
                    )}
                    
                    {/* Error output */}
                    {output.output_type === 'error' && (
                      <div className="bg-red-900/20 border border-red-600 p-2 rounded text-sm">
                        <div className="text-red-400 font-semibold">
                          {output.data?.ename || output.ename}: {output.data?.evalue || output.evalue}
                        </div>
                        {(output.data?.traceback || output.traceback) && (
                          <div className="text-red-300 font-mono text-xs mt-1 whitespace-pre-wrap">
                            {(output.data?.traceback || output.traceback).join('\n')}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Execute result (for expressions that return values) */}
                    {output.output_type === 'execute_result' && output.data && (
                      <div className="bg-neutral-800 p-2 rounded text-sm">
                        {output.data['text/plain'] && (
                          <div className="text-blue-400 font-mono whitespace-pre-wrap">
                            {output.data['text/plain']}
                          </div>
                        )}
                        {output.data['text/html'] && (
                          <div 
                            dangerouslySetInnerHTML={{ 
                              __html: output.data['text/html'] 
                            }}
                            className="max-w-full"
                          />
                        )}
                        {output.data['image/svg+xml'] && (
                          <div 
                            dangerouslySetInnerHTML={{ 
                              __html: output.data['image/svg+xml'] 
                            }}
                            className="max-w-full"
                          />
                        )}
                        {output.data['image/png'] && (
                          <img 
                            src={`data:image/png;base64,${output.data['image/png']}`} 
                            alt="Result image" 
                            className="max-w-full h-auto rounded"
                          />
                        )}
                        {output.data['image/jpeg'] && (
                          <img 
                            src={`data:image/jpeg;base64,${output.data['image/jpeg']}`} 
                            alt="Result image" 
                            className="max-w-full h-auto rounded"
                          />
                        )}
                        {output.data['application/json'] && (
                          <pre className="text-blue-400 font-mono text-sm whitespace-pre-wrap bg-neutral-900 p-2 rounded">
                            {JSON.stringify(JSON.parse(output.data['application/json']), null, 2)}
                          </pre>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="mt-2 p-2 text-xs text-neutral-500">
                No outputs yet. Run the cell to see results.
              </div>
            )}
          </div>
        </motion.div>
      </>
    );
  };

  // Drop Indicator Component
  const DropIndicator = ({ beforeId }: { beforeId: string | null }) => {
    return (
      <div
        data-before={beforeId || "-1"}
        data-cell-indicator
        className="my-0.5 h-0.5 w-full bg-violet-400 opacity-0"
      />
    );
  };

  // Cell Insertion Zone
  interface CellInsertionZoneProps {
    onAddCode: () => void;
    onAddMarkdown: () => void;
    isEndZone?: boolean;
  }

  const CellInsertionZone = ({ onAddCode, onAddMarkdown, isEndZone = false }: CellInsertionZoneProps) => {
    const [isHovered, setIsHovered] = useState(false);

    if (isEndZone) {
      return (
        <div className="mt-6 flex justify-center">
          <button
            onClick={onAddCode}
            className="flex items-center gap-2 px-6 py-3 bg-neutral-700 hover:bg-neutral-600 border border-neutral-600 hover:border-neutral-500 rounded-lg text-neutral-200 transition-all duration-200 shadow-sm hover:shadow-md"
          >
            <FiPlus className="w-4 h-4" />
            <span className="font-medium">Add a cell</span>
          </button>
        </div>
      );
    }

    return (
      <div 
        className="relative py-1 group"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Invisible hover area */}
        <div className="absolute inset-0 -my-1 cursor-pointer" />
        
        {/* Visible insertion buttons */}
        <div className={`flex items-center justify-center gap-2 transition-all duration-200 ${isHovered ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
          <button
            onClick={onAddCode}
            className="flex items-center gap-1 px-2 py-1 bg-neutral-700 hover:bg-neutral-600 rounded text-xs text-neutral-200 transition-colors shadow-sm"
          >
            <FiPlus className="w-3 h-3" />
             Code
          </button>
          <button
            onClick={onAddMarkdown}
            className="flex items-center gap-1 px-2 py-1 bg-neutral-700 hover:bg-neutral-600 rounded text-xs text-neutral-200 transition-colors shadow-sm"
          >
            <FiPlus className="w-3 h-3" />
             Markdown
          </button>
        </div>
      </div>
    );
  };