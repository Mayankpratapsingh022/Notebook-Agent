# Notebook Agent

A full-stack Jupyter-style notebook applicatio, built with Next.js frontend and FastAPI backend.


## Architecture

```
Notebook Agent/
├── FrontEnd/                 # Next.js frontend
│   └── my-app/
│       ├── app/
│       │   ├── page.tsx      # Main page
│       │   └── notebook/     # Notebook page
│       └── package.json
├── Backend/                  # FastAPI backend
│   ├── main.py              # FastAPI application
│   ├── requirements.txt      # Python dependencies
│   └── start_backend.sh     # Startup script
└── README.md
```

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Modal account and CLI setup

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd Backend
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Modal:**
   ```bash
   modal token set
   ```

5. **Start the backend server:**
   ```bash
   python main.py
   # Or use the startup script:
   chmod +x start_backend.sh
   ./start_backend.sh
   ```

   The backend will be available at `http://localhost:8000`
   API documentation at `http://localhost:8000/docs`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd FrontEnd/my-app
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:3000`

## Usage

1. **Open the application** in your browser at `http://localhost:3000`
2. **Initialize the kernel** by clicking the "Initialize Kernel" button
3. **Add cells** by hovering between existing cells and clicking "+ Code" or "+ Markdown"
4. **Write code** in code cells and click the play button to execute
5. **View outputs** including text, plots, and error messages
6. **Drag and drop** cells to reorder them
7. **Use keyboard shortcuts** like Shift+Enter to run cells

## API Endpoints

### Notebook Management
- `GET /api/notebook` - Get current notebook data
- `POST /api/notebook` - Update entire notebook
- `POST /api/cells` - Create new cell
- `PUT /api/cells/{cell_id}` - Update cell
- `DELETE /api/cells/{cell_id}` - Delete cell

### Kernel Management
- `POST /api/kernel/initialize` - Initialize Modal kernel
- `GET /api/kernel/status` - Get kernel status
- `POST /api/kernel/shutdown` - Shutdown kernel

### Cell Execution
- `POST /api/cells/{cell_id}/execute` - Execute cell code

## Configuration

### Backend Configuration

The backend supports various configuration options:

```python
# GPU Configuration
gpu_config: "cpu" | "T4" | "A100"

# Resource Allocation
cpu_cores: float = 4.0
memory_mb: int = 8192
timeout: int = 300
```

### Frontend Configuration

Environment variables for the frontend:

```bash
# API Base URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Environment
NODE_ENV=development
```

## Development

### Backend Development

```bash
cd Backend
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Frontend Development

```bash
cd FrontEnd/my-app
npm install
npm run dev
```

### Testing

```bash
# Backend tests
cd Backend
python -m pytest

# Frontend tests
cd FrontEnd/my-app
npm test
```

## Deployment

### Backend Deployment

1. **Deploy to Modal:**
   ```bash
   modal deploy main.py
   ```

2. **Or deploy to cloud provider:**
   ```bash
   # Using Docker
   docker build -t notebook-backend .
   docker run -p 8000:8000 notebook-backend
   ```

### Frontend Deployment

1. **Build for production:**
   ```bash
   cd FrontEnd/my-app
   npm run build
   ```

2. **Deploy to Vercel:**
   ```bash
   vercel --prod
   ```

## Troubleshooting

### Common Issues

1. **Modal authentication:**
   ```bash
   modal token set
   ```

2. **Port conflicts:**
   - Backend: Change port in `main.py`
   - Frontend: Change port in `package.json`

3. **CORS issues:**
   - Update CORS settings in `main.py`

4. **Kernel initialization fails:**
   - Check Modal credentials
   - Verify network connectivity
   - Check resource limits

### Debug Mode

Enable debug logging:

```python
# Backend
import logging
logging.basicConfig(level=logging.DEBUG)
```

```bash
# Frontend
NODE_ENV=development npm run dev
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API docs at `/docs` endpoint
