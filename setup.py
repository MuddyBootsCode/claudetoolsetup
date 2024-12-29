#!/usr/bin/env python3
import subprocess
import os
import sys
import platform
from pathlib import Path
import shutil
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def find_uv_path():
    """Find the UV executable path in the system.

    This function searches for the UV executable in common installation locations
    and environment variables.

    Returns:
        str | None: Path to the UV executable if found, None otherwise.
    """
    # First check if UV_PATH environment variable is set
    if "UV_PATH" in os.environ and Path(os.environ["UV_PATH"]).exists():
        return os.environ["UV_PATH"]
    
    # Check if uv is in PATH
    uv_in_path = shutil.which('uv')
    if uv_in_path:
        return uv_in_path
    
    # Check common installation locations
    possible_locations = [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
        Path("/usr/bin/uv")
    ]
    
    # On Windows, add Windows-specific paths
    if os.name == 'nt':
        possible_locations.extend([
            Path(os.environ.get('LOCALAPPDATA', '')) / "Programs" / "uv" / "uv.exe",
            Path(os.environ.get('APPDATA', '')) / "uv" / "uv.exe"
        ])
    
    for path in possible_locations:
        if path.exists():
            return str(path)
    
    return None

def install_uv():
    """Install uv using the official install script.

    This function downloads and executes the official UV installer script.
    On Windows, it will exit with an error message since manual installation is required.

    Returns:
        str: Path to the installed UV executable.

    Raises:
        SystemExit: If installation fails or if running on Windows.
        subprocess.CalledProcessError: If any installation command fails.
    """
    logger.info("Installing uv...")
    if platform.system() == "Windows":
        logger.error("Please install uv manually from: https://github.com/astral-sh/uv")
        sys.exit(1)
    
    try:
        # Create a temporary file for the installer
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.sh', delete=False) as temp_file:
            # Download the installer
            subprocess.run([
                "curl", "-L", "-o", temp_file.name,
                "https://astral.sh/uv/install.sh"
            ], check=True)
            
            # Make the installer executable
            os.chmod(temp_file.name, 0o755)
            
            # Run the installer
            subprocess.run([temp_file.name], check=True)
            
            # Clean up
            os.unlink(temp_file.name)
        
        # Update PATH to include ~/.local/bin
        local_bin = str(Path.home() / ".local" / "bin")
        os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"
        
        # Verify installation
        uv_path = find_uv_path()
        if uv_path:
            logger.info("✓ uv installed successfully")
            return uv_path
        else:
            logger.error("Could not locate uv after installation.")
            logger.info(f"Please try adding {local_bin} to your PATH manually:")
            logger.info("    export PATH=\"$HOME/.local/bin:$PATH\"")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install uv: {e}")
        sys.exit(1)

def run_command(cmd, cwd=None):
    """Execute a command and handle errors.

    Args:
        cmd (str): The command to execute.
        cwd (str | Path | None): Working directory for command execution. Defaults to None.

    Returns:
        bool: True if command executed successfully, False otherwise.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing '{cmd}':")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def update_pyproject_toml():
    """Update or create pyproject.toml with configuration.

    Creates or overwrites the pyproject.toml file with predefined configuration
    including project metadata, dependencies, and build settings.

    Returns:
        None
    """
    content = """[project]
name = "weather"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28.1",
    "mcp>=1.1.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/weather"]

[project.scripts]
weather = "weather:main"
"""
    with open("pyproject.toml", "w") as f:
        f.write(content)
    logger.info("✓ Updated pyproject.toml configuration")

def setup_claude_config():
    """Set up the Claude desktop configuration with the weather server.

    Configures the Claude desktop application to use the weather server by creating
    or updating the configuration file with appropriate server settings.

    Returns:
        None

    Raises:
        RuntimeError: If UV executable cannot be found.
        json.JSONDecodeError: If existing config file is invalid JSON.
    """
    # Get the absolute path to the weather project
    project_dir = Path.cwd()
    if project_dir.name != "weather":
        project_dir = project_dir / "weather"
    
    # Find UV path
    uv_path = find_uv_path()
    if not uv_path:
        raise RuntimeError("Could not find uv executable. Please ensure it's installed and in PATH.")
    
    # Construct the path to the Claude config based on OS
    if platform.system() == "Windows":
        config_path = Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json"
    else:
        config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    # Create default config if it doesn't exist
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            "mcpServers": {
                "weather": {
                    "command": uv_path,
                    "args": [
                        "--directory",
                        str(project_dir),
                        "run",
                        "python3",
                        "-m",
                        "weather.server"
                    ]
                }
            }
        }
        config_path.write_text(json.dumps(default_config, indent=4))
        logger.info(f"✓ Created new Claude config at {config_path}")
        return

    # Update existing config
    try:
        config = json.loads(config_path.read_text())
        
        # Initialize mcpServers if it doesn't exist
        if "mcpServers" not in config:
            config["mcpServers"] = {}
            
        # Add or update weather server configuration
        config["mcpServers"]["weather"] = {
            "command": uv_path,
            "args": [
                "--directory",
                str(project_dir),
                "run",
                "python3",
                "-m",
                "weather.server"
            ]
        }
        
        # Write updated config back to file
        config_path.write_text(json.dumps(config, indent=4))
        logger.info(f"✓ Updated Claude config at {config_path}")
        
    except json.JSONDecodeError:
        logger.warning(f"Existing config at {config_path} is not valid JSON. Creating backup and writing new config.")
        backup_path = config_path.with_suffix('.json.bak')
        config_path.rename(backup_path)
        
        # Write new config
        default_config = {
            "mcpServers": {
                "weather": {
                    "command": uv_path,
                    "args": [
                        "--directory",
                        str(project_dir),
                        "run",
                        "python3",
                        "-m",
                        "weather.server"
                    ]
                }
            }
        }
        config_path.write_text(json.dumps(default_config, indent=4))
        logger.info(f"✓ Created new Claude config at {config_path}")

def setup_project():
    """Set up the complete weather server project.

    Performs full project setup including:
    - Installing UV if not present
    - Creating project directory structure
    - Setting up virtual environment
    - Installing dependencies
    - Creating necessary Python files
    - Configuring Claude desktop integration

    Returns:
        None

    Raises:
        SystemExit: If critical setup steps fail.
        Exception: For other unexpected errors during setup.
    """
    # Check/install uv
    uv_path = find_uv_path()
    if not uv_path:
        logger.info("uv not found in system PATH")
        uv_path = install_uv()
    else:
        logger.info("✓ uv is already installed")

    # Get the current working directory
    current_dir = Path.cwd()
    
    # Check if we're already in a weather directory
    if current_dir.name == "weather":
        project_dir = current_dir
        logger.info("✓ Already in weather directory")
        if not Path("pyproject.toml").exists():
            if not run_command(f"{uv_path} init"):
                return
            update_pyproject_toml()
    else:
        # Create project directory and initialize
        if not run_command(f"{uv_path} init weather"):
            return
        project_dir = current_dir / "weather"
        os.chdir(project_dir)
        logger.info(f"✓ Changed directory to: {project_dir}")
        update_pyproject_toml()
    
    # Create and activate virtual environment
    if not Path(".venv").exists():
        if not run_command(f"{uv_path} venv"):
            return
    else:
        logger.info("✓ Virtual environment already exists")
    
    # Install dependencies
    if not run_command(f"{uv_path} pip install mcp httpx"):
        return
    
    # Remove template file if it exists
    template_file = Path("hello.py")
    if template_file.exists():
        template_file.unlink()
        logger.info("✓ Removed hello.py")
    
    # Create project structure
    src_dir = Path("src")
    weather_dir = src_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Created directory structure")
    
    # Create necessary Python files with content
    # __init__.py content
    init_content = '''from . import server
import asyncio

def main():
    """Main entry point for the package."""
    asyncio.run(server.main())

__all__ = ['main', 'server']
'''
    init_file = weather_dir / "__init__.py"
    init_file.write_text(init_content)
    logger.info("✓ Created __init__.py with content")
    
    # server.py content
    server_content = '''from typing import Any
import asyncio
import httpx
import logging
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"
server = Server("weather")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    logger.debug("Listing available tools")
    return [
        types.Tool(
            name="get-alerts",
            description="Get weather alerts for a state",
            inputSchema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Two-letter state code (e.g. CA, NY)",
                    },
                },
                "required": ["state"],
            },
        ),
        types.Tool(
            name="get-forecast",
            description="Get weather forecast for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the location",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the location",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        ),
    ]

async def make_nws_request(client: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    try:
        logger.debug(f"Making request to NWS API: {url}")
        response = await client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error making NWS request: {e}")
        return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a concise string."""
    props = feature["properties"]
    return (
        f"Event: {props.get('event', 'Unknown')}\\n"
        f"Area: {props.get('areaDesc', 'Unknown')}\\n"
        f"Severity: {props.get('severity', 'Unknown')}\\n"
        f"Status: {props.get('status', 'Unknown')}\\n"
        f"Headline: {props.get('headline', 'No headline')}\\n"
        "---"
    )

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can fetch weather data and notify clients of changes.
    """
    logger.info(f"Tool call received: {name} with arguments: {arguments}")
    
    if not arguments:
        logger.error("Missing arguments")
        raise ValueError("Missing arguments")
  
    if name == "get-alerts":
        state = arguments.get("state")
        if not state:
            logger.error("Missing state parameter")
            raise ValueError("Missing state parameter")

        # Convert state to uppercase to ensure consistent format
        state = state.upper()
        if len(state) != 2:
            logger.error(f"Invalid state code: {state}")
            raise ValueError("State must be a two-letter code (e.g. CA, NY)")

        async with httpx.AsyncClient() as client:
            alerts_url = f"{NWS_API_BASE}/alerts?area={state}"
            alerts_data = await make_nws_request(client, alerts_url)

            if not alerts_data:
                logger.error("Failed to retrieve alerts data")
                return [types.TextContent(type="text", text="Failed to retrieve alerts data")]

            features = alerts_data.get("features", [])
            if not features:
                logger.info(f"No active alerts for {state}")
                return [types.TextContent(type="text", text=f"No active alerts for {state}")]

            # Format each alert into a concise string
            formatted_alerts = [format_alert(feature) for feature in features[:20]] # only take the first 20 alerts
            alerts_text = f"Active alerts for {state}:\\n\\n" + "\\n".join(formatted_alerts)

            logger.info(f"Successfully retrieved alerts for {state}")
            return [
                types.TextContent(
                    type="text",
                    text=alerts_text
                )
            ]
    elif name == "get-forecast":
        try:
            latitude = float(arguments.get("latitude"))
            longitude = float(arguments.get("longitude"))
        except (TypeError, ValueError):
            logger.error("Invalid coordinates provided")
            return [types.TextContent(
                type="text",
                text="Invalid coordinates. Please provide valid numbers for latitude and longitude."
            )]
            
        # Basic coordinate validation
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            logger.error(f"Coordinates out of range: {latitude}, {longitude}")
            return [types.TextContent(
                type="text",
                text="Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180."
            )]

        async with httpx.AsyncClient() as client:
            # First get the grid point
            lat_str = f"{latitude}"
            lon_str = f"{longitude}"
            points_url = f"{NWS_API_BASE}/points/{lat_str},{lon_str}"
            points_data = await make_nws_request(client, points_url)

            if not points_data:
                logger.error(f"Failed to retrieve grid point data for {latitude}, {longitude}")
                return [types.TextContent(type="text", text=f"Failed to retrieve grid point data for coordinates: {latitude}, {longitude}. This location may not be supported by the NWS API (only US locations are supported).")]

            # Extract forecast URL from the response
            properties = points_data.get("properties", {})
            forecast_url = properties.get("forecast")
            
            if not forecast_url:
                logger.error("Failed to get forecast URL from grid point data")
                return [types.TextContent(type="text", text="Failed to get forecast URL from grid point data")]

            # Get the forecast
            forecast_data = await make_nws_request(client, forecast_url)
            
            if not forecast_data:
                logger.error("Failed to retrieve forecast data")
                return [types.TextContent(type="text", text="Failed to retrieve forecast data")]

            # Format the forecast periods
            periods = forecast_data.get("properties", {}).get("periods", [])
            if not periods:
                logger.error("No forecast periods available")
                return [types.TextContent(type="text", text="No forecast periods available")]

            # Format each period into a concise string
            formatted_forecast = []
            for period in periods:
                forecast_text = (
                    f"{period.get('name', 'Unknown')}:\\n"
                    f"Temperature: {period.get('temperature', 'Unknown')}°{period.get('temperatureUnit', 'F')}\\n"
                    f"Wind: {period.get('windSpeed', 'Unknown')} {period.get('windDirection', '')}\\n"
                    f"{period.get('shortForecast', 'No forecast available')}\\n"
                    "---"
                )
                formatted_forecast.append(forecast_text)

            forecast_text = f"Forecast for {latitude}, {longitude}:\\n\\n" + "\\n".join(formatted_forecast)

            logger.info(f"Successfully retrieved forecast for {latitude}, {longitude}")
            return [types.TextContent(
                type="text",
                text=forecast_text
            )]
    else:
        logger.error(f"Unknown tool: {name}")
        raise ValueError(f"Unknown tool: {name}")

async def main():
    logger.info("Starting weather server...")
    try:
        # Run the server using stdin/stdout streams
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Got stdio streams, initializing server...")
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="weather",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Server shutting down...")

if __name__ == "__main__":
    asyncio.run(main())
'''
    server_file = weather_dir / "server.py"
    server_file.write_text(server_content)
    logger.info("✓ Created server.py with content")
    
    # Set up Claude configuration
    setup_claude_config()
    
    logger.info("\n✓ Project setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Add your server implementation to src/weather/server.py")
    logger.info("2. Restart Claude Desktop to apply the configuration changes")

if __name__ == "__main__":
    try:
        setup_project()
    except KeyboardInterrupt:
        logger.error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nSetup failed: {e}")
        sys.exit(1)