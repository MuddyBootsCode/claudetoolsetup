import subprocess
import os
import sys
from pathlib import Path
import shutil

def check_uv_exists():
    """Check if uv is available in the system."""
    return shutil.which('uv') is not None

def install_uv():
    """Install uv using the official install script."""
    print("Installing uv...")
    try:
        process = subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✓ uv installed successfully")
        
        # Refresh PATH to include uv
        os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing uv:")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def update_pyproject_toml():
    """Update pyproject.toml with additional configuration."""
    additional_config = """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/weather"]

[project.scripts]
weather = "weather:main"
"""
    with open("pyproject.toml", "a") as f:
        f.write(additional_config)
    print("✓ Updated pyproject.toml with build system and scripts configuration")

def run_command(command, cwd=None):
    """Execute a command and wait for it to complete."""
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing '{command}':")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def setup_claude_config():
    """Set up the Claude desktop configuration with the weather server."""
    import json
    from pathlib import Path
    
    # Get the absolute path to the weather project
    project_dir = Path.cwd()
    if project_dir.name != "weather":
        project_dir = project_dir / "weather"
    
    # Construct the path to the Claude config
    config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    # Create default config if it doesn't exist
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            "mcpServers": {
                "weather": {
                    "command": "uv",
                    "args": [
                        "--directory",
                        str(project_dir),
                        "run",
                        "weather"
                    ]
                }
            }
        }
        config_path.write_text(json.dumps(default_config, indent=4))
        print(f"✓ Created new Claude config at {config_path}")
        return

    # Update existing config
    try:
        config = json.loads(config_path.read_text())
        
        # Initialize mcpServers if it doesn't exist
        if "mcpServers" not in config:
            config["mcpServers"] = {}
            
        # Add or update weather server configuration
        config["mcpServers"]["weather"] = {
            "command": "uv",
            "args": [
                "--directory",
                str(project_dir),
                "run",
                "weather"
            ]
        }
        
        # Write updated config back to file
        config_path.write_text(json.dumps(config, indent=4))
        print(f"✓ Updated Claude config at {config_path}")
        
    except json.JSONDecodeError:
        print(f"Warning: Existing config at {config_path} is not valid JSON. Creating backup and writing new config.")
        # Create backup of invalid config
        backup_path = config_path.with_suffix('.json.bak')
        config_path.rename(backup_path)
        
        # Write new config
        default_config = {
            "mcpServers": {
                "weather": {
                    "command": "uv",
                    "args": [
                        "--directory",
                        str(project_dir),
                        "run",
                        "weather"
                    ]
                }
            }
        }
        config_path.write_text(json.dumps(default_config, indent=4))
        print(f"✓ Created new Claude config at {config_path}")

def setup_project():
    # Existing setup code...
    if not check_uv_exists():
        print("uv not found in system PATH")
        if not install_uv():
            print("Failed to install uv. Please install it manually and try again.")
            return
    else:
        print("✓ uv is already installed")

    # Get the current working directory
    current_dir = Path.cwd()
    
    # Check if we're already in a weather directory
    if current_dir.name == "weather":
        project_dir = current_dir
        print("✓ Already in weather directory")
        # Initialize uv in the current directory if pyproject.toml doesn't exist
        if not Path("pyproject.toml").exists():
            if not run_command("uv init"):
                return
            update_pyproject_toml()
    else:
        # Create project directory and initialize
        if not run_command("uv init weather"):
            return
        project_dir = current_dir / "weather"
        os.chdir(project_dir)
        print(f"✓ Changed directory to: {project_dir}")
        update_pyproject_toml()
    
    # Create and activate virtual environment
    if not Path(".venv").exists():
        if not run_command("uv venv"):
            return
    else:
        print("✓ Virtual environment already exists")
    
    # Note: source cannot be run directly as it's a shell built-in
    # Instead, we'll activate the venv in the current Python process
    venv_activate_script = project_dir / ".venv" / "bin" / "activate"
    os.environ["VIRTUAL_ENV"] = str(project_dir / ".venv")
    os.environ["PATH"] = str(project_dir / ".venv" / "bin") + os.pathsep + os.environ["PATH"]
    print("✓ Activated virtual environment")
    
    # Install dependencies
    if not run_command("uv add mcp httpx"):
        return
    
    # Remove template file if it exists
    template_file = Path("hello.py")
    if template_file.exists():
        template_file.unlink()
        print("✓ Removed hello.py")
    
    # Create project structure
    src_weather_dir = Path("src/weather")
    src_weather_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {src_weather_dir}")
    
    # Create and write content to files
    init_content = '''from . import server
import asyncio

def main():
    """Main entry point for the package."""
    asyncio.run(server.main())

__all__ = ['main', 'server']
'''
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
    init_file = src_weather_dir / "__init__.py"
    server_file = src_weather_dir / "server.py"
    init_file.write_text(init_content)
    server_file.write_text(server_content)
    print("✓ Created Python files with initial content")
    
    print("\nProject setup completed successfully!")

if __name__ == "__main__":
    setup_project()
    setup_claude_config()