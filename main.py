"""Entry point: MCP server or API server (implementation pending)."""

from devcontext.config.settings import settings

def main():
    print("DevContext starting...")
    print(f"Model     : {settings.model_name}")
    print(f"Docs dir  : {settings.docs_dir}")
    print(f"Chroma dir: {settings.chroma_db_dir}")
    print(f"MCP port  : {settings.mcp_server_port}")

if __name__ == "__main__":
    main()