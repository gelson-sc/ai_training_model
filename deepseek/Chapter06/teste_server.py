from fastmcp import FastMCP

mcp = FastMCP(name="MyAssistantServer")

@mcp.tool
def get_weather(city: str) -> str:
    """Obtém a previsão do tempo para uma cidade."""
    return f"O tempo em {city} está ensolarado com máxima de 25°C."

@mcp.resource("users://{user_id}/profile")
def get_profile(user_id: int):
    return {"name": f"Usuário {user_id}", "status": "ativo"}

if __name__ == "__main__":
    mcp.run()