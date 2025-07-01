from fastmcp import FastMCP
from httpx import AsyncClient

app = FastMCP()


@app.tool("echo")
async def echo(message: str) -> str:
    """
    Echo tool for the FastMCP server.

    Args:
        message (str): The message to echo back.

    Returns:
        str: The echoed message.
    """
    return f"Echo: {message}"


@app.tool("cat_fact")
async def cat_fact() -> str:
    """
    Cat fact tool for the FastMCP server.

    Returns:
        str: A random cat fact.
    """
    async with AsyncClient() as client:
        response = await client.get("https://catfact.ninja/fact")
        data = response.json()
        print(f"Received cat fact: {data}")
        return data["fact"]


@app.tool("age_prediction")
async def age_prediction(name: str) -> str:
    """
    Age prediction tool for the FastMCP server.

    Args:
        name (str): The name of the person to predict age for.

    Returns:
        str: A predicted age for the given name.
    """
    async with AsyncClient() as client:
        response = await client.get(f"https://api.agify.io?name={name}")
        data = response.json()
        print(f"Received age prediction: {data}")
        return data["age"]


if __name__ == "__main__":
    #app.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")
    app.run(transport="sse", host="127.0.0.1", port=8001, path="/mcp/sse")
    print("FastMCP server is running on http://localhost:8000")
