import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_use import MCPClient, MCPAgent
from mcp_use.adapters.langchain_adapter import LangChainAdapter

os.environ["GOOGLE_API_KEY"] = "AIzaSyBZP6QUlAIFnC8krn6ToNyQ7ocILJlMPzI"

os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"


async def main():
    """
    Main function to run the MCP client application.
    """

    config = {"mcpServers": {"http": {"url": "http://localhost:8001/mcp/sse/"}}}

    client = MCPClient.from_dict(config)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )

    adapter = LangChainAdapter()

    # Get tools directly from the client
    tools = await adapter.create_tools(client)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant with access to powerful tools. "
                "Try to respond in the same language as the user, translate if necessary.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    previous_ai_message = ""
    previous_user_message = ""

    try:
        while True:
            message = input("User message: ")
            previous_user_message = message

            if message.lower() == "exit":
                print("Exiting the client application.")
                break

            result = await agent_executor.ainvoke(
                {
                    "input": message,
                    "chat_history": [
                        HumanMessage(content=previous_user_message),
                        AIMessage(content=previous_ai_message),
                    ],
                }
            )

            print("AI Response:", result["output"])
            previous_ai_message = result["output"]

    except KeyboardInterrupt:
        print("Client application interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    print("Client application has started.")


async def execute_query(agent, query: str) -> str:
    """
    Function to query the agent with a given input text.

    Args:
        input_text (str): The input text to query the agent.

    Returns:
        str: The response from the agent.
    """
    print("🤖 Starting agent task...")
    print("-" * 50)

    current_step = 1

    async for event in agent.astream_events(query, version="v1"):
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "on_chat_model_start":
            print(f"\n📝 Step {current_step}: Planning next action...")

        elif event_type == "on_tool_start":
            tool_name = data.get("input", {}).get("tool_name", "unknown")
            print(f"\n🔧 Using tool: {tool_name}")

        elif event_type == "on_tool_end":
            print(" ✅ Tool completed")
            current_step += 1

        elif event_type == "on_chat_model_stream":
            token = data.get("chunk", {}).get("content", "")
            if token:
                print(token, end="", flush=True)

        elif event_type == "on_chain_end":
            print("\n\n🎉 Task completed successfully!")
