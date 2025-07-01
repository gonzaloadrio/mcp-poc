import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_use import MCPClient, MCPAgent
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

os.environ["GOOGLE_API_KEY"] = "AIzaSyBZP6QUlAIFnC8krn6ToNyQ7ocILJlMPzI"

os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve the chat message history for a given session.

    If the session does not exist in the store, a new ChatMessageHistory is created and stored.

    Args:
        session_id (str): The unique identifier for the chat session.

    Returns:
        BaseChatMessageHistory: The message history associated with the session.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


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
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        return_intermediate_steps=True,
    )
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    try:
        while True:
            message = input("User message: ")

            if message.lower() == "exit":
                print("Exiting the client application.")
                break

            result = await agent_with_chat_history.ainvoke(
                {"input": message},
                config={"configurable": {"session_id": "<foo>"}},
            )
            print(f"\n🤖 Agent response: {result['output']}")

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
