import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from  mcp_use import MCPAgent, MCPClient
import os

async def run_memory_chat():
    """RUn a chat using MCPAgent's built in conversation memory"""
    
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    #Config file path -change this to your config file
    config_file = "browser_mcp.json"

    print("Initializing MCP Agent")

    ##create MCP client and agent with memory enabled
    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model = "qwen-qwq-32b")

    ##create agent with memory enabled
    agent = MCPAgent(
        llm = llm,
        client = client,
        max_steps =15,
        memory_enabled = True,
    )

    print("\n====== Interactive MCP Chat ======")
    print("Type exit to end the chat")
    print("Type clear to clear the conversation history")
    print("=====================================\n")


    try:
        #Main chat loop
        while True:
            #Get user input
            user_input = input("You: ")

            #Exit condition
            if user_input.lower() == "exit":
                print("Exiting chat...")
                break

            #Clear conversation history
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared")
                continue


            #get response from agent
            print("\n Assistant",end ="",flush = True)

            try:
                #run the agent with the user input (memory handling is automatic)
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"\nError: {e}")

    finally:
        #clean up
        if client and client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_memory_chat())