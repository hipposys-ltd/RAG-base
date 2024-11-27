from enum import Enum
from typing import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage, AIMessage, AIMessageChunk
from langchain_core.prompts.prompt import PromptTemplate

from app.databases.vector import VectorDB
from app.databases.postgres import Database
from app.models import ChatModel
from app.utils.logger import Logger


PROMPT_MESSAGE = """When answering the user question using data from the tools, be sure to:
1. First list the sources you are going to use in your answer. The list of sources should be of the form:
[1] - source_id | source_name | modified_at | (index of document in the retriever's answer (e.g. 0, 4): <The quote that being used>
[2] - ...
2. Add a separator '\n\n+++++++++++++\n\n'
3. Write a concise answer based only on the sources and quotes you listed above.

If you don't know the answer, answer that you don't know based on the information you have.
If you're not sure, state that you're not sure."""


class LLMEventType(Enum):
    """Event types for the LLM agent."""

    STORED_MESSAGE = 'stored_message'
    RETRIEVER_START = 'on_retriever_start'
    RETRIEVER_END = 'on_retriever_end'
    CHAT_CHUNK = 'on_chat_model_stream'
    DONE = 'done'


class ChatMessage:
    """A message that can be sent to the client.
    
    This is a simplified version of the messages returned by the LLM agent. It is
    used to provide the client with information about the conversation as well as
    the answers from the agent.

    The reason this is needed is that the messages returned by the LLM agent are
    too verbose and complex. Also, there's a difference between the messages returned
    during the conversation (events) and the historical messages.
    """

    class Sender(Enum):
        """The sender of the message."""
        SYSTEM = 'system'
        AI = 'ai'
        HUMAN = 'human'
        TOOL = 'tool'

    def __init__(self, type: LLMEventType, sender: Sender, content: str, payload: dict = None):
        self.type = type
        self.sender: str = sender.value
        self.content = content
        self.payload = payload or {}

    @classmethod
    def from_base_message(cls, message: BaseMessage) -> 'ChatMessage':
        """Converts a `BaseMessage` inherited object to a `ChatMessage` object."""

        # Mapping of message types to the types used in the client.
        message_type_lookup = {
            HumanMessage: cls.Sender.HUMAN,
            SystemMessage: cls.Sender.SYSTEM,
            ToolMessage: cls.Sender.TOOL,
            AIMessage: cls.Sender.AI,
            AIMessageChunk: cls.Sender.AI,
        }

        # Different message types have different structures.
        try:
            content = message.content[0]['text']
        except (KeyError, TypeError, IndexError):
            content = message.content

        return ChatMessage(
            LLMEventType.STORED_MESSAGE,
            sender=message_type_lookup[type(message)],
            content=content,
        )

    @classmethod
    def from_event(cls, event: dict) -> 'ChatMessage':
        """Convert an event from the LLM agent to a `ChatMessage` object."""

        match event['event']:
            # The Retrival process was triggered.
            case 'on_retriever_start':
                return ChatMessage(
                    LLMEventType.RETRIEVER_START,
                    cls.Sender.SYSTEM,
                    'Searching Vector DB',
                    payload={'search_query': event['data']['input']['query']},
                )
            # The retriever returned results
            case 'on_retriever_end':
                return ChatMessage(
                    LLMEventType.RETRIEVER_END,
                    cls.Sender.SYSTEM,
                    'Analyzing retrieved data for answers',
                    payload={'retrieved_data': [{
                        'source_name': document.metadata['source_name'],
                        'source_id': document.metadata['source_id'],
                        'modified': document.metadata['modified_at'],
                        'content': document.page_content,
                    } for document in event['data']['output']]},
                )
            # A part of the agent's response. The response is streamed in chunks.
            case 'on_chat_model_stream' if event['data']['chunk'].content:
                return cls._handle_on_chat_model_stream(event)
            # The conversation is done.
            case 'done':
                return ChatMessage(
                    LLMEventType.DONE,
                    cls.Sender.SYSTEM,
                    'Done',
                )
            # Known events that we ignore.
            case 'on_chat_model_start' | 'on_chain_start' | 'on_chain_end' | 'on_chat_model_stream' \
                | 'on_chat_model_end' | 'on_chain_stream' | 'on_tool_start' | 'on_tool_end':
                Logger().get_logger().debug('Ignoring message', event['event'])
                return None
            # Unknown events.
            case _:
                raise ValueError('Unknown event', event)
            
    @classmethod
    def _handle_on_chat_model_stream(cls, event: dict) -> 'ChatMessage':
        """Handle the `on_chat_model_stream` event."""

        # Handle the fact that the content structure can be different, depending on model type.
        content = event['data']['chunk'].content
        content_type = ''
        if not isinstance(content, str):
            content_type = content[0]['type']
            content = content[0].get('text')

        # If the message is a tool call, just print a debug message.
        if content_type in ('tool_use', 'tool_call'):
            Logger().get_logger().debug('Stream.tool_calls:', event['data']['chunk'].tool_calls, flush=True)
            return None
        else:
            return ChatMessage(LLMEventType.CHAT_CHUNK, cls.Sender.AI, content)

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the message."""
        return {
            'sender': self.sender,
            'content': self.content,
            'payload': self.payload,
        }


# TODO: Consider using a pool of agents for better performance, or some other caching mechanism.
class LLMAgent:
    """A wrapper for the LangChain agent.
    
    Provides a cleaner interface to work with the LLM agent.

    Example usage:
    >>> async with LLMAgent() as llm_agent:
    ...     async for event_type, event in llm_agent.astream_events('Hello', chat_session):
    ...         print(event)
    """

    def __init__(self):
        self._agent = None
        self._llm = None
        self.retriever_tool_name = 'Internal_Company_Info_Retriever'
        self._checkpointer_ctx = None

    async def __aenter__(self) -> 'LLMAgent':
        """Initialize the LLM agent."""

        # The Retriever in the RAG model.
        retriever = VectorDB().as_retriever(search_kwargs={'k': 8})

        # The ChatBot LLM
        self._llm = ChatModel()

        # Retriever tool, for the R in RAG.
        tool = create_retriever_tool(
            retriever,
            self.retriever_tool_name,
            'Searches and retrieves data from the corpus of documents that the company has',

            # Controls how the returned results will look when passed to the LLM.
            # To see available `variables`, check the `retriever.invoke('some query')[0].metadata.keys()`
            document_prompt=PromptTemplate(
                template_format='jinja2',
                input_variables=['source_id', 'source_name', 'modified_at', 'page_content'],
                template='{'
                    '"source_name": "{{source_name}}", '  # E.g. filename, article title, etc.
                    '"source_id": "{{source_id}}", '      # E.g. URL, document ID, etc.
                    '"modified": "{{modified_at}}", '
                    '"content": "{{page_content|replace(\'"\', \'""\')}}"'
                '}',
            ),
            document_separator='\n',
        )
        tools = [tool]

        # Checkpointer for the agent.
        self._checkpointer_ctx = AsyncPostgresSaver.from_conn_string(Database().get_connection_string())
        checkpointer = await self._checkpointer_ctx.__aenter__()

        # Create the agent itself.
        self._agent = create_react_agent(
            self._llm,
            tools,
            checkpointer=checkpointer,
            state_modifier=SystemMessage(PROMPT_MESSAGE),
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the agent and the checkpointer."""
        await self._checkpointer_ctx.__aexit__(exc_type, exc_val, exc_tb) 
        self._llm = None
        self._agent = None
        self._checkpointer_ctx = None

    async def astream_events(self, message: str, chat_session: dict) -> AsyncGenerator[ChatMessage, None]:
        """Stream the events from the LLM agent.
        
        :param message: The message to send to the agent.
        :param chat_session: The chat session configuration. This is used to keep track of the conversation.

        :return: An async generator that yields the events from the agent. Each event is a 2-tuple of:
            - The event type (`LLMEventType`).
            - The event data.
        """

        # Send the message to the agent and yield the events.
        async for event in self._agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=chat_session,
            version='v2',
        ):
            # Process the event and if relevant, yield a message to the user.
            message = ChatMessage.from_event(event)
            if message:
                yield message

        # Let the client know that the conversation is done.
        yield ChatMessage.from_event({'event': 'done'})

    async def aget_history(self, chat_session: dict) -> list[dict]:
        """Get the chat history for the given chat session."""
        
        state = await self._agent.aget_state(chat_session)
        return [ChatMessage.from_base_message(message).to_dict() for message in state.values['messages']]
