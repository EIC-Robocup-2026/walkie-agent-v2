"""Planning and task management middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
)
from langchain.tools import InjectedToolCallId


class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item."""


class PlanningState(AgentState[Any]):
    """State schema for the todo middleware."""

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """List of todo items for tracking task progress."""


WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for physical and perceptual tasks (navigation, finding people or objects, arm actions, etc.). This helps you track progress on multi-step robot tasks and gives the user visibility into what you are doing.

Only use this tool when the request involves 3 or more distinct steps. For one or two simple actions (e.g., "go forward 1 meter", "what is my position?"), do the task directly and do NOT call this tool.

## When to Use This Tool
Use this tool in these scenarios:

1. **Multi-step navigation and actions** — e.g., go to the kitchen, find the coffee mug, bring it to the living room
2. **Search then act** — e.g., find person X, then navigate to them and greet them; or find where the keys are, then go there
3. **Several physical steps** — e.g., turn to face the door, move forward 2 m, wave, then return to base
4. **User gives a list of tasks** — e.g., "first go to the office, then check if the door is open, then come back"
5. **User explicitly asks for a plan** — e.g., "can you make a plan to deliver this to John?"

## How to Use This Tool
1. When you start working on a task — Mark it as in_progress BEFORE you begin (e.g., before calling control_actuators or use_vision).
2. After completing a step — Mark it as completed and add any new follow-up steps you discover (e.g., "Bring object to user" after finding it).
3. You can update the list: remove tasks that are no longer needed, or add new ones. Do not change completed tasks.
4. You can update several items at once (e.g., mark one completed and the next in_progress).

## When NOT to Use This Tool
Skip this tool when:
1. **Single movement or query** — e.g., "move forward 1 meter", "what do you see?", "where are you?"
2. **Simple greeting or conversation** — e.g., "hello", "how are you?", "what can you do?"
3. **One or two trivial steps** — e.g., "turn left" then "go forward" with no larger goal
4. **Purely informational** — answering a question that does not involve physical or perceptual steps

## Task States and Management

1. **Task states:** pending (not started), in_progress (doing it now), completed (done).
2. **Rules:** Mark the first task in_progress as soon as you create the list. Mark tasks completed only when fully done. Remove tasks that become irrelevant. Unless everything is done, keep at least one task in_progress.
3. **Completion:** Only mark completed when the step is fully accomplished. If something fails or you are blocked, keep it in_progress or add a new task for resolving the blocker.
4. **Breakdown:** Use clear, actionable steps (e.g., "Navigate to kitchen (x=2, y=3)", "Locate coffee mug using vision", "Pick up mug with arm").

Remember: For short, clear requests that need only one or two actions, do them directly without calling this tool."""

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to plan and track multi-step robot tasks (navigation, finding people/objects, arm actions, etc.). Use it when the user's request has 3 or more distinct physical or perceptual steps, so you can track progress and show the user your plan.

Mark todos as completed as soon as you finish each step. Do not batch completions. For simple requests (one or two actions), complete them directly and do NOT use this tool — it costs time and tokens.

**Important:** Call `write_todos` at most once per turn. Revise the list as you go: add new steps if you discover them, remove steps that are no longer needed."""


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )


class TodoListMiddleware(AgentMiddleware):
    """Middleware that provides todo list management capabilities to agents.

    This middleware adds a `write_todos` tool that allows agents to create and manage
    structured task lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    The middleware automatically injects system prompts that guide the agent on when
    and how to use the todo functionality effectively. It also enforces that the
    `write_todos` tool is called at most once per model turn, since the tool replaces
    the entire todo list and parallel calls would create ambiguity about precedence.

    Example:
        ```python
        from langchain.agents.middleware.todo import TodoListMiddleware
        from langchain.agents import create_agent

        agent = create_agent("openai:gpt-4o", middleware=[TodoListMiddleware()])

        # Agent now has access to write_todos tool and todo state tracking
        result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})

        print(result["todos"])  # Array of todo items with status tracking
        ```
    """

    state_schema = PlanningState

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
        initial_todos: list[Todo] | None = None,
    ) -> None:
        """Initialize the `TodoListMiddleware` with optional custom prompts.

        Args:
            system_prompt: Custom system prompt to guide the agent on using the todo
                tool.
            tool_description: Custom description for the `write_todos` tool.
            initial_todos: Optional list of initial todo items to seed the agent with
                at the start of execution. These todos will be set in the state during
                ``before_agent`` if the state does not already contain todos.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description
        self.initial_todos: list[Todo] = list(initial_todos) if initial_todos else []
        # Dynamically create the write_todos tool with the custom description
        @tool(description=self.tool_description)
        def write_todos(
            todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command[Any]:
            """Create and manage a structured task list for your current work session."""
            
            return Command(
                update={
                    "todos": todos,
                    "messages": [
                        ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
                    ],
                }
            )

        self.tools = [write_todos]

    @override
    def before_agent(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
        """Seed the state with initial todos if provided and no todos exist yet.

        This hook runs once at the start of agent execution. If ``initial_todos``
        were provided during middleware initialization and the current state does
        not already contain todos, the initial todos are injected into the state.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            A dict with the ``todos`` key set to the initial todos list, or
            ``None`` if no seeding is needed.
        """
        if self.initial_todos and not state.get("todos"):
            return {"todos": list(self.initial_todos)}
        return None

    @override
    async def abefore_agent(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
        """Async version of ``before_agent``. Seeds initial todos into state.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            A dict with the ``todos`` key set to the initial todos list, or
            ``None`` if no seeding is needed.
        """
        return self.before_agent(state, runtime)

    @staticmethod
    def _format_todos(todos: list[Todo]) -> str:
        """Format a list of todos into a readable string for the system prompt.

        Args:
            todos: The list of todo items to format.

        Returns:
            A formatted string representation of the todos, or an empty string
            if the list is empty.
        """
        if not todos:
            return ""

        status_icons = {
            "pending": "⬜",
            "in_progress": "🔄",
            "completed": "✅",
        }
        lines = ["## Current Todo List"]
        for i, todo in enumerate(todos, 1):
            icon = status_icons.get(todo.get("status", "pending"), "⬜")
            lines.append(f"{i}. {icon} [{todo.get('status', 'pending')}] {todo.get('content', '')}")

        print(f"Formatted todos: {lines}")
        return "\n".join(lines)

    def _build_system_message(self, request: ModelRequest) -> SystemMessage:
        """Build the system message with the todo prompt and current todos injected.

        Args:
            request: The model request containing the current state and system message.

        Returns:
            A new ``SystemMessage`` with the todo system prompt and current todos
            appended.
        """
        todos: list[Todo] = request.state.get("todos", [])
        todos_text = self._format_todos(todos)

        extra_text = f"\n\n{self.system_prompt}"
        if todos_text:
            extra_text += f"\n\n{todos_text}"

        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": extra_text},
            ]
        else:
            new_system_content = [{"type": "text", "text": extra_text.lstrip()}]

        return SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Update the system message to include the todo system prompt and current todos.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.
        """
        new_system_message = self._build_system_message(request)
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Update the system message to include the todo system prompt and current todos.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.
        """
        new_system_message = self._build_system_message(request)
        return await handler(request.override(system_message=new_system_message))

    @override
    def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
        """Check for parallel write_todos tool calls and return errors if detected.

        The todo list is designed to be updated at most once per model turn. Since
        the `write_todos` tool replaces the entire todo list with each call, making
        multiple parallel calls would create ambiguity about which update should take
        precedence. This method prevents such conflicts by rejecting any response that
        contains multiple write_todos tool calls.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Count write_todos tool calls
        write_todos_calls = [tc for tc in last_ai_msg.tool_calls if tc["name"] == "write_todos"]

        if len(write_todos_calls) > 1:
            # Create error tool messages for all write_todos calls
            error_messages = [
                ToolMessage(
                    content=(
                        "Error: The `write_todos` tool should never be called multiple times "
                        "in parallel. Please call it only once per model invocation to update "
                        "the todo list."
                    ),
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in write_todos_calls
            ]

            # Keep the tool calls in the AI message but return error messages
            # This follows the same pattern as HumanInTheLoopMiddleware
            return {"messages": error_messages}

        return None

    @override
    async def aafter_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
        """Check for parallel write_todos tool calls and return errors if detected.

        Async version of `after_model`. The todo list is designed to be updated at
        most once per model turn. Since the `write_todos` tool replaces the entire
        todo list with each call, making multiple parallel calls would create ambiguity
        about which update should take precedence. This method prevents such conflicts
        by rejecting any response that contains multiple write_todos tool calls.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        return self.after_model(state, runtime)
