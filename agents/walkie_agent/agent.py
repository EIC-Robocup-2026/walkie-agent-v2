from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

from audio.walkie import WalkieAudio
from agents.robot_state import RobotState

from .prompts import WALKIE_AGENT_SYSTEM_PROMPT
from .tools import create_sub_agents_tools, create_speak_tool, create_follow_person_tool, create_go_to_raised_hand_tool, think
from ..middleware import RobotStateMiddleware, SequentialToolCallMiddleware, TodoListMiddleware

checkpointer = InMemorySaver()

def create_walkie_agent(model, walkieAudio: WalkieAudio, walkie_vision, walkie_db, tools=[]):
    """Create the main Walkie agent with sub-agent tools.

    Args:
        model: The LLM model to use for this agent and its sub-agents
        walkieAudio: Optional WalkieAudio for the speak tool
        walkie_vision: Optional WalkieVision for the vision agent tools
        walkie_db: Optional WalkieVectorDB for vision agent (find_object, find_scene, scan_and_remember)
        tools: Additional tools to add to the agent

    Returns:
        The configured Walkie agent
    """
    robot = walkie_vision._camera._bot
    tools = create_sub_agents_tools(model, robot=robot, walkie_vision=walkie_vision, walkie_db=walkie_db) + tools

    if walkieAudio:
        tools.append(create_speak_tool(walkieAudio))
    if walkie_vision and walkieAudio:
        tools.append(create_follow_person_tool(robot, walkie_vision, walkieAudio))
    if walkie_vision:
        tools.append(create_go_to_raised_hand_tool(robot, walkie_vision))
    tools.append(think)

    # Check available tools
    robot_state = RobotState(robot, vision_enabled=True)

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", 4000),
                keep=("messages", 10),
            ),
            SequentialToolCallMiddleware(),
            RobotStateMiddleware(robot_state),
            TodoListMiddleware(),
        ],
        system_prompt=WALKIE_AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent
