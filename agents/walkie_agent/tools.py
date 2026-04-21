import math
import threading
import time

from langchain_core.tools import tool

from audio.walkie import WalkieAudio
from ..actuators_agent import create_actuator_agent
from ..vision_agent import create_vision_agent


def create_sub_agents_tools(model, robot, walkie_vision, walkie_db):
    """Initialize sub-agents with the provided model and optional vision/db. Call this before using the tools."""
    _actuator_agent = create_actuator_agent(model, robot=robot)
    _vision_agent = create_vision_agent(model, walkie_vision=walkie_vision, walkie_db=walkie_db)

    tools = []

    @tool(parse_docstring=True)
    def control_actuators(task: str) -> str:
        """Command a movement or physical action to the Actuator Agent. Use for drive-base navigation (absolute or relative) or arm actions.

        When to use:
        - Move to map coordinates: "go to x=5, y=3" or "navigate to (2, 1) facing 90 degrees"
        - Move relative to current pose: "go forward 1 meter", "move 0.5 m to the left", "turn left 90 degrees"
        - Arm gestures or manipulation: "wave hello", "point to the left", "pick up the cup", "shake hands"
        - Check pose: "what is my current position?" or "where am I?"

        When NOT to use:
        - For seeing or recognizing something (use use_vision instead)
        - For speaking to the user (use speak instead)
        - For planning a list of steps (use write_todos if there are 3+ steps)

        Args:
            task: Natural language description of the movement or action. Be specific: absolute vs relative, units (meters, degrees), and arm action if needed.

        Returns:
            str: Result of the movement or action (success, pose, or error message).
        """
        if _actuator_agent is None:
            return "Error: Actuator agent not initialized. Please initialize sub-agents first."
        
        print(f"Control actuators: {task}")
        result = _actuator_agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })
        
        # Extract the final response from the agent
        return result["messages"][-1].content

    tools.append(control_actuators)
    
    
    if _vision_agent is not None:
        @tool(parse_docstring=True)
        def use_vision(task: str) -> str:
            """Delegate a vision or perception task to the Vision Agent. Use when you need to see, recognize, or find something in the environment.

            When to use:
            - Describe current view: "what do you see?", "describe what's in front of you", "look around and summarize"
            - People: "how many people are in view?", "detect people and their poses", "is anyone I know? check faces", "where is John?" (search by name/FaceID)
            - Objects and places: "where is the coffee mug?" (in view or in database), "find the kitchen" or "find a room with a whiteboard"
            - Text: "read the sign in front of you", "what does the it say?"

            When NOT to use:
            - For moving or turning (use control_actuators)
            - For speaking (use speak)
            - When the user only asks a general question with no need to look (e.g., "what time is it?")

            Args:
                task: Natural language description of what to look at, detect, or find. Be specific (e.g., "find the red cup" not just "find object").

            Returns:
                str: Vision result (descriptions, positions, identities, or "vision disabled" / error).
            """
            if _vision_agent is None:
                return "Error: Vision agent not initialized. Please initialize sub-agents first."
            
            print(f"Use vision: {task}")
            result = _vision_agent.invoke({
                "messages": [{"role": "user", "content": task}]
            })
            
            # Extract the final response from the agent
            return result["messages"][-1].content
    
        tools.append(use_vision)
    
    return tools


def create_speak_tool(walkieAudio: WalkieAudio) -> str:
    @tool(parse_docstring=True)
    def speak(text: str) -> str:
        """Speak the given text out loud. This is a tool that you can use to speak to the user to give them information beforing calling other tools.
        
        When to use:
        - To give the user information before calling other tools (performing actions)
        - You MAY NOT use this tool to speak out loud the final answer to the user. Instead, return the final answer in the agent response.
        
        Args:
            text: The text to speak
        
        Returns:
            str: The result of the speech
        """
        print(f"Speaking: {text}")
        walkieAudio.speak(text)
        return "Speech completed"
    return speak

@tool(parse_docstring=True)
def think(thought: str) -> str:
    """This is a tool that you can use to think about the task at hand.
    
    Args:
        thought: The thought to think about
    
    Returns:
        str: The result of the thinking
    """
    print(f"Thinking: {thought}")
    return "Thinking completed"


FOLLOW_STOP_DISTANCE = 0.7  # meters – how close the robot approaches the person
APPROACH_DISTANCE = 1.0  # meters – how close before go_to_raised_hand finishes
RAISED_HAND_TIMEOUT = 60.0  # seconds – max time to scan before giving up


def create_follow_person_tool(robot, walkie_vision, walkie_audio: WalkieAudio):
    """Create a tool that continuously follows the nearest person until the user says 'stop'.

    Args:
        robot: WalkieRobot instance for navigation and position helpers.
        walkie_vision: WalkieVision instance for capturing images and detecting objects.
        walkie_audio: WalkieAudio instance for listening to voice commands.

    Returns:
        A langchain tool.
    """

    @tool(parse_docstring=True)
    def follow_person() -> str:
        """Continuously follow the nearest person in view, keeping ~0.7 m distance.

        The robot will detect the biggest visible person each cycle, compute a goal
        position 0.7 m away from them, and navigate there. It keeps looping until
        the user says "stop" (detected via the microphone).

        When to use:
        - The user asks you to follow them or follow a person.
        - "follow me", "come with me", "follow that person", etc.

        When NOT to use:
        - The user just wants to move to a fixed location (use control_actuators).
        - The user wants to look at or identify a person (use use_vision).

        Returns:
            str: Summary of the follow session (stopped by user or error).
        """
        stop_event = threading.Event()

        def _listen_for_stop():
            """Background thread: listen for the word 'stop' via STT."""
            while not stop_event.is_set():
                try:
                    text = walkie_audio.listen(timeout=10.0, min_duration=1.0)
                    if text and "stop" in text.lower():
                        print(f"[follow_person] Heard stop command: '{text}'")
                        stop_event.set()
                        return
                except Exception as e:
                    print(f"[follow_person] STT listener error: {e}")
                    # Brief pause before retrying to avoid tight error loops
                    time.sleep(0.5)

        # Start the STT listener thread
        listener_thread = threading.Thread(target=_listen_for_stop, daemon=True)
        listener_thread.start()
        print("[follow_person] Started following. Say 'stop' to end.")

        iterations = 0
        try:
            while not stop_event.is_set():
                image = walkie_vision.capture()
                if image is None:
                    time.sleep(0.1)
                    continue

                objects = walkie_vision.detect_objects(image)
                persons = [obj for obj in objects if obj.class_name and obj.class_name.lower() == "person"]

                if persons:
                    # Pick the biggest person by bbox area (w * h)
                    biggest_person = max(persons, key=lambda obj: (obj.bbox[2] * obj.bbox[3]))

                    # Get world position of the person
                    target_pos = robot.tools.bboxes_to_positions([biggest_person.bbox])[0]
                    curr_pos = robot.status.get_pose()

                    tx, ty = target_pos[0], target_pos[1]
                    rx, ry = curr_pos["x"], curr_pos["y"]

                    dx = rx - tx
                    dy = ry - ty
                    dist = math.sqrt(dx**2 + dy**2)

                    if dist > 0:
                        ratio = FOLLOW_STOP_DISTANCE / dist
                        goal_x = tx + (dx * ratio)
                        goal_y = ty + (dy * ratio)

                        # Face the person
                        angle_to_person = math.atan2(-dy, -dx)
                        robot.nav.go_to(goal_x, goal_y, angle_to_person, blocking=False)

                    iterations += 1
                else:
                    # No person visible – stop moving and wait
                    robot.nav.stop()

                time.sleep(0.1)

        except Exception as e:
            robot.nav.stop()
            stop_event.set()
            listener_thread.join(timeout=2.0)
            return f"Follow person ended due to error: {e}"

        # Clean up
        robot.nav.stop()
        listener_thread.join(timeout=2.0)
        print("[follow_person] Stopped following.")
        return "Stopped following the person as requested."

    return follow_person


def _has_raised_hand(person, min_conf: float = 0.5) -> bool:
    """Check whether a PersonPose has at least one wrist above its shoulder.

    A hand is considered "raised" when the wrist keypoint is higher than the
    corresponding shoulder keypoint in pixel coordinates (y increases
    downward, so wrist.y < shoulder.y means the wrist is above).

    Keypoint indices (COCO):
        5 = left_shoulder,  9  = left_wrist
        6 = right_shoulder, 10 = right_wrist
    """
    kpts = person.keypoints
    if len(kpts) < 17:
        return False
    # Left side: wrist(9) above shoulder(5)
    left = (
        kpts[9].confidence >= min_conf
        and kpts[5].confidence >= min_conf
        and kpts[9].y < kpts[5].y
    )
    # Right side: wrist(10) above shoulder(6)
    right = (
        kpts[10].confidence >= min_conf
        and kpts[6].confidence >= min_conf
        and kpts[10].y < kpts[6].y
    )
    return left or right


def create_go_to_raised_hand_tool(robot, walkie_vision):
    """Create a tool that scans for a person raising their hand and navigates to them.

    Args:
        robot: WalkieRobot instance for navigation and position helpers.
        walkie_vision: WalkieVision instance (must have pose_provider configured).

    Returns:
        A langchain tool.
    """

    @tool(parse_docstring=True)
    def go_to_raised_hand() -> str:
        """Scan for a person raising their hand and navigate to them.

        The robot continuously captures frames and runs pose estimation.
        When it detects someone with a raised hand (wrist above shoulder),
        it navigates towards that person. The tool finishes when the robot
        is within ~0.7 m of the person, or after 60 seconds if nobody
        raises their hand.

        When to use:
        - The user asks you to go to a person raising their hand.
        - "go to whoever is raising their hand", "approach the person waving"
        - "someone is calling me", "go to the person who needs help"

        When NOT to use:
        - The user wants to follow someone continuously (use follow_person).
        - The user just wants to move to a fixed location (use control_actuators).
        - The user wants to look at or identify a person (use use_vision).

        Returns:
            str: Result message (arrived, timed out, or error).
        """
        start_time = time.time()
        print("[go_to_raised_hand] Scanning for a person with a raised hand...")

        try:
            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > RAISED_HAND_TIMEOUT:
                    robot.nav.stop()
                    print("[go_to_raised_hand] Timed out after {:.0f}s.".format(RAISED_HAND_TIMEOUT))
                    return (
                        f"Timed out after {RAISED_HAND_TIMEOUT:.0f} seconds. "
                        "No person with a raised hand was detected."
                    )

                image = walkie_vision.capture()
                if image is None:
                    time.sleep(0.1)
                    continue

                # Run pose estimation
                poses = walkie_vision.estimate_poses(image)

                # Find persons with a raised hand
                raised = [p for p in poses if _has_raised_hand(p)]

                if not raised:
                    # Nobody raising their hand – pause and wait
                    robot.nav.stop()
                    time.sleep(0.1)
                    continue

                # Pick the biggest person (by bbox area w*h) among those raising a hand
                target = max(raised, key=lambda p: p.bbox[2] * p.bbox[3])

                # Compute world position of the target person
                target_pos = robot.tools.bboxes_to_positions([target.bbox])[0]
                curr_pos = robot.status.get_pose()

                tx, ty = target_pos[0], target_pos[1]
                rx, ry = curr_pos["x"], curr_pos["y"]

                dx = rx - tx
                dy = ry - ty
                dist = math.sqrt(dx ** 2 + dy ** 2)

                # Check if we are close enough
                if dist <= APPROACH_DISTANCE:
                    robot.nav.stop()
                    print(f"[go_to_raised_hand] Arrived! Distance: {dist:.2f} m")
                    return (
                        f"Arrived at the person who raised their hand. "
                        f"Final distance: {dist:.2f} m."
                    )

                # Navigate towards the person, stopping APPROACH_DISTANCE away
                ratio = APPROACH_DISTANCE / dist
                goal_x = tx + (dx * ratio)
                goal_y = ty + (dy * ratio)

                # Face the person
                angle_to_person = math.atan2(-dy, -dx)
                robot.nav.go_to(goal_x, goal_y, angle_to_person, blocking=False)

                time.sleep(0.1)

        except Exception as e:
            robot.nav.stop()
            print(f"[go_to_raised_hand] Error: {e}")
            return f"Go to raised hand ended due to error: {e}"

    return go_to_raised_hand
