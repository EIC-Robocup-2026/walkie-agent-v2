<!--
This file is appended to the Walkie MAIN agent's system prompt under a
"# Current task: <NAME>" heading. Write challenge-specific instructions only —
the base identity, the no-plain-text `speak` contract, and the delegation rules
(delegate_to_actuator / _vision / _database / _human) are already in place.

To override a SUB-agent's prompt instead, drop a file at
prompts/<agent>.md (vision_agent, actuator_agent, database_agent, human_agent).
Delete this comment when you write the real prompt.
-->

You are competing in a RoboCup challenge. Describe the task's goal, the rules
the referee enforces, and the step-by-step procedure you should follow here.
