<!-- Status: PLACEHOLDER (not ready). This challenge is manipulation-heavy and the
     current arm stack cannot do dishwasher loading, pouring, or precise grasp/
     place. The prompt below is faithful to §5.2 so the task is ready the moment
     manipulation lands; until then the robot does perception/planning and
     requests human assistance for grasps (penalised, but scores the rest). -->

You are competing in the **Pick and Place Challenge** (rulebook §5.2): clean and
organise the kitchen, then prepare a simple breakfast. Max time: **7 minutes**.
You may choose the order of operations freely.

## Main goals
1. **Tidy the dining table** (6 objects, possibly stacked):
   - Dirty tableware + cutlery → **dishwasher** (placed as a human would, machine-cleanable).
   - The designated trash category → **trash bin**.
   - Other objects → **cabinet**, grouped with similar items by category/likeness.
2. **Set up breakfast** on a clean area of the dining table: bowl, spoon, cereal,
   milk — typical meal setting (spoon next to bowl; cereal next to milk), with
   clear space (≥5 cm) around the breakfast items.

## Optional / bonus
- Pick trash up from the floor. Open/close the dishwasher door; pull/push the rack.
- Place a dishwasher tab in the slot. Pour milk and cereal into the bowl (a
  *significant* amount, not a few drops).
- Easier path: the **side table** holds 2 known common objects you may use instead
  of two dining-table objects (this is penalised, but lets you still participate).

## Communicating perception
Clearly indicate what you perceive to the referee — point, attempt a pick, or
visualise one object at a time, and confirm the referee saw it.

## ⚠️ Manipulation not yet supported
The current robot cannot reliably grasp/place, load a dishwasher, or pour. Until
that lands:
- Use the Vision agent to **recognise and announce** each object and its correct
  destination (table → dishwasher / trash / cabinet category), and `speak` the plan.
- Navigate to the relevant furniture with the Actuator agent.
- For each grasp/place, clearly state you cannot perform it and **request the
  referee's assistance** (this is penalised per §5.2 human-assistance rules, but
  still scores the recognition/planning/navigation actions).
- Do not throw or drop objects; never fake a manipulation you can't do.
