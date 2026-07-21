"""
TASK PLANNER — Goal Decomposition & Execution Planning

Features:
    - Break large goals into smaller actionable steps
    - Prioritize tasks by dependency and importance
    - Create execution plans with ordering
    - Support iterative planning (refine plans as you go)
    - Store plan history for review

Architecture:
    Goal → Decompose → Prioritize → Order → Execute Plan

    The planner takes a high-level goal and produces an ordered
    list of tasks with dependencies, priority scores, and status
    tracking. Plans can be iteratively refined as tasks complete.

Usage:
    planner = TaskPlanner()
    plan = planner.create_plan("Learn AI engineering")
    plan.print_plan()
    plan.complete_task(1)
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

# Project root sys.path hack removed from module level (M-1)


# ──────────────────────────────────────────────
#  ENUMS
# ──────────────────────────────────────────────

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class PlanningStrategy(Enum):
    """Different approaches to decomposing goals."""
    SEQUENTIAL = "sequential"       # Linear order, one after another
    PARALLEL = "parallel"           # Independent tasks can run together
    ITERATIVE = "iterative"         # Refine in cycles
    HIERARCHICAL = "hierarchical"   # Sub-goals with nested tasks


# ──────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────

@dataclass
class Task:
    """
    A single actionable step in a plan.

    Attributes:
        id:           Unique task identifier
        title:        Short description of the task
        description:  Detailed explanation
        priority:     TaskPriority level
        status:       Current TaskStatus
        dependencies: Task IDs that must complete first
        estimated_effort: Rough effort estimate (low/medium/high)
        tags:         Categorization tags
        result:       Output/result after completion
    """
    id: int
    title: str
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[int] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    tags: list[str] = field(default_factory=list)
    result: str = ""

    @property
    def is_ready(self) -> bool:
        """Whether this task can be started (all deps completed)."""
        if self.status != TaskStatus.PENDING:
            return False
        return True  # Dependencies checked externally

    @property
    def is_done(self) -> bool:
        """Whether this task is finished."""
        return self.status == TaskStatus.COMPLETED


@dataclass
class Plan:
    """
    An execution plan — a collection of ordered tasks.

    Attributes:
        goal:           The original high-level goal
        strategy:       How tasks are organized
        tasks:          List of Task objects
        created_at:     When the plan was created
        history:        Log of plan changes
    """
    goal: str
    strategy: PlanningStrategy = PlanningStrategy.SEQUENTIAL
    tasks: list[Task] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    history: list[str] = field(default_factory=list)

    def get_task(self, task_id: int) -> Optional[Task]:
        """Find a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to start (deps satisfied)."""
        completed_ids = {
            t.id for t in self.tasks if t.status == TaskStatus.COMPLETED
        }

        ready = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            if all(d in completed_ids for d in task.dependencies):
                ready.append(task)

        # Sort by priority (highest first)
        ready.sort(key=lambda t: t.priority.value, reverse=True)
        return ready

    def get_blocked_tasks(self) -> list[Task]:
        """Get tasks that are blocked by incomplete dependencies."""
        completed_ids = {
            t.id for t in self.tasks if t.status == TaskStatus.COMPLETED
        }

        blocked = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            if any(d not in completed_ids for d in task.dependencies):
                blocked.append(task)

        return blocked

    def complete_task(self, task_id: int, result: str = ""):
        """Mark a task as completed."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.result = result
            self.history.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Completed task {task_id}: {task.title}"
            )

    def start_task(self, task_id: int):
        """Mark a task as in progress."""
        task = self.get_task(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.IN_PROGRESS
            self.history.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Started task {task_id}: {task.title}"
            )

    def skip_task(self, task_id: int, reason: str = ""):
        """Skip a task."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.SKIPPED
            self.history.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Skipped task {task_id}: {task.title} ({reason})"
            )

    @property
    def progress(self) -> float:
        """Overall completion percentage."""
        if not self.tasks:
            return 0.0
        completed = sum(
            1 for t in self.tasks if t.status == TaskStatus.COMPLETED
        )
        return completed / len(self.tasks)

    @property
    def is_complete(self) -> bool:
        """Whether all tasks are done or skipped."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for t in self.tasks
        )

    def summary(self) -> dict:
        """Plan statistics."""
        status_counts = {}
        for task in self.tasks:
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "goal": self.goal,
            "strategy": self.strategy.value,
            "total_tasks": len(self.tasks),
            "status": status_counts,
            "progress": f"{self.progress:.0%}",
            "is_complete": self.is_complete,
            "created_at": self.created_at,
        }

    def print_plan(self):
        """Pretty-print the full plan."""
        print("\n" + "=" * 60)
        print(f"  📋 PLAN: {self.goal}")
        print("=" * 60)
        print(f"   Strategy: {self.strategy.value}")
        print(f"   Progress: {self.progress:.0%}")
        print()

        for task in self.tasks:
            status_icon = {
                TaskStatus.PENDING: "⬜",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.SKIPPED: "⏭️",
            }.get(task.status, "•")

            priority_icon = {
                TaskPriority.CRITICAL: "🔴",
                TaskPriority.HIGH: "🟠",
                TaskPriority.MEDIUM: "🟡",
                TaskPriority.LOW: "🟢",
            }.get(task.priority, "")

            deps = f" (after: {', '.join(map(str, task.dependencies))})" if task.dependencies else ""

            print(f"   {status_icon} [{task.id}] {priority_icon} {task.title}{deps}")
            if task.description:
                print(f"       {task.description[:80]}")

        print(f"\n   History ({len(self.history)} events):")
        for event in self.history[-5:]:  # Last 5 events
            print(f"     {event}")
        print()


# ──────────────────────────────────────────────
#  GOAL DECOMPOSER
# ──────────────────────────────────────────────

class GoalDecomposer:
    """
    Break a high-level goal into actionable tasks.

    Uses pattern matching on goal keywords to select
    appropriate decomposition templates.
    """

    # Decomposition templates for common goal types
    TEMPLATES = {
        "learn": {
            "strategy": PlanningStrategy.ITERATIVE,
            "pattern": r"\b(learn|study|understand|master)\b",
            "steps": [
                ("Gather knowledge", "Read and collect relevant information", TaskPriority.HIGH, []),
                ("Extract key concepts", "Identify and list important concepts", TaskPriority.HIGH, [1]),
                ("Organize knowledge", "Structure concepts into a coherent framework", TaskPriority.MEDIUM, [2]),
                ("Create practice tasks", "Design exercises to reinforce understanding", TaskPriority.MEDIUM, [3]),
                ("Review and refine", "Iterate on knowledge and fill gaps", TaskPriority.MEDIUM, [4]),
            ],
        },
        "build": {
            "strategy": PlanningStrategy.SEQUENTIAL,
            "pattern": r"\b(build|create|develop|make|implement)\b",
            "steps": [
                ("Define requirements", "Clarify what needs to be built and why", TaskPriority.CRITICAL, []),
                ("Research approach", "Study possible solutions and technologies", TaskPriority.HIGH, [1]),
                ("Design architecture", "Plan the structure and components", TaskPriority.HIGH, [2]),
                ("Implement core", "Build the main functionality", TaskPriority.CRITICAL, [3]),
                ("Test and validate", "Verify the implementation works correctly", TaskPriority.HIGH, [4]),
                ("Refine and document", "Polish and write documentation", TaskPriority.MEDIUM, [5]),
            ],
        },
        "analyze": {
            "strategy": PlanningStrategy.SEQUENTIAL,
            "pattern": r"\b(analyze|review|examine|investigate|audit)\b",
            "steps": [
                ("Collect data", "Gather all relevant information and sources", TaskPriority.HIGH, []),
                ("Identify patterns", "Find recurring themes and structures", TaskPriority.HIGH, [1]),
                ("Evaluate findings", "Assess quality and relevance of findings", TaskPriority.MEDIUM, [2]),
                ("Draw conclusions", "Synthesize insights and recommendations", TaskPriority.HIGH, [3]),
                ("Report results", "Document the analysis and conclusions", TaskPriority.MEDIUM, [4]),
            ],
        },
        "solve": {
            "strategy": PlanningStrategy.ITERATIVE,
            "pattern": r"\b(solve|fix|debug|resolve|troubleshoot)\b",
            "steps": [
                ("Reproduce the issue", "Understand and recreate the problem", TaskPriority.CRITICAL, []),
                ("Identify root cause", "Trace the source of the problem", TaskPriority.CRITICAL, [1]),
                ("Design solution", "Plan the fix approach", TaskPriority.HIGH, [2]),
                ("Implement fix", "Apply the solution", TaskPriority.HIGH, [3]),
                ("Verify resolution", "Confirm the problem is solved", TaskPriority.CRITICAL, [4]),
            ],
        },
    }

    def decompose(
        self,
        goal: str,
        strategy: Optional[PlanningStrategy] = None,
        custom_tasks: Optional[list[tuple]] = None,
    ) -> Plan:
        """
        Decompose a goal into a task plan.

        Args:
            goal:        The high-level goal statement
            strategy:    Override the auto-detected strategy
            custom_tasks: Manually specify tasks as list of
                         (title, description, priority, dependencies)

        Returns:
            Plan object with ordered tasks
        """
        if custom_tasks:
            return self._build_plan_from_custom(goal, custom_tasks, strategy)

        # Find matching template
        template = self._match_template(goal)

        if template:
            return self._build_plan_from_template(goal, template, strategy)

        # Generic fallback
        return self._build_generic_plan(goal, strategy)

    def _match_template(self, goal: str) -> Optional[dict]:
        """Find the best matching template for the goal."""
        import re
        goal_lower = goal.lower()

        for name, tmpl in self.TEMPLATES.items():
            if re.search(tmpl["pattern"], goal_lower):
                return tmpl

        return None

    def _build_plan_from_template(
        self,
        goal: str,
        template: dict,
        strategy_override: Optional[PlanningStrategy] = None,
    ) -> Plan:
        """Build a plan from a matched template."""
        strategy = strategy_override or template["strategy"]
        plan = Plan(goal=goal, strategy=strategy)

        for i, (title, desc, priority_str, deps) in enumerate(template["steps"], 1):
            priority = TaskPriority[priority_str.upper()] if isinstance(priority_str, str) else priority_str
            task = Task(
                id=i,
                title=title,
                description=desc,
                priority=priority,
                dependencies=deps,
            )
            plan.tasks.append(task)

        plan.history.append(f"Plan created from '{template.get('pattern', 'custom')}' template")
        return plan

    def _build_plan_from_custom(
        self,
        goal: str,
        custom_tasks: list[tuple],
        strategy: Optional[PlanningStrategy] = None,
    ) -> Plan:
        """Build a plan from custom task specifications."""
        plan = Plan(
            goal=goal,
            strategy=strategy or PlanningStrategy.SEQUENTIAL,
        )

        for i, (title, desc, priority_str, deps) in enumerate(custom_tasks, 1):
            priority = TaskPriority[priority_str.upper()] if isinstance(priority_str, str) else priority_str
            task = Task(
                id=i,
                title=title,
                description=desc,
                priority=priority,
                dependencies=deps,
            )
            plan.tasks.append(task)

        plan.history.append("Plan created from custom tasks")
        return plan

    def _build_generic_plan(
        self,
        goal: str,
        strategy: Optional[PlanningStrategy] = None,
    ) -> Plan:
        """Build a generic plan when no template matches."""
        plan = Plan(
            goal=goal,
            strategy=strategy or PlanningStrategy.ITERATIVE,
        )

        generic_steps = [
            ("Understand the goal", "Clarify requirements and scope", TaskPriority.HIGH, []),
            ("Research and gather info", "Collect relevant knowledge and resources", TaskPriority.HIGH, [1]),
            ("Break into sub-tasks", "Decompose into smaller actionable items", TaskPriority.HIGH, [2]),
            ("Execute plan", "Work through each sub-task systematically", TaskPriority.CRITICAL, [3]),
            ("Review results", "Evaluate outcomes and identify improvements", TaskPriority.MEDIUM, [4]),
        ]

        for i, (title, desc, priority_str, deps) in enumerate(generic_steps, 1):
            priority = TaskPriority[priority_str.upper()] if isinstance(priority_str, str) else priority_str
            task = Task(
                id=i,
                title=title,
                description=desc,
                priority=priority,
                dependencies=deps,
            )
            plan.tasks.append(task)

        plan.history.append("Plan created from generic template")
        return plan

    def refine_plan(
        self,
        plan: Plan,
        new_tasks: list[tuple],
        after_task_id: int = 0,
    ) -> Plan:
        """
        Add new tasks to an existing plan (iterative planning).

        Args:
            plan:          The existing plan
            new_tasks:     New tasks as (title, desc, priority, deps)
            after_task_id: Insert after this task ID

        Returns:
            Updated Plan
        """
        max_id = max((t.id for t in plan.tasks), default=0)

        for i, (title, desc, priority_str, deps) in enumerate(new_tasks, max_id + 1):
            priority = TaskPriority[priority_str.upper()] if isinstance(priority_str, str) else priority_str

            # Adjust dependencies to account for insertion point
            adjusted_deps = [
                d if d <= after_task_id else d + len(new_tasks)
                for d in deps
            ]

            task = Task(
                id=i,
                title=title,
                description=desc,
                priority=priority,
                dependencies=adjusted_deps,
            )
            plan.tasks.append(task)

        plan.history.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Refined plan: added {len(new_tasks)} tasks after task {after_task_id}"
        )

        return plan


# ──────────────────────────────────────────────
#  PLAN HISTORY
# ──────────────────────────────────────────────

class PlanHistory:
    """
    Store and manage multiple plans over time.

    Allows tracking plan evolution, comparing versions,
    and reviewing completed goals.
    """

    def __init__(self):
        self._plans: list[Plan] = []

    def add_plan(self, plan: Plan):
        """Add a plan to history."""
        self._plans.append(plan)

    def get_latest(self) -> Optional[Plan]:
        """Get the most recent plan."""
        return self._plans[-1] if self._plans else None

    def get_completed(self) -> list[Plan]:
        """Get all completed plans."""
        return [p for p in self._plans if p.is_complete]

    def get_active(self) -> list[Plan]:
        """Get plans that are still in progress."""
        return [p for p in self._plans if not p.is_complete]

    @property
    def count(self) -> int:
        return len(self._plans)

    def summary(self) -> str:
        """Overall history summary."""
        completed = len(self.get_completed())
        active = len(self.get_active())
        return (
            f"PlanHistory: {self.count} plans total, "
            f"{completed} completed, {active} active"
        )


# ──────────────────────────────────────────────
#  TASK PLANNER (Main Class)
# ──────────────────────────────────────────────

class TaskPlanner:
    """
    Main task planning interface.

    Orchestrates goal decomposition, plan management,
    and plan history.

    Usage:
        planner = TaskPlanner()
        plan = planner.plan("Build a web scraper")
        plan.print_plan()
    """

    def __init__(self):
        self._decomposer = GoalDecomposer()
        self.history = PlanHistory()

    def plan(
        self,
        goal: str,
        strategy: Optional[PlanningStrategy] = None,
        custom_tasks: Optional[list[tuple]] = None,
    ) -> Plan:
        """
        Create a new execution plan for a goal.

        Args:
            goal:        The high-level goal
            strategy:    Override auto-detected strategy
            custom_tasks: Manually specify tasks

        Returns:
            Plan object
        """
        plan = self._decomposer.decompose(
            goal, strategy=strategy, custom_tasks=custom_tasks
        )
        self.history.add_plan(plan)
        return plan

    def refine(
        self,
        plan: Plan,
        new_tasks: list[tuple],
        after_task_id: int = 0,
    ) -> Plan:
        """
        Refine an existing plan with new tasks.

        Args:
            plan:          Existing plan
            new_tasks:     Tasks to add
            after_task_id: Insert after this task

        Returns:
            Updated plan
        """
        return self._decomposer.refine_plan(plan, new_tasks, after_task_id)

    def get_next_actions(self, plan: Plan) -> list[Task]:
        """Get tasks that are ready to execute next."""
        return plan.get_ready_tasks()

    def plan_status(self, plan: Plan) -> dict:
        """Get plan status summary."""
        return plan.summary()


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure project root is on path for standalone execution
    import os
    import sys
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    print("=" * 60)
    print("  TASK PLANNER — Quick Test")
    print("=" * 60 + "\n")

    planner = TaskPlanner()

    # ── Test 1: Learn goal ──
    print("--- Test 1: Learning Goal ---")
    plan1 = planner.plan("Learn AI engineering")
    plan1.print_plan()

    # Simulate progress
    plan1.start_task(1)
    plan1.complete_task(1, "Read 5 articles on AI engineering")
    plan1.start_task(2)
    plan1.complete_task(2, "Extracted 20 key concepts")
    print(f"   Progress: {plan1.progress:.0%}")

    # ── Test 2: Build goal ──
    print("\n" + "-" * 60)
    print("--- Test 2: Build Goal ---")
    plan2 = planner.plan("Build a web scraper")
    plan2.print_plan()

    # ── Test 3: Custom tasks ──
    print("\n" + "-" * 60)
    print("--- Test 3: Custom Tasks ---")
    plan3 = planner.plan(
        "Deploy ML model",
        custom_tasks=[
            ("Prepare model", "Train and export the model", "critical", []),
            ("Set up server", "Configure the API endpoint", "high", [1]),
            ("Deploy", "Push to production server", "critical", [2]),
            ("Monitor", "Set up logging and alerting", "medium", [3]),
        ],
    )
    plan3.print_plan()

    # ── Test 4: Refine plan ──
    print("\n" + "-" * 60)
    print("--- Test 4: Iterative Refinement ---")
    planner.refine(
        plan3,
        new_tasks=[
            ("Load test", "Test under high traffic", "high", [3]),
            ("Rollback plan", "Prepare rollback procedure", "critical", [3]),
        ],
        after_task_id=3,
    )
    plan3.print_plan()

    # ── History ──
    print("\n" + "-" * 60)
    print("--- Plan History ---")
    print(f"   {planner.history.summary()}")
