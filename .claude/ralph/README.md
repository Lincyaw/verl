# Ralph Agent Workflow

> This is a simple AI coding agent workflow based on a Bash loop, inspired by the "Ralph" concept. It uses an LLM to iteratively work through a backlog of tasks, maintaining context and verifying code via tests.

---

## File Structure

Ensure your project contains the following structure (relative to `.claude/ralph/`):

```
.claude/ralph/
├── ralph.sh            # Main automation loop script
├── progress.txt        # Agent's persistent memory (must be in .claude/ralph)
├── README.md           # This documentation
└── plans/
    └── prd.json        # Task backlog (must be inside 'plans' folder)
```


## Specifications

### 1. `plans/prd.json` (The Backlog)

This file acts as the state of truth for what needs to be built. It is a valid JSON file containing an array of task objects.

**Structure:**

```json
[
  {
    "id": "string (unique identifier)",
    "story": "string (feature title or user story)",
    "priority": "string ('high' | 'medium' | 'low')",
    "requirements": [
      "string (specific verification step 1)",
      "string (specific verification step 2)"
    ],
    "passes": false
  }
]
```

**Example:**

```json
[
  {
    "story": "User Login",
    "priority": "high",
    "requirements": ["Users can log in with email/pass", "Show error on failure"],
    "passes": false
  }
]
```

---

## Prompt

Please refer to `.claude/ralph/README.md` for the specifications of the Ralph agent workflow, including the required file structure and formats for `.claude/ralph/plans/prd.json` and `progress.txt`. You need to write your plan to `.claude/ralph/plans/prd.json`.

### 2. `progress.txt` (The Context)

This file acts as the agent's long-term memory across iterations. It is an append-only plain text file.

**Usage:**

- **Initial State:** Populate it with project context, tech stack, and important commands.
- **During Loop:** The agent appends a summary of what it completed, any technical decisions made, and notes for the "next" agent (itself in the next loop).
- **Format:** Free-form text, usually separated by headers or dividers (e.g., `---`).
