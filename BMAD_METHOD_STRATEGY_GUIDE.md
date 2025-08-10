# BMAD Method Agent Strategy - Complete Reference Guide

## 🎯 BMAD Core Principles (Memorize First)

**B**reakthrough - **M**ethod - **A**gile - **A**I-Driven - **D**evelopment

### The 5 Pillars
1. **Breakthrough**: Innovative problem-solving approaches
2. **Method**: Structured, repeatable processes
3. **Agile**: Rapid iteration and adaptation
4. **AI-Driven**: Intelligent decision-making and optimization
5. **Development**: Focus on working software delivery

## 🤖 BMAD Sub-Agent Architecture

### 8 Domain-Specific Agents

| Agent | Domain | Complexity Target | Primary Focus |
|-------|--------|-------------------|---------------|
| **Data Pipeline Agent** | Data ingestion, processing | ≤3 | ETL, validation, streaming |
| **Kalman Filter Agent** | Mathematical models | ≤3 | Filters, regime detection |
| **Backtesting Agent** | Strategy testing | ≤3 | Simulation, metrics |
| **API Backend Agent** | Server infrastructure | ≤3 | REST/WebSocket APIs |
| **UI Frontend Agent** | User interfaces | ≤3 | React/Streamlit dashboards |
| **Trading Execution Agent** | Order management | ≤3 | Execution, fills |
| **Risk Management Agent** | Portfolio risk | ≤3 | VaR, limits, controls |
| **Testing & Quality Agent** | Validation | ≤3 | Unit/integration tests |

## 📋 BMAD Workflow Process (6 Phases)

### Phase 1: ANALYZE
```bash
task-master analyze-complexity --research
```
- **Goal**: Understand current complexity distribution
- **Output**: Complexity report with task breakdown
- **Action**: Identify tasks with complexity >3

### Phase 2: DECOMPOSE
```bash
task-master expand --id=<complex_task_id> --research --force
```
- **Goal**: Break complex tasks into subtasks ≤3
- **Method**: Use AI research for intelligent breakdown
- **Target**: Each subtask complexity ≤3

### Phase 3: ASSIGN
- **Goal**: Map subtasks to appropriate BMAD agents
- **Strategy**: Match task domain to agent expertise
- **Documentation**: Update task with agent assignment

### Phase 4: COORDINATE
```bash
task-master update-subtask --id=<id> --prompt="agent coordination notes"
```
- **Goal**: Orchestrate multi-agent collaboration
- **Method**: Clear interfaces between agents
- **Tracking**: Progress visibility across agents

### Phase 5: EXECUTE
```bash
task-master set-status --id=<id> --status=in-progress
```
- **Goal**: Implement subtasks using agent specialization
- **Principle**: Each agent focuses on its domain expertise
- **Quality**: Maintain complexity ≤3 per implementation unit

### Phase 6: INTEGRATE
```bash
task-master set-status --id=<id> --status=done
```
- **Goal**: Combine agent outputs into working system
- **Validation**: Test integration points
- **Documentation**: Update system architecture

## 🧠 BMAD Agent Decision Framework

### For Every Task Decision:
1. **Is complexity >3?** → DECOMPOSE
2. **Which agent domain?** → ASSIGN  
3. **What dependencies exist?** → COORDINATE
4. **How to validate?** → TEST
5. **What's the next step?** → ITERATE

## 💡 BMAD Command Patterns (Memorize These)

### Daily Development Loop
```bash
# Morning: Check progress
task-master list
task-master next

# Work: Focus on single agent domain
task-master show <id>
task-master set-status --id=<id> --status=in-progress
# ... implement ...
task-master update-subtask --id=<id> --prompt="implementation notes"
task-master set-status --id=<id> --status=done

# Evening: Plan next day
task-master next
task-master analyze-complexity --research
```

### Weekly Planning Cycle
```bash
# Monday: Analyze and decompose
task-master analyze-complexity --research
task-master expand --all --research

# Mid-week: Execute focused work
task-master next
task-master set-status --id=<id> --status=in-progress

# Friday: Review and integrate
task-master complexity-report
task-master validate-dependencies
```

### Crisis Management
```bash
# When stuck on complex task
task-master expand --id=<stuck_task> --research --force
task-master add-dependency --id=<new_subtask> --depends-on=<prerequisite>

# When dependencies break
task-master validate-dependencies
task-master update --from=<broken_id> --prompt="fix dependency chain"
```

## 🎯 BMAD Agent Specialization Matrix

### Task-to-Agent Mapping Rules

| Task Type | Primary Agent | Secondary Agent | Complexity Limit |
|-----------|---------------|-----------------|------------------|
| Data fetching | Data Pipeline | - | ≤3 per source |
| Math modeling | Kalman Filter | - | ≤3 per model |
| Strategy testing | Backtesting | Risk Management | ≤3 per test |
| API endpoints | API Backend | Trading Execution | ≤3 per endpoint |
| UI components | UI Frontend | - | ≤3 per component |
| Order execution | Trading Execution | Risk Management | ≤3 per order type |
| Risk calculations | Risk Management | Backtesting | ≤3 per metric |
| Test suites | Testing & Quality | All agents | ≤3 per test module |

## 🚀 BMAD Success Patterns

### The "Complexity Cascade"
1. **Identify** complexity >3 task
2. **Decompose** into ≤3 subtasks  
3. **Assign** to specialized agent
4. **Execute** with domain expertise
5. **Integrate** through clear interfaces
6. **Validate** end-to-end functionality

### The "Agent Handoff Protocol"
1. **Complete** current agent's subtask
2. **Document** output interface/format
3. **Update** task with handoff notes
4. **Notify** next agent (via task-master update)
5. **Validate** interface compatibility

### The "Parallel Development Strategy"
1. **Identify** independent subtasks
2. **Assign** to different agents
3. **Set** clear completion criteria
4. **Monitor** progress in parallel
5. **Integrate** when all complete

## 🧰 BMAD Troubleshooting Playbook

### When Complexity Exceeds Target (>3)
```bash
# Emergency decomposition
task-master expand --id=<complex_task> --research --force
task-master add-task --prompt="break down <specific_component>" --research
```

### When Agent Domains Overlap
1. **Primary agent** owns the task
2. **Secondary agent** provides consultation
3. **Document** the decision in task notes
4. **Update** agent assignment clearly

### When Dependencies Create Bottlenecks
```bash
# Identify critical path
task-master validate-dependencies
# Restructure if needed
task-master move --from=<bottleneck_id> --to=<new_parent>
# Add parallel work
task-master add-task --prompt="parallel implementation for <component>" --research
```

## 📊 BMAD Metrics & KPIs

### Complexity Metrics (Track Weekly)
- **Tasks with complexity >3**: Target = 0
- **Average subtask complexity**: Target ≤ 2.5
- **Agent workload distribution**: Target = Balanced

### Velocity Metrics (Track Daily)
- **Subtasks completed per day**: Track trend
- **Agent specialization ratio**: >80% tasks in primary domain
- **Integration success rate**: >95% first-time success

### Quality Metrics (Track per Release)
- **Test coverage per agent**: >80%
- **Cross-agent integration tests**: 100% critical paths
- **Documentation completeness**: 100% public interfaces

## 🎓 BMAD Mastery Levels

### Level 1: Basic (Week 1-2)
- ✅ Use all 6 core task-master commands
- ✅ Understand 8 agent domains
- ✅ Apply complexity ≤3 rule consistently

### Level 2: Intermediate (Week 3-4)
- ✅ Design multi-agent workflows
- ✅ Handle complex decomposition scenarios
- ✅ Coordinate dependencies effectively

### Level 3: Advanced (Month 2+)
- ✅ Optimize agent specialization
- ✅ Predict and prevent bottlenecks
- ✅ Design new agent architectures

### Level 4: Expert (Month 3+)
- ✅ Teach BMAD to others
- ✅ Innovate new BMAD patterns
- ✅ Scale BMAD across projects

## 📝 BMAD Quick Reference Card

### Essential Commands (Print & Keep)
```bash
# Analysis
task-master analyze-complexity --research
task-master complexity-report

# Task Management  
task-master list
task-master next
task-master show <id>

# Status Updates
task-master set-status --id=<id> --status=<pending|in-progress|done>
task-master update-subtask --id=<id> --prompt="<notes>"

# Decomposition
task-master expand --id=<id> --research --force
task-master add-task --prompt="<description>" --research

# Coordination
task-master add-dependency --id=<id> --depends-on=<dep_id>
task-master validate-dependencies
```

### Agent Assignment Template
```
Agent: [Agent Name]
Domain: [Primary Domain]
Complexity: [≤3]
Dependencies: [List]
Interface: [Input/Output specification]
Testing: [Validation approach]
```

## 🎯 BMAD Memory Palace Technique

### Visualization for Memorization
1. **BMAD Building**: 5-story building (5 principles)
2. **Agent Floor**: 8 rooms (8 agents) 
3. **Workflow Elevator**: 6 stops (6 phases)
4. **Complexity Gauge**: Always shows ≤3
5. **Command Terminal**: Quick reference station

### Memory Anchors
- **B**ig **M**ath **A**rchitecture **A**gent **D**elivers
- **8 Agents** = 2³ (easy to remember)
- **Complexity ≤3** = Magic number for human cognition
- **6 Phases** = Complete workflow cycle

## 🚀 BMAD Implementation Checklist

### Before Starting Any Task:
- [ ] Check if complexity >3
- [ ] Identify appropriate agent
- [ ] Verify dependencies clear
- [ ] Confirm interface specifications
- [ ] Plan testing approach

### During Task Execution:
- [ ] Stay within agent domain
- [ ] Keep complexity ≤3
- [ ] Document progress regularly
- [ ] Test incrementally
- [ ] Communicate handoffs clearly

### After Task Completion:
- [ ] Update task status
- [ ] Document outputs
- [ ] Validate integrations
- [ ] Update complexity metrics
- [ ] Plan next iteration

---

## 🎖️ BMAD Mastery Achievement

**You have mastered BMAD when you can:**
1. **Instantly identify** which agent handles any task
2. **Automatically decompose** complexity >3 into ≤3 subtasks
3. **Seamlessly coordinate** multi-agent workflows
4. **Consistently deliver** working software through agent specialization
5. **Teach others** the BMAD methodology

**Remember**: BMAD isn't just a method—it's a mindset for scalable, intelligent software development! 🚀

---

*Print this guide and keep it handy during development. The more you use BMAD patterns, the more natural they become!*