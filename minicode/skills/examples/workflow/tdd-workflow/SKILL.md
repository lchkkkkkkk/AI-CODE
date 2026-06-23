---
name: tdd-workflow
description: Test-Driven Development workflow - write tests first, then implement
domain: testing
layer: workflow
tags: [tdd, test, testing, workflow, development, red-green-refactor, unittest, pytest]
intents:
  - write tests first
  - test driven development
  - TDD workflow
  - add tests for
  - implement with tests
  - 先写测试
  - 测试驱动开发
  - TDD 流程
  - 用 TDD 方式
  - 写单元测试
boundary:
  can_use:
    - is_python_project
  cannot_use: []
input_examples:
  - "用 TDD 方式实现用户登录功能"
  - "先写测试再写代码"
  - "add tests for the user service"
  - "write tests first, then implement"
  - "帮我用测试驱动的方式开发这个模块"
version: "1.0.0"
priority: 4
---

# TDD Workflow Skill

Follow the Red-Green-Refactor cycle:

## Step 1: RED - Write a failing test
- Create or update a test file in the `tests/` directory.
- Write a minimal test that captures the desired behavior.
- Run the test to confirm it **fails** (red).

## Step 2: GREEN - Make it pass
- Write the minimum code needed to make the test pass.
- Do NOT over-engineer — just enough to go green.
- Run the test again to confirm it **passes** (green).

## Step 3: REFACTOR - Clean up
- Improve the code structure without changing behavior.
- Remove duplication, improve naming, add type hints.
- Run the test again — it must stay green.
- If you add new tests during refactoring, cycle back to RED.

## Guiding Principles
- One test at a time. Don't write multiple tests before any implementation.
- Tests should be independent and fast.
- Use `test_runner` tool to discover and run tests.
- If the project uses pytest, follow pytest conventions.
- If the project uses unittest, follow unittest conventions.
