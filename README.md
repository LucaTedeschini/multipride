# [Multipride Task - EVALITA 2026](https://multipride-evalita.github.io/)

## Task Description
This is a binary classification task where systems must identify whether a term related to the LGBTQ+ context is used with **reclamatory** intent.  
In other words, the goal is to detect when a term (including potentially derogatory ones) is being used *non-discriminatorily*, as a form of self-identification or community belonging.

The task consists of two subtasks:

---

### Task A
In this subtask, the model has access **only to the tweet text**.  
Participants can approach the task in either a *constrained* or *unconstrained* setting:

- **Constrained**: Additional training data is **not** allowed. However, external resources (e.g., lexicons) may be used.
- **Unconstrained**: Participants **may** use additional training data.  
  This choice **must be explicitly declared** upon submission.

This subtask includes datasets in three languages: **Italian**, **English**, and **Spanish**.  
It is also possible to train a multilingual system by combining datasets, although no official multilingual leaderboard is provided (this approach is still encouraged).

---

### Task B
In this subtask, in addition to the tweet text, the model also has access to the *user bio* as contextual information.

This subtask is available only for **Italian** and **Spanish**.

---

## Repository Structure
This repository addresses all subtasks.  
To maintain clarity, each subtask is developed in a separate branch:

- `Task_A`
- `Task_B`
