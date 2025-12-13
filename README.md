Inventory Simulator Design Document

1. Executive Summary

This document outlines the design for a simple, configurable inventory simulator. The system is built entirely in Python using pandas DataFrames for state management, eliminating the need for an external database. It models a closed-loop system where a fixed number of items are moved between a series of shelves.

The core of the design is the separation between a Simulator and an Observer. The Simulator maintains the absolute ground truth of the inventory at all times. In contrast, the Observer can only see a small part of the system at any given moment, simulating a realistic, partially observable environment. This dual structure allows for the analysis of estimation accuracy and the impact of observation patterns on system knowledge.

2. System Architecture

2.1 Core Components

The system is composed of two primary components that operate in parallel:

The Simulator (Ground Truth): This component has complete knowledge of the inventory state. It manages the location and quantity of every item across all shelves and executes all item movements. Its data represents the "perfect" state of the world, hidden from the observer.
The Observer (Estimates): This component simulates a user or an automated agent attempting to understand the inventory state with limited information. It can only "see" one shelf at a time and must build an estimated model of the entire system based on a sequence of these partial observations.
2.2 Architectural Principles

In-Memory Data: All system states (both for the simulator and the observer) will be managed using pandas DataFrames. This simplifies the design by removing the need for a database.
Conservation of Items: The simulation starts with a fixed total number of items of a single type. These items are only moved between shelves; they are never created or destroyed.
Circular Arrangement: Shelves are arranged in a circular pattern. Every shelf has a neighbor to its "left" and "right". For example, in a 20-shelf system, shelf #19 is adjacent to shelf #18 and shelf #0.
Constrained Movement: An item can only be moved from its current shelf to one of its two immediate neighbors.
Partial, Round-Robin Observation: The Observer inspects shelves sequentially (e.g., shelf #1, #2, ..., #19, #1, ...), getting an accurate count of the currently observed shelf. This creates a natural data staleness, as some shelves will have been observed more recently than others.
Unobservable Location: A designated shelf (e.g., shelf #0) is intentionally never included in the Observer's round-robin schedule. The Observer must infer the contents of this shelf without ever seeing it directly.
3. System Configuration

The simulator is designed to be configurable at initialization.

Number of Shelves: The total number of shelves in the circular arrangement (e.g., 20).
Shelf Capacity: The maximum number of items any single shelf can hold (e.g., 100).
Total Items: The constant, total number of items present in the simulation. This number must be less than or equal to the total system capacity (number of shelves × shelf capacity).
4. Data Structures (Pandas DataFrames)

4.1 Simulator Ground Truth

A single DataFrame will represent the simulator's complete and accurate knowledge of the inventory.

simulator_inventory

Column

Description

Data Type

shelf_id

Unique identifier for the shelf (e.g., 0 to 19).

integer

quantity

The actual quantity of items on that shelf.

integer

4.2 Observer Estimates

A separate DataFrame will represent the observer's estimated knowledge, which is updated at each observation step.

observer_estimates

Column

Description

Data Type

shelf_id

Unique identifier for the shelf.

integer

estimated_quantity

The observer's best guess of the quantity on that shelf.

integer

last_observed_step

The simulation step at which this shelf was last observed. A value of -1 or None can indicate it has never been observed.

integer

uncertainty

A metric representing the staleness of the data. Could be measured as current_step - last_observed_step.

integer

5. Core Workflows

5.1 Initialization

Initialize the simulator_inventory DataFrame with the configured number of shelves.
Distribute the total number of items across the shelves, ensuring no shelf exceeds its capacity.
Initialize the observer_estimates DataFrame with zero or NaN values, as the observer starts with no information.
5.2 Simulator Movement Step

At each step of the simulation, the simulator executes a single item movement:

Randomly select a shelf that is not empty. This is the source_shelf.
Randomly choose a direction (left or right). The neighboring shelf in this direction is the destination_shelf.
If the destination_shelf is not at full capacity, move one item.
Atomically update the simulator_inventory DataFrame:
Decrement the quantity for the source_shelf.
Increment the quantity for the destination_shelf.
5.3 Observer Observation Step

The observer follows a strict round-robin pattern, skipping the unobserved shelf. For a 20-shelf system where shelf #0 is skipped:

The observer's target shelf cycles from 1, 2, 3, ..., 19, and then repeats from 1.
At its step, the observer requests the contents of its target shelf.
The simulator provides the true quantity from simulator_inventory for that specific shelf.
The observer updates its observer_estimates DataFrame for the observed shelf:
The estimated_quantity is set to the true, observed quantity.
The last_observed_step is updated to the current simulation step.
The uncertainty for this shelf is reset to 0.
For all other shelves, the uncertainty value is incremented by 1.
6. Uncertainty Model

Uncertainty in this system is a direct function of time.

Most Accurate: The most recently observed shelf has an uncertainty of 0 and its estimated_quantity is guaranteed to be correct at that specific moment.
Most Uncertain: In a system with N observable shelves, the shelf that is next on the observation schedule (i.e., the one observed N-1 steps ago) has the highest uncertainty among all observed shelves.
The Unobserved Shelf: Shelf #0 presents a unique challenge. Since it is never directly observed, its estimated_quantity can only be inferred. A possible estimation method is to sum the estimated quantities of all other shelves and subtract this from the known total number of items. The uncertainty for this shelf is always considered maximum.
7. Analytics and Validation

The dual-DataFrame structure is ideal for measuring the observer's performance against the ground truth. At any simulation step, we can:

Calculate Total Error: Compare the observer_estimates DataFrame with the simulator_inventory DataFrame to calculate metrics like Mean Absolute Error (MAE) across all shelves.
MAE = (|observer_estimates.estimated_quantity - simulator_inventory.quantity|).mean()
Track Shelf #0 Accuracy: Specifically measure how well the observer's inference for the unobserved shelf matches the reality in the simulator.
This allows for direct analysis of how different movement patterns or system configurations affect the observer's ability to build an accurate picture of the world.

 