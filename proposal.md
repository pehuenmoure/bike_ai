---
# CS 4701 Project Proposal
#### Kenta Takatsu | Pehuen Moore
---
### Summary
Our project will teach AI how to ride a bicycle.
We picked this agent because the bicycle requires multi-tasking: maneuvering and balancing. We designed this project with objectives that progressively become challenging.

### Objectives
Experiment how the more complicated machine learning algorithms are capable of difficult tasks.
- Single-task agent
  * Objective: Balance the bicycle as long as possible


- Multi-task agent 
  * Objective: Move between points as fast as possible while balancing the agent


- Environment-aware agent
  * Objective: Move maps with obstacles such as walls, blocks. See if bicycle can keep balancing and reach the destination without hitting them.


- Stochastic information + environment aware
  * Objective: In addition to the information above, we give the agent a battery that decrease as time passes. See if bicycle can budget their battery and reach the destination, or move to the station where they can charge the battery.


- Interactive agents
  * Objective: Create multiple agents that are capable of everything above. Teach bicycle how to maneuver while avoiding collisions.


### Algorithms
  * Genetic algorithm
  * Naive bayes network
  * Markov chain
  * Feed-forward neural networks
  * Convolutional neural networks
  * RNN



### Agent
The bicycle agent contains following parameters:
  * Point mass
  * Velocity
  * Acceleration
  * Angular velocity
  * Coordinate information


The bicycle has two dimensional vector that can update each state.
  * Acceleration increment: (-5, +5) (with maximum acceleration)
  * The increment in directional vector: (-10/v, 10/v) (faster it gets, harder it is to change the direction)

Operation is a function that maps environmental information to the 2 dimensional vector.


```
O(s) -> v  s: state, v: 2 dimensional vector
```

The bicycle will get information about its surrounding in a form of the multi-dimensional vector space and returns optimal operation.


state vector s:
  * vector directions to the obstacles
  * vector to the destination
  * Current balance measure



### Evaluation Functions
Evaluation function will encourage the learning phase of reinforcement learning (dictates the gradient decent) or the gene selection of genetic algorithm (fitness)


```
F(s) -> c   c: numeric value from 0 to 1
```

### Learning

### Simulation
