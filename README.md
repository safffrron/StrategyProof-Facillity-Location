# CS 6001 Project

## Overview

This repository contains the project work for **CS 6001** course at **IIT Bombay**.

## Author

- **Sivakumar Dogga** — [24m1528@iitb.ac.in](mailto:24m1528@iitb.ac.in)

## Repository Structure

> Project files and documentation will be added as the project progresses.

## facility_location.py
The file has the following default paramters
Number of Agents N = 10
Locations on the line R = | {0,1,2...100} |
[Please note that the locations are discrete values in the current setup]
The file checks if other agents are reporting truthfully is any agent incentivised to misreport their value.
The complexity of the code is O(NR).
To show that reporting truthfully is a Dominant Strategy we would need complexity O(NR^N).

The file can be run with:
```bash
python facility_location.py
```
The file takes certain in line parameters:
-c for cost function which may be social or egalitarian.
-n for Number of Agents.
-r for Range on the line.
-m for mechanism which is median.
--debug which shows which agents have incentive and which dont.


# Facility Location Game

A simple implementation of a strategic facility-location game on a discrete line.

## Current Setup

* **Number of agents:** \(N = 10\)
* **Location range:** \(R = \{0,1,2,\ldots,100\}\)
* **Location domain:** Discrete integer values
* **Mechanism:** Median mechanism
* **Cost functions:** Social and Egalitarian

## What Does the Program Do?

The program generates random true locations for \(N\) agents and initially assumes that every agent reports their true location.

For each agent, the program then checks whether they can reduce their own cost by **unilaterally misreporting** their location while all other agents continue to report truthfully.

Each agent's cost is the distance between their **true location** and the facility location:

$$
\text{cost}_i = |x_i - f|
$$

where \(x_i\) is the agent's true location and \(f\) is the facility location determined by the mechanism.

### Complexity

For the current experiment, each of the \(N\) agents tests all \(R+1\) possible reports:

$$
O(NR)
$$

This experiment does **not** by itself prove that truthful reporting is a dominant strategy.

To verify dominant strategy by brute force, we would need to consider every possible combination of reports by the other agents. This requires approximately:

$$
O(NR^N)
$$

which becomes infeasible as \(N\) and \(R\) increase.

## Running the Program

Using the default parameters:

```bash
python facility_location.py
```

### Command-Line Arguments

| Argument                | Description                                     | Default       |
| ----------------------- | ----------------------------------------------- | ------------- |
| `-n`, `--num-agents`    | Number of agents                                | `10`          |
| `-r`, `--range`         | Maximum location on the line                    | `100`         |
| `-m`, `--mechanism`     | Facility-location mechanism                     | `median`      |
| `-c`, `--cost-function` | Cost function (`social` or `egalitarian`)       | `egalitarian` |
| `--debug`               | Show individual agents' incentives to misreport | Off           |

### Examples

Run with 20 agents:

```bash
python facility_location.py -n 20
```

Use social cost:

```bash
python facility_location.py -c social
```

Show individual agent results:

```bash
python facility_location.py --debug
```

Combine parameters:

```bash
python facility_location.py -n 20 -r 200 -c social --debug
```

## Current Mechanism

The current implementation uses the **median mechanism**, which places the facility at the median of the reported agent locations.

More mechanisms can be added using the `--mechanism` argument as the project is extended.


## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Sivadogga007/CS-6001-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CS-6001-Project
   ```

## License

This project is for academic purposes only.
