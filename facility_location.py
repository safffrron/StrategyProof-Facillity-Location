import random
import argparse
import math

random.seed(42)  # Set a seed for reproducibility
#region Arguement Parsing 
parser = argparse.ArgumentParser(
    description="Facility Location Game"
)
parser.add_argument("-r", "--range", type=int, default=100)
parser.add_argument("-n", "--num-agents", type=int, default=10)
parser.add_argument(
    "-m", "--mechanism",
    choices=["median"],
    default="median",
    help="Facility location mechanism to use"
)
parser.add_argument(
    "-c", "--cost-function",
    choices=["egalitarian", "social"],
    default="egalitarian",
    help="Cost function to use"
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug output"
)
args = parser.parse_args()
#endregion

#region Random Agent Generation
def generate_random_agents(num_agents, range_limit):
    agents = []
    for _ in range(num_agents):
        x = random.randint(0, range_limit)
        agents.append(x)
    return agents
#endregion

# region Helper Functions
def median(values):
    values = sorted(values)
    return values[(len(values) - 1) // 2]
# endregion

# region Mechanism Implementation
def mechanism(agent_reports):
    if args.mechanism == "median":
        location = median(agent_reports)
        return location
    else:
        return None
# endregion

# region Cost Calculation
def agent_payoff(agent_reports, agent_index, agent_truth):
    facility_location = mechanism(agent_reports)
    return abs(agent_truth - facility_location)

def objective_cost_function(agent_reports, agent_truths):
    facility_location = mechanism(agent_reports)
    cost = 0
    if args.cost_function == "egalitarian":
        for i, report in enumerate(agent_reports):
            cost = max(cost, abs(agent_truths[i] - facility_location))
    elif args.cost_function == "social":
        for i, report in enumerate(agent_reports):
            cost += abs(agent_truths[i] - facility_location)
        # cost /= len(agent_reports)
    else:
        return None
    return cost

def optimal_cost(agent_truths):
    if(args.cost_function == "egalitarian"):
        min_val = min(agent_truths)
        max_val = max(agent_truths)
        return math.ceil((max_val - min_val) / 2)
    elif args.cost_function == "social":
        facility = median(agent_truths)
        return sum(abs(x - facility) for x in agent_truths)
        
# endregion

# region Strategic Agent Reporting
def strategic_reporting(agent_reports, index, agent_truths):
    original_report = agent_reports[index]
    best_report = original_report
    best_payoff = agent_payoff(agent_reports, index, agent_truths[index])
    for x in range(args.range + 1):
        agent_reports[index] = x
        new_payoff = agent_payoff(agent_reports, index, agent_truths[index])
        if new_payoff < best_payoff:
            best_payoff = new_payoff
            best_report = x
    agent_reports[index] = original_report  # Restore original report
    return best_report
# endregion

# region Main Execution
if __name__ == "__main__":
    agents = generate_random_agents(args.num_agents, args.range)
    print(f"Generated Agent True Locations: {agents}")

    # Check whether any gain can be made by misreporting if others are reporting truthfully
    is_best_strategy = True
    agent_reports = agents.copy()  # Start with truthful reports
    for i in range(args.num_agents):
        original_report = agent_reports[i]
        best_report = strategic_reporting(agent_reports, i, agents)
        if best_report != original_report:
            is_best_strategy = False
            if(args.debug):
                print(f"Agent {i} can benefit by reporting {best_report} instead of {original_report}.")
        else:
            if(args.debug):
                print(f"Agent {i} has no incentive to misreport.")

    if is_best_strategy:
        print("Reporting true values is best strategy for all agents if other agents are reporting truthfully.")
    else:
        print("There exist agents who can benefit by misreporting.")