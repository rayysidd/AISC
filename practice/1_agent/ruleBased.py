class RuleBasedAgent:
    def __init__(self, rule_name):
        self.agent_name = rule_name
        # You can store simple thresholds here
        self.thresholds = {
            'cold_limit': 15,
            'hot_limit': 30,
            'windy_limit': 20
        }

    def evaluate(self, inputs):
        """
        inputs: dict containing the environment state
        e.g., {'temp': 12, 'raining': True, 'wind_speed': 5}
        """
        recommendations = []
        
        # --- START: MODIFY RULES HERE FOR YOUR PROBLEM ---
        
        # Rule 1: Temperature Logic (Range-based)
        temp = inputs.get('temp', 20) # Default to 20 if missing
        if temp < self.thresholds['cold_limit']:
            recommendations.append("Wear a heavy coat and scarf.")
        elif temp > self.thresholds['hot_limit']:
            recommendations.append("Wear shorts and a light t-shirt.")
        else:
            recommendations.append("Jeans and a light jacket are fine.")

        # Rule 2: Boolean Logic (Yes/No)
        if inputs.get('raining') == True:
            recommendations.append("Don't forget an Umbrella!")

        # Rule 3: Compound Logic (AND/OR)
        # Example: If it's cold AND windy
        if temp < 15 and inputs.get('wind_speed') > self.thresholds['windy_limit']:
            recommendations.append("ALERT: Wind chill is high. Wear thermal layers.")
            
        # --- END RULES ---

        return recommendations

# --- Usage Example (Batch A2) ---
agent = RuleBasedAgent("Outfit Advisor")

# Case 1: Cold and Rainy
current_weather = {'temp': 10, 'raining': True, 'wind_speed': 5}
print(f"Scenario 1 ({current_weather}):")
results = agent.evaluate(current_weather)
for r in results:
    print(f"- {r}")

print("-" * 20)

# Case 2: Hot and Dry
summer_weather = {'temp': 35, 'raining': False, 'wind_speed': 10}
print(f"Scenario 2 ({summer_weather}):")
results = agent.evaluate(summer_weather)
for r in results:
    print(f"- {r}")