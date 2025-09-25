import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# 1. Load CSV


df_raw = pd.read_csv("rooms.csv", parse_dates=['Timestamp'])


# 2. Rule-Based Label Generation

def predict_with_rules(row):
    lights, fans, acs = 0, 0, 0
    occupancy = row['Occupancy']
    temp = row['Temperature']
    hour = row['Timestamp'].hour

    # Rule 1: No one in the room, everything is off
    if occupancy == 0:
        return pd.Series([0, 0, 0], index=['Lights_Rule', 'Fans_Rule', 'ACs_Rule'])

    # --- Light Rules ---
    if hour < 9 or hour >= 17:
        if 1 <= occupancy <= 15: lights = 1
        elif 16 <= occupancy <= 30: lights = 2
        else: lights = 4

    # --- Fan Rules ---
    if temp > 25:
        if 1 <= occupancy <= 20: fans = 2
        else: fans = 4

    # --- AC Rules ---
    if temp > 28 and occupancy > 10:
        if temp > 30 and occupancy > 20:
            acs = 2
        else:
            acs = 1

    return pd.Series([lights, fans, acs], index=['Lights_Rule', 'Fans_Rule', 'ACs_Rule'])


# Apply rules to generate labels
rule_predictions = df_raw.apply(predict_with_rules, axis=1)
df = pd.concat([df_raw, rule_predictions], axis=1)


# 3. Feature Engineering

# Time of the day feature
df['Time of the day'] = df['Timestamp'].dt.hour.apply(
    lambda h: 'Morning' if h < 12 else 'Afternoon' if h < 17 else 'Evening'
)

# Encode occupancy ranges
def get_occupancy_range(n):
    if n == 0: return '0'
    elif 1 <= n <= 10: return '1-10'
    elif 11 <= n <= 50: return '11-50'
    else: return '51-100'

df['Occupancy_range'] = df['Occupancy'].apply(get_occupancy_range)

# Encode categorical features
le_occupancy = LabelEncoder()
le_time = LabelEncoder()

df['Occupancy_enc'] = le_occupancy.fit_transform(df['Occupancy_range'])
df['Time_enc'] = le_time.fit_transform(df['Time of the day'])


# 4. Prepare Training Data

X = df[['Occupancy_enc', 'Time_enc', 'Temperature']]
y_fan = df['Fans_Rule']
y_light = df['Lights_Rule']
y_ac = df['ACs_Rule']


# 5. Train Decision Tree Models

fan_model = DecisionTreeClassifier(random_state=42).fit(X, y_fan)
light_model = DecisionTreeClassifier(random_state=42).fit(X, y_light)
ac_model = DecisionTreeClassifier(random_state=42).fit(X, y_ac)

print("✅ Models trained!")

# 6. Prediction Function

def get_predictions_from_models(occupancy_num, time_str, temp):
    occ_cat = get_occupancy_range(occupancy_num)
    occ_encoded = le_occupancy.transform([occ_cat])
    time_encoded = le_time.transform([time_str])

    new_data = pd.DataFrame({
        'Occupancy_enc': occ_encoded,
        'Time_enc': time_encoded,
        'Temperature': [temp]
    })

    fan_pred = fan_model.predict(new_data)[0]
    light_pred = light_model.predict(new_data)[0]
    ac_pred = ac_model.predict(new_data)[0]

    return f"Fans: {fan_pred}\nLights: {light_pred}\nACs: {ac_pred}"


# 7. Example Usage
print("\n--- Example Prediction ---")
occupancy_input = 45
time_input = 'Afternoon'
temp_input = 29

predictions = get_predictions_from_models(occupancy_input, time_input, temp_input)
print(f"Inputs → Occupancy: {occupancy_input}, Time: {time_input}, Temp: {temp_input}°C")
print(predictions)

if __name__ == "__main__":
    print("\n--- Room Energy Prediction ---")
    
    # Ask user for inputs
    occupancy_input = int(input("Enter occupancy (number of people): "))
    time_input = input("Enter time of day (Morning, Afternoon, Evening: ").strip().capitalize()
    temp_input = float(input("Enter temperature (°C): "))

    # Make prediction
    predictions = get_predictions_from_models(occupancy_input, time_input, temp_input)

    print("\n--- Prediction Result ---")
    print(f"Occupancy: {occupancy_input}, Time: {time_input}, Temp: {temp_input}°C")
    print(predictions)