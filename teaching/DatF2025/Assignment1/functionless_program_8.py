# Daily temperature measurements (in Celsius) for 4 cold cities over 2 weeks
copenhagen_temps = [5.2, 3.8, 7.1, 9.3, 6.7, 2.9, 4.5, 8.1, 6.3, 5.9, 7.8, 3.2, 4.6, 8.9]
stockholm_temps = [2.1, 0.8, 4.3, 6.7, 3.2, -1.5, 1.8, 5.4, 3.7, 2.9, 4.6, 0.7, 2.1, 6.2]
oslo_temps = [1.8, -0.3, 3.9, 5.8, 2.4, -2.1, 0.9, 4.7, 2.8, 1.6, 3.8, -0.5, 1.4, 5.3]
helsinki_temps = [3.4, 1.7, 5.2, 7.9, 4.8, 0.2, 2.6, 6.3, 4.5, 3.8, 5.7, 1.9, 3.2, 7.1]

all_cities = [copenhagen_temps, stockholm_temps, oslo_temps, helsinki_temps]
city_names = ["Copenhagen", "Stockholm", "Oslo", "Helsinki"]

def calculate_simple_average(temps):
    """Calculate the simple average of a list of temperatures."""
    total = 0
    for temp in temps:
        total = total + temp
    return total / len(temps)

def calculate_pipe_heating_days(temps, threshold=4):
    """Count days where temperature is below the threshold (pipe heating needed)."""
    heating_days = 0
    for temp in temps:
        if temp < threshold:
            heating_days += 1
    return heating_days

def find_warmest_city(all_cities, city_names):
    """Find the city with the highest average temperature."""
    warmest_city_index = 0
    warmest_avg = -999

    for i in range(len(all_cities)):
        temps = all_cities[i]
        city_avg = calculate_simple_average(temps)

        if city_avg > warmest_avg:
            warmest_avg = city_avg
            warmest_city_index = i

    return warmest_city_index, warmest_avg

# Analyze each city's temperature data
for i in range(len(all_cities)):
    city_temps = all_cities[i]
    city = city_names[i]

    # Calculate and display simple average
    simple_avg = calculate_simple_average(city_temps)
    print(f"Simple average: {simple_avg:.2f}°C")

    # Calculate and display pipe heating days
    heating_days = calculate_pipe_heating_days(city_temps)
    print(f"Pipe heating days: {heating_days}")

# Compare cities - find the warmest overall
warmest_index, warmest_avg = find_warmest_city(all_cities, city_names)
print(f"Warmest city: {city_names[warmest_index]} ({warmest_avg:.2f}°C average)")