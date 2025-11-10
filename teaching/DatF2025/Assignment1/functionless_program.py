# Daily temperature measurements (in Celsius) for 4 cold cities over 2 weeks
copenhagen_temps = [5.2, 3.8, 7.1, 9.3, 6.7, 2.9, 4.5, 8.1, 6.3, 5.9, 7.8, 3.2, 4.6, 8.9]
stockholm_temps = [2.1, 0.8, 4.3, 6.7, 3.2, -1.5, 1.8, 5.4, 3.7, 2.9, 4.6, 0.7, 2.1, 6.2]
oslo_temps = [1.8, -0.3, 3.9, 5.8, 2.4, -2.1, 0.9, 4.7, 2.8, 1.6, 3.8, -0.5, 1.4, 5.3]
helsinki_temps = [3.4, 1.7, 5.2, 7.9, 4.8, 0.2, 2.6, 6.3, 4.5, 3.8, 5.7, 1.9, 3.2, 7.1]

all_cities = [copenhagen_temps, stockholm_temps, oslo_temps, helsinki_temps]
city_names = ["Copenhagen", "Stockholm", "Oslo", "Helsinki"]

# Analyze each city's temperature data
for i in range(len(all_cities)):
    city_temps = all_cities[i]
    city = city_names[i]
    
    # Also calculate simple average for comparison
    simple_total = 0
    for temp in city_temps:
        simple_total = simple_total + temp
    simple_avg = simple_total / len(city_temps)

    print(f"Simple average: {simple_avg:.2f}°C")
    
    # Calculate days where pipe heating is needed (days below 4°C)
    heating_degree_days = 0
    for temp in city_temps:
        if temp < 4:
            heating_degree_days += 1
    
    print(f"Pipe heating days: {heating_degree_days:.1f}")

# Compare cities - find the warmest overall
warmest_city_index = 0
warmest_avg = -999

for i in range(len(all_cities)):
    temps = all_cities[i]
    
    # Calculate average for each city
    city_total = 0
    for temp in temps:
        city_total = city_total + temp
    city_avg = city_total / len(temps)
    
    # Compare to current warmest
    if city_avg > warmest_avg:
        warmest_avg = city_avg
        warmest_city_index = i


print(f"Warmest city: {city_names[warmest_city_index]} ({warmest_avg:.2f}°C average)")