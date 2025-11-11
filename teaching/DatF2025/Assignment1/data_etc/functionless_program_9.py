import numpy as np

# Daily temperature measurements (in Celsius) for 4 cold cities over 2 weeks
copenhagen_temps = np.array([5.2, 3.8, 7.1, 9.3, 6.7, 2.9, 4.5, 8.1, 6.3, 5.9, 7.8, 3.2, 4.6, 8.9])
stockholm_temps = np.array([2.1, 0.8, 4.3, 6.7, 3.2, -1.5, 1.8, 5.4, 3.7, 2.9, 4.6, 0.7, 2.1, 6.2])
oslo_temps = np.array([1.8, -0.3, 3.9, 5.8, 2.4, -2.1, 0.9, 4.7, 2.8, 1.6, 3.8, -0.5, 1.4, 5.3])
helsinki_temps = np.array([3.4, 1.7, 5.2, 7.9, 4.8, 0.2, 2.6, 6.3, 4.5, 3.8, 5.7, 1.9, 3.2, 7.1])

all_cities = np.array([copenhagen_temps, stockholm_temps, oslo_temps, helsinki_temps])
city_names = ["Copenhagen", "Stockholm", "Oslo", "Helsinki"]

# Analyze each city's temperature data
for i in range(len(all_cities)):
    city_temps = all_cities[i]
    city = city_names[i]

    # Calculate simple average using numpy
    simple_avg = np.mean(city_temps)
    print(f"Simple average: {simple_avg:.2f}°C")

    # Calculate days where pipe heating is needed (days below 4°C) using numpy
    heating_degree_days = np.sum(city_temps < 4)
    print(f"Pipe heating days: {heating_degree_days}")

# Compare cities - find the warmest overall using numpy
city_averages = np.mean(all_cities, axis=1)  # Calculate average for each city
warmest_city_index = np.argmax(city_averages)  # Find index of maximum average
warmest_avg = city_averages[warmest_city_index]

print(f"Warmest city: {city_names[warmest_city_index]} ({warmest_avg:.2f}°C average)")