import numpy as np
from bisect import bisect_left
# from bisect import bisect_right

classification_table: list = [
    ["A", "A", "B", "D", "F"],
    ["A", "B", "C", "D", "E"],
    ["B", "B", "C", "D", "D"],
    ["C", "C", "D", "D", "D"],
    ["C", "D", "D", "D", "F"],
]
classification_label: list[str] = sorted({x for row in classification_table for x in row})
windspeed_thresholds: list = [0.3, 2, 3, 4, 6]
wether_conditions: dict = {
    "clear": 0,
    "fine": 0,
    "sunny": 1,
    "cloudy": 2,
    "scattered": 2,
    "overcast": 3,
    "dark": 3,
    "rain": 4,
    "snow": 4,
    "night": 4,
}
conditions_wether: dict = {
    0 : "clear/fine",
    1 : "sunny",
    2 : "cloudy/scattered",
    3 : "overcast",
    4 : "rain/night/snow"
}


def classify_atomosphere_stability(windspeed: float, wether: str):
    row = bisect_left(windspeed_thresholds, windspeed)-1
    if row < 0:
        raise ValueError("WINDSPEED ERROR!")
    col = wether_conditions.get(wether, 4)
    stability_class = classification_table[row][col]
    return stability_class

def inverse_stab_class_to_wether(windspeed: float, stab_class: str):
    row = bisect_left(windspeed_thresholds, windspeed)-1
    if row < 0:
        raise ValueError("WINDSPEED ERROR!")
    idx = [i for i, x in enumerate(classification_table[row]) if x==stab_class] 
    weather_list = [conditions_wether.get(i, "unknown") for i in idx]
    return weather_list
