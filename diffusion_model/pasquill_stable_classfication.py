import numpy as np
import bisect

classification_table: list = [
    ["A", "A", "B", "D", "F"],
    ["A", "B", "C", "D", "E"],
    ["B", "B", "C", "D", "D"],
    ["C", "C", "D", "D", "D"],
    ["C", "D", "D", "D", "F"],
]
classification_label: str = sorted({x for row in classification_table for x in row})
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


def classify_atomosphere_stability(windspeed: float, wether: str):
    row = bisect.bisect_left(windspeed_thresholds, windspeed)
    if row == 0:
        Exception("WINDSPEED ERROR!")
    col = wether_conditions.get(wether, 4)
    stability_class = classification_table[row][col]
    return stability_class
