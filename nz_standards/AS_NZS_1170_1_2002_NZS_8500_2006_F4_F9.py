import pandas as pd

barrier_data = pd.DataFrame(
    [
        # Domestic occupancy
        {
            "occupance_type": "Domestic",
            "specific_use_type": "Pool",
            "barrier_type": ["Pool fence"],
            "Building_code_clause": ["NZS8500-2006", "F9"],
            "Horizontal_design_loading": {"value": 0.33, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.2, "unit": "m"},
            "Infill_face_design_loading": {"value": 0.5, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 0.25, "unit": "kN"},
        },
        {
            "occupance_type": "Domestic",
            "specific_use_type": "Stairs",
            "barrier_type": ["Balustrade"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.35, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 0.9, "unit": "m"},
            "Infill_face_design_loading": {"value": 0.5, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 0.25, "unit": "kN"},
        },
        {
            "occupance_type": "Domestic",
            "specific_use_type": "Walkways/Landings",
            "barrier_type": ["Boundary fence", "Retaining wall"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.35, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.0, "unit": "m"},
            "Infill_face_design_loading": {"value": 0.5, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 0.25, "unit": "kN"},
        },
        {
            "occupance_type": "Domestic",
            "specific_use_type": "Decks/Terraces/Balconies",
            "barrier_type": ["Retaining wall", "Balustrade", "Boundary fence"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {
                "Single dwelling": {"value": 1.0, "unit": "m"},
                "Multi dwelling": {"value": 1.1, "unit": "m"},
            },
            "Infill_face_design_loading": {"value": 0.5, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 0.25, "unit": "kN"},
        },
        # Commercial/Industrial occupancy
        {
            "occupance_type": "Commercial/Industrial",
            "specific_use_type": "Stairs",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.35, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
            "Infill_face_design_loading": {"value": 1.0, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 0.5, "unit": "kN"},
        },
        {
            "occupance_type": "Commercial/Industrial",
            "specific_use_type": "Walkways/Landings",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.35, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
            "Infill_face_design_loading": {"value": 1.0, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 0.5, "unit": "kN"},
        },
        {
            "occupance_type": "Commercial/Industrial",
            "specific_use_type": "Decks/Terraces/Balconies",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
            "Infill_face_design_loading": {"value": 1.0, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 0.5, "unit": "kN"},
        },
        # Public occupancy
        {
            "occupance_type": "Public",
            "specific_use_type": "Walkways/Landings",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
            "Infill_face_design_loading": {"value": 1.5, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 1.5, "unit": "kN"},
        },
        {
            "occupance_type": "Public",
            "specific_use_type": "Decks/Terraces/Balconies",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
            "Infill_face_design_loading": {"value": 1.5, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 1.5, "unit": "kN"},
        },
        {
            "occupance_type": "Public",
            "specific_use_type": "Stairs",
            "barrier_type": ["Balustrade"],
            "Building_code_clause": ["F4", "NZS1170-1-2002"],
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Point_design_loading": {"value": 0.6, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
            "Infill_face_design_loading": {"value": 1.5, "unit": "kPa"},
            "Infill_point_design_loading": {"value": 1.5, "unit": "kN"},
        },
    ]
)

def get_horizontal_design_loading(occupance_type, specific_use_type, barrier_type):
    # Filter the dataframe using boolean indexing
    filtered_data = barrier_data[
        (barrier_data["occupance_type"] == occupance_type)
        & (barrier_data["specific_use_type"] == specific_use_type)
    ]

    # Find rows where barrier_type is in the barrier_type list
    matching_rows = filtered_data[
        filtered_data["barrier_type"].apply(lambda x: barrier_type in x)
    ]

    # Return the dictionary of values and units
    row_data = matching_rows.iloc[0]
    return {
        "horizontal_line_design_loading": row_data["Horizontal_design_loading"],
        "point_design_loading": row_data["Point_design_loading"],
        "infill_face_design_loading": row_data["Infill_face_design_loading"],
        "infill_point_design_loading": row_data["Infill_point_design_loading"],
    }


def get_minimum_overall_barrier_height(
    occupance_type, specific_use_type, barrier_type, dwelling_type=None
):
    # Filter the dataframe using boolean indexing
    filtered_data = barrier_data[
        (barrier_data["occupance_type"] == occupance_type)
        & (barrier_data["specific_use_type"] == specific_use_type)
    ]

    # Find rows where barrier_type is in the barrier_type list
    matching_rows = filtered_data[
        filtered_data["barrier_type"].apply(lambda x: barrier_type in x)
    ]

    # Get the height data
    height_data = matching_rows.iloc[0]["Minimum_overall_barrier_height"]

    # Handle different data formats
    if isinstance(height_data, dict):
        # Check if it's a dictionary with dwelling type keys
        if dwelling_type and dwelling_type in height_data:
            return height_data[dwelling_type]["value"]
        # Otherwise return the "value" key
        else:
            return height_data["value"]
    else:
        # Direct float value
        return height_data
