import pandas as pd

barrier_data = pd.DataFrame(
    [
        # Domestic occupancy
        {
            "occupance_type": "Domestic",
            "specific_use_type": "Pool",
            "barrier_type": ["Pool fence"],
            "Building_code_clause": "F9",
            "Horizontal_design_loading": {"value": 0.33, "unit": "kN"},
            "Minimum_overall_barrier_height": {"value": 1.2, "unit": "m"},
        },
        {
            "occupance_type": "Domestic",
            "specific_use_type": "Stairs",
            "barrier_type": ["Balustrade"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.35, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {"value": 0.9, "unit": "m"},
        },
        {
            "occupance_type": "Domestic",
            "specific_use_type": "Walkways/Landings",
            "barrier_type": ["Boundary fence", "Retaining wall"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.35, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {"value": 1.0, "unit": "m"},
        },
        {
            "occupance_type": "Domestic",
            "specific_use_type": "Decks/Terraces/Balconies",
            "barrier_type": ["Retaining wall", "Balustrade", "Boundary fence"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {
                "Single dwelling": {"value": 1.0, "unit": "m"},
                "Multi dwelling": {"value": 1.1, "unit": "m"},
            },
        },
        # Commercial/Industrial occupancy
        {
            "occupance_type": "Commercial/Industrial",
            "specific_use_type": "Stairs",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.35, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
        },
        {
            "occupance_type": "Commercial/Industrial",
            "specific_use_type": "Walkways/Landings",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.35, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
        },
        {
            "occupance_type": "Commercial/Industrial",
            "specific_use_type": "Decks/Terraces/Balconies",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
        },
        # Public occupancy
        {
            "occupance_type": "Public",
            "specific_use_type": "Walkways/Landings",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
        },
        {
            "occupance_type": "Public",
            "specific_use_type": "Decks/Terraces/Balconies",
            "barrier_type": ["Balustrade", "Boundary fence", "Retaining wall"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
        },
        {
            "occupance_type": "Public",
            "specific_use_type": "Stairs",
            "barrier_type": ["Balustrade"],
            "Building_code_clause": "F4",
            "Horizontal_design_loading": {"value": 0.75, "unit": "kN/m"},
            "Minimum_overall_barrier_height": {"value": 1.1, "unit": "m"},
        },
    ]
)


def get_horizontal_design_loading(occupance_type, specific_use_type, barrier_type):
    try:
        # Filter the dataframe using boolean indexing
        filtered_data = barrier_data[
            (barrier_data["occupance_type"] == occupance_type)
            & (barrier_data["specific_use_type"] == specific_use_type)
        ]

        # Find rows where barrier_type is in the barrier_type list
        # We need to use apply since we're checking membership in a list
        matching_rows = filtered_data[
            filtered_data["barrier_type"].apply(lambda x: barrier_type in x)
        ]

        # If we found a match, return the value
        if not matching_rows.empty:
            return matching_rows.iloc[0]["Horizontal_design_loading"]["value"]

        # If no exact match, try to find a more general match
        # For example, if specific_use_type is "Stairs" but no match found,
        # try "Stairs/Walkways/Landings"
        if specific_use_type == "Stairs" or specific_use_type == "Walkways/Landings":
            alternative_use = "Stairs/Walkways/Landings"
            alt_filtered = barrier_data[
                (barrier_data["occupance_type"] == occupance_type)
                & (barrier_data["specific_use_type"].str.contains(specific_use_type))
            ]
            alt_matching = alt_filtered[
                alt_filtered["barrier_type"].apply(lambda x: barrier_type in x)
            ]
            if not alt_matching.empty:
                return alt_matching.iloc[0]["Horizontal_design_loading"]["value"]

        # If still no match, use the most conservative value for the occupancy type
        conservative_value = (
            barrier_data[barrier_data["occupance_type"] == occupance_type][
                "Horizontal_design_loading"
            ]
            .apply(lambda x: x["value"])
            .max()
        )

        print(
            f"No exact match found for {occupance_type}, {specific_use_type}, {barrier_type}. Using conservative value: {conservative_value}"
        )
        return conservative_value

    except Exception as e:
        print(f"Error finding horizontal design loading: {e}")
        # Return a conservative default value
        return 0.75  # Conservative default in kN/m


def get_minimum_overall_barrier_height(
    occupance_type, specific_use_type, barrier_type, dwelling_type=None
):
    try:
        # Filter the dataframe using boolean indexing
        filtered_data = barrier_data[
            (barrier_data["occupance_type"] == occupance_type)
            & (barrier_data["specific_use_type"] == specific_use_type)
        ]

        # Find rows where barrier_type is in the barrier_type list
        matching_rows = filtered_data[
            filtered_data["barrier_type"].apply(lambda x: barrier_type in x)
        ]

        # If we found a match, return the value
        if not matching_rows.empty:
            height_data = matching_rows.iloc[0]["Minimum_overall_barrier_height"]

            # Handle different data formats
            if isinstance(height_data, dict):
                # Check if it's a dictionary with dwelling type keys
                if dwelling_type in height_data:
                    return height_data[dwelling_type]["value"]
                # Check if it has a "value" key
                elif "value" in height_data:
                    return height_data["value"]
            # Handle case where it's a direct float value
            elif isinstance(height_data, (int, float)):
                return height_data
            # Handle case where it's stored as Minimum_overall_barrier_height_m
            elif "Minimum_overall_barrier_height_m" in matching_rows.iloc[0]:
                return matching_rows.iloc[0]["Minimum_overall_barrier_height_m"]

        # If no exact match, try to find a more general match
        if specific_use_type in ["Stairs", "Walkways/Landings"]:
            alt_filtered = barrier_data[
                (barrier_data["occupance_type"] == occupance_type)
                & (barrier_data["specific_use_type"].str.contains(specific_use_type))
            ]
            alt_matching = alt_filtered[
                alt_filtered["barrier_type"].apply(lambda x: barrier_type in x)
            ]
            if not alt_matching.empty:
                height_data = alt_matching.iloc[0]["Minimum_overall_barrier_height"]
                if isinstance(height_data, dict):
                    if dwelling_type in height_data:
                        return height_data[dwelling_type]["value"]
                    elif "value" in height_data:
                        return height_data["value"]
                elif isinstance(height_data, (int, float)):
                    return height_data

        # If still no match, use the most conservative value for the occupancy type
        conservative_value = 1.1  # Default conservative value

        # Try to find the most conservative value from the data
        try:
            occ_filtered = barrier_data[
                barrier_data["occupance_type"] == occupance_type
            ]
            if not occ_filtered.empty:
                max_values = []
                for _, row in occ_filtered.iterrows():
                    height_data = row.get("Minimum_overall_barrier_height")
                    if isinstance(height_data, dict):
                        if "value" in height_data:
                            max_values.append(height_data["value"])
                        elif dwelling_type in height_data:
                            max_values.append(height_data[dwelling_type]["value"])
                        else:
                            # If it's a dict with multiple dwelling types, get the max
                            for dwell_type, val in height_data.items():
                                if isinstance(val, dict) and "value" in val:
                                    max_values.append(val["value"])
                    elif isinstance(height_data, (int, float)):
                        max_values.append(height_data)
                    elif "Minimum_overall_barrier_height_m" in row:
                        max_values.append(row["Minimum_overall_barrier_height_m"])

                if max_values:
                    conservative_value = max(max_values)
        except Exception as inner_e:
            print(f"Error finding conservative value: {inner_e}")

        print(
            f"No exact match found for {occupance_type}, {specific_use_type}, {barrier_type}. Using conservative value: {conservative_value}"
        )
        return conservative_value

    except Exception as e:
        print(f"Error finding minimum overall barrier height: {e}")
        return 1.1  # Conservative default in meters
