import numpy as np
import random

from fastfuels_core.fuel_moisture.fosberg import (
    calculate_1hr_fuel_moisture,
    _calculate_reference_fuel_moisture,  # noqa
    _calculate_1hr_fuel_moisture_corrections,  # noqa
    _convert_aspect_to_cardinal,  # noqa
    TEMP_BREAKPOINTS,
    RH_BREAKPOINTS,
    RFM_TABLE,
    TABLE_B,
    TABLE_C,
    TABLE_D,
    ShadingBoolean,
    Aspect,
    TIME_RANGES,
    SLOPE_RANGES,
)

import pytest


class TestCalculateReferenceFuelMoisture:

    @pytest.fixture(scope="class")
    def table_test_cases(self):
        cases = []
        for i in range(len(TEMP_BREAKPOINTS)):
            for j in range(len(RH_BREAKPOINTS)):
                # Get a random temperature in range
                temp_low = TEMP_BREAKPOINTS[i]
                temp_high = TEMP_BREAKPOINTS[min(i + 1, len(TEMP_BREAKPOINTS) - 1)]
                if i == len(TEMP_BREAKPOINTS) - 1:
                    temp = random.uniform(temp_low, temp_low + 20)
                else:
                    temp = random.uniform(temp_low, temp_high)

                # Get a random RH in range
                rh_low = RH_BREAKPOINTS[j]
                rh_high = RH_BREAKPOINTS[min(j + 1, len(RH_BREAKPOINTS) - 1)]
                rh = random.uniform(rh_low, rh_high)

                expected = RFM_TABLE[i][j]

                cases.append((temp, rh, expected, i, j))

        return cases

    def test_table(self, table_test_cases):
        for temp, rh, expected, i, j in table_test_cases:
            result = _calculate_reference_fuel_moisture(temp, rh)
            assert result == expected, (
                f"Failed for temp={temp:.2f}, rh={rh:.2f}. Expected {expected}, but got {result}. "
                f"This corresponds to temperature range index {i} and RH range index {j}."
            )

    @pytest.mark.parametrize(
        "temp, rh, expected",
        [
            (10, 0, 1),  # Minimum temp, minimum RH
            (10, 100, 14),  # Minimum temp, maximum RH
            (109, 0, 1),  # Just below maximum temp range, minimum RH
            (109, 100, 13),  # Just below maximum temp range, maximum RH
            (110, 0, 1),  # Maximum temp range, minimum RH
            (110, 100, 12),  # Maximum temp range, maximum RH
            (200, 50, 7),  # Well above maximum temp range, medium RH
        ],
    )
    def test_boundary_conditions(self, temp, rh, expected):
        assert (
            _calculate_reference_fuel_moisture(temp, rh) == expected
        ), f"Boundary condition failed for temp={temp}, rh={rh}. Expected {expected}, but got {_calculate_reference_fuel_moisture(temp, rh)}"

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            _calculate_reference_fuel_moisture(9, 50)
        with pytest.raises(ValueError):
            _calculate_reference_fuel_moisture(70, -1)
        with pytest.raises(ValueError):
            _calculate_reference_fuel_moisture(70, 101)

    def test_array_inputs_outputs(self):
        temps = np.array([[75, 91], [55, 72]])
        rhs = np.array([[20, 70], [40, 10]])
        expected = np.array([[3, 9], [6, 2]])
        result = _calculate_reference_fuel_moisture(temps, rhs)
        assert np.array_equal(
            result, expected
        ), f"Failed for array inputs. Expected {expected}, but got {result}"


class TestConvertAspectToCardinal:
    test_size = 100

    @pytest.mark.parametrize(
        "aspect, expected",
        [(0, 0), (45, 1), (90, 1), (135, 2), (180, 2), (225, 3), (270, 3), (315, 0)],
    )
    def test_edges(self, aspect, expected):
        assert (
            _convert_aspect_to_cardinal(aspect) == expected
        ), f"Failed for edge case {aspect}. Expected {expected}, but got {_convert_aspect_to_cardinal(aspect)}"

    @pytest.mark.parametrize("direction", ["North", "East", "South", "West"])
    def test_directions(self, direction):
        if direction == "North":
            expected = 0
            cases = np.random.uniform(low=360 - 45, high=360 + 45, size=self.test_size)
        elif direction == "East":
            expected = 1
            cases = np.random.uniform(low=90 - 45, high=90 + 45, size=self.test_size)
        elif direction == "South":
            expected = 2
            cases = np.random.uniform(low=180 - 45, high=180 + 45, size=self.test_size)
        else:
            expected = 3
            cases = np.random.uniform(low=270 - 45, high=270 + 45, size=self.test_size)
        cases %= 360
        for case in cases:
            assert (
                _convert_aspect_to_cardinal(case) == expected
            ), f"Failed for {direction} direction in case {case}. Expected {expected}, but got {_convert_aspect_to_cardinal(case)}"

    def test_array(self):
        aspects = np.array([[0, 45, 90, 135], [180, 225, 270, 315]])
        expected = np.array([[0, 1, 1, 2], [2, 3, 3, 0]])
        result = _convert_aspect_to_cardinal(aspects)
        assert np.array_equal(
            result, expected
        ), f"Failed for array inputs. Expected {expected}, but got {result}"


class TestCalculate1HrFuelMoistureCorrections:

    @pytest.mark.parametrize("correction_table", [TABLE_B, TABLE_C, TABLE_D])
    @pytest.mark.parametrize(
        "shading_category", [ShadingBoolean.Unshaded, ShadingBoolean.Shaded]
    )
    @pytest.mark.parametrize(
        "aspect_category", [Aspect.North, Aspect.East, Aspect.South, Aspect.West]
    )
    @pytest.mark.parametrize("slope_range", [SLOPE_RANGES[0], SLOPE_RANGES[1]])
    @pytest.mark.parametrize("time_range", [*TIME_RANGES])
    @pytest.mark.parametrize("relative_elevation", [0, 1, 2])
    def test_scalar_inputs(
        self,
        correction_table,
        shading_category,
        aspect_category,
        slope_range,
        time_range,
        relative_elevation,
    ):
        shading = (
            random.uniform(0, 0.5)
            if shading_category == ShadingBoolean.Unshaded
            else random.uniform(0.5, 1)
        )
        aspect = random.uniform(aspect_category * 90 - 45, aspect_category * 90 + 45)
        slope = (
            random.uniform(0, 30)
            if slope_range == SLOPE_RANGES[0]
            else random.uniform(30, 90)
        )
        lower_time = int(time_range.split("-")[0])
        upper_time = int(time_range.split("-")[1])
        time = int(str(int(random.uniform(lower_time, upper_time))).zfill(4))

        result = _calculate_1hr_fuel_moisture_corrections(
            correction_table, shading, aspect, slope, time, relative_elevation
        )
        assert isinstance(
            result, (int, float)
        ), "Result should be a scalar value"  # assert 0 <= result <= 6, "Result should be between 0 and 6"
        assert (
            correction_table[shading_category][aspect_category][slope_range][
                time_range
            ][relative_elevation]
            == result
        )

    def test_array_inputs(self):
        shading = np.array(
            [
                [ShadingBoolean.Unshaded, ShadingBoolean.Shaded],
                [ShadingBoolean.Unshaded, ShadingBoolean.Shaded],
            ]
        )

        aspect = np.array([[Aspect.North, Aspect.South], [Aspect.East, Aspect.West]])
        slope = np.array([[15, 45], [1, 30]])
        time = 1200
        relative_elevation = 1

        result = _calculate_1hr_fuel_moisture_corrections(
            TABLE_B, shading, aspect, slope, time, relative_elevation
        )
        assert isinstance(result, np.ndarray), "Result should be a numpy array"
        assert result.shape == (2, 2), "Result shape should match input shape"

        expected = np.array(
            [
                [
                    TABLE_B[0][0][SLOPE_RANGES[0]][TIME_RANGES[2]][1],
                    TABLE_B[1][2][SLOPE_RANGES[1]][TIME_RANGES[2]][1],
                ],
                [
                    TABLE_B[0][1][SLOPE_RANGES[0]][TIME_RANGES[2]][1],
                    TABLE_B[1][3][SLOPE_RANGES[0]][TIME_RANGES[2]][1],
                ],
            ]
        )
        assert np.array_equal(
            result, expected
        ), f"Failed for array inputs. Expected {expected}, but got {result}"


class TestCalculate1HRFuelMoisture:
    def test_example_scalar(self):
        temp = 75
        rh = 30
        aspect = 180
        slope = 20
        time = 1200
        month = "June"
        elevation = 1
        shading = 0.2

        result = calculate_1hr_fuel_moisture(
            temp, rh, aspect, slope, time, month, elevation, shading
        )
        expected = 5
        assert (
            result == expected
        ), f"Failed for scalar inputs. Expected {expected}, but got {result}"

    def test_example_array(self):
        temps = np.array([80, 70])
        rhs = np.array([30, 50])
        aspects = np.array([180, 270])
        slope = 20
        time = 1200
        month = "June"
        elevation = 1
        shading = 0.2

        result = calculate_1hr_fuel_moisture(
            temps, rhs, aspects, slope, time, month, elevation, shading
        )

        expected = np.array([5, 7])
        assert np.array_equal(
            result, expected
        ), f"Failed for array inputs. Expected {expected}, but got {result}"
