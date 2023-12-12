# Internal imports
from fastfuels_core.point_process import run_point_process
from fastfuels_core.base import ObjectIterableGeoDataFrame


# External Imports
import pandera as pa
from pandera import Check


class TreeCollection(ObjectIterableGeoDataFrame):
    schema = pa.DataFrameSchema(
        columns={
            "TreeMapID": pa.Column(int, required=False),
            "PlotID": pa.Column(int, required=False),
            "TreeID": pa.Column(int, required=False),
            "SPCD": pa.Column(int),
            "STATUSCD": pa.Column(
                int,
                checks=Check.isin([1, 2, 3]),
            ),
            "DIA": pa.Column(
                float,
                checks=[
                    Check.gt(0),
                    Check.lt(1200),
                ],
                nullable=True,
            ),
            "HT": pa.Column(
                float,
                checks=[
                    Check.gt(0),
                    Check.le(116),
                ],
                nullable=True,
            ),
            "CR": pa.Column(
                float,
                checks=Check.in_range(min_value=0, max_value=1),
                nullable=True,
            ),
            "TPA_UNADJ": pa.Column(float, nullable=True),
            "X": pa.Column(float, nullable=True),
            "Y": pa.Column(float, nullable=True),
        },
        coerce=True,
        add_missing_columns=True,
        index=pa.Index(int, unique=True),
    )

    @classmethod
    def from_fia_data(cls, data):
        """
        Create a TreeCollection from FIA data. This function takes in a
        dataframe with FIA tree data in imperial units, converts it to metric
        units, and creates a TreeCollection object.

        Columns that are converted from imperial to metric units:
            - CR: percentage to fraction
            - HT: feet to meters
            - DIA: inches to centimeters
            - TPA_UNADJ: trees per acre to trees per m^2

        Columns that are renamed:
            - PLT_CN to PlotID
            - CN to TreeID

        Parameters
        ----------
        data : DataFrame
            A dataframe containing FIA data in imperial units.

        Returns
        -------
        TreeCollection
            A TreeCollection object containing the FIA data in metric units.

        Raises
        ------
        SchemaError
            If the dataframe does not match the required schema for a
            TreeCollection.
        """
        # Create a copy of the data
        metric_data = data.copy()

        # Convert CR from percentage to fraction
        metric_data["CR"] = metric_data["CR"] / 100

        # Convert HT from feet to meters
        metric_data["HT"] = metric_data["HT"] * 0.3048

        # Convert DIA from inches to centimeters
        metric_data["DIA"] = metric_data["DIA"] * 2.54

        # Convert TPA_UNADJ from trees per acre to trees per m^2
        metric_data["TPA_UNADJ"] = metric_data["TPA_UNADJ"] / 4046.85642

        # Rename PLT_CN to PlotID, CN to TreeID
        metric_data = metric_data.rename(columns={"PLT_CN": "PlotID", "CN": "TreeID"})

        return cls(metric_data)

    def expand_to_roi(self, process, roi, **kwargs):
        """
        Expands the trees to the Region of Interest (ROI) using a specified
        point process.

        This method generates a new set of trees that fit within the ROI
        based on the characteristics of the specified point process. It
        utilizes the 'run_point_process' function from the 'point_process'
        module to create an instance of the PointProcess class, generate tree
        locations using the specified point process, and assigns these
        locations to trees.

        Parameters
        ----------
        process : str
            The type of point process to use for expanding the trees.
            Currently, only 'inhomogeneous_poisson' is supported.
        roi : geopandas.GeoDataFrame
            The Region of Interest to which the trees should be expanded. It
            should be a GeoDataFrame containing the spatial boundaries of the
            region.
        **kwargs : dict
            Additional parameters to pass to the point process. See the
            documentation for the 'point_process' module for more information.

        Returns
        -------
        Trees
            A new Trees instance where the trees have been expanded to fit
            within the ROI based on the specified point process.

        Example
        -------
        >>> import fastfuels_core as ff_core
        >>> import geopandas as gpd
        >>>
        >>> # Create a TreeMap connector object
        >>> treemap = ff_core.TreeMap(
        ...     raster_path="<raster_path>",
        ...     tree_table_path="<tree_table_path>",
        ...     cn_lookup_path="<cn_lookup_path>"
        ... )
        >>>
        >>> # Load a region of interest geojson
        >>> my_roi = gpd.read_file("<roi_geojson_path>")
        >>> my_roi = my_roi.to_crs(roi.estimate_utm_crs())
        >>>
        >>> # Extract treemap plots from the region of interest
        >>> plots = ff_core.Plots.from_treemap(treemap, my_roi)
        >>> trees = ff_core.Trees.from_treemap(treemap, plots)
        >>>
        >>> # Use the expand_to_roi method to expand the trees to the ROI
        >>> expanded_trees = trees.expand_to_roi("inhomogeneous_poisson",
        >>>                                      my_roi,
        >>>                                      plots=plots,
        >>>                                      intensity_resolution=15)
        """
        return TreeCollection(run_point_process(process, roi, self, **kwargs))
