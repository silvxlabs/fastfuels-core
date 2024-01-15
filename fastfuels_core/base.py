# External Imports
import geopandas as gpd
from pandas import DataFrame
from geopandas import GeoDataFrame
from pandera import DataFrameSchema


class ObjectIterableDataFrame:
    schema: DataFrameSchema
    data: DataFrame | GeoDataFrame

    def __init__(self, data):
        self.data = self.schema.validate(data)

    def __getattr__(self, name):
        """Delegate attribute and method access to the underlying dataframe."""
        return getattr(self.data, name)

    def __iter__(self):
        """Return an iterator for the object."""
        self._index = 0
        return self

    def __next__(self):
        """Return the next item in the iterator."""
        if self._index < len(self.data):
            row = self.data.iloc[self._index]
            iter_plot = self._row_to_object(row)
            self._index += 1
            return iter_plot
        else:
            raise StopIteration

    def __getitem__(self, item):
        """Return the item at the given index."""
        result = self.data[item]
        if isinstance(result, gpd.GeoSeries):
            return self.__class__(result)
        else:
            return result

    def __setitem__(self, key, value):
        """Set the item at the given key."""
        self.data[key] = value

    def __len__(self):
        """Return the length of the object."""
        return len(self.data)

    def _row_to_object(self, *args):
        """
        Convert a row of the dataframe to an object.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.

        Notes
        -----
        This method must be implemented by the subclass.
        """
        raise NotImplementedError("_row_to_object() must be implemented by subclass")

    def dataframe_to_geodataframe(self, crs=None) -> GeoDataFrame:
        """
        Convert the object to a GeoDataFrame using columns X and Y as geometry.

        Parameters
        ----------
        crs : str, optional
            The coordinate reference system to use for the GeoDataFrame.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame representation of the object.
        """
        try:
            gdf = GeoDataFrame(
                self.data,
                geometry=gpd.points_from_xy(self.data.X, self.data.Y),
                crs=crs,
            )
        except AttributeError:
            raise AttributeError(
                "Cannot convert to GeoDataFrame. DataFrame does not contain X and Y columns."
            )

        self.data = gdf.drop(columns=["X", "Y"])

    def geodataframe_to_dataframe(self) -> DataFrame:
        """
        Convert the object to a DataFrame by dropping the geometry column.

        Returns
        -------
        DataFrame
            A DataFrame representation of the object.
        """
        try:
            # Add X and Y columns to the dataframe from the geometry
            df = DataFrame(self.data)
            df["X"] = self.data.geometry.x
            df["Y"] = self.data.geometry.y
            df = df.drop(columns=["geometry"])
        except AttributeError:
            raise AttributeError(
                "Cannot convert to DataFrame. Object does not contain geometry column."
            )

        self.data = df
