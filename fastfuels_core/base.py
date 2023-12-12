# External Imports
import geopandas as gpd
from pandera import DataFrameSchema
from geopandas import GeoDataFrame


class ObjectIterableGeoDataFrame:
    schema: DataFrameSchema
    data: GeoDataFrame

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
