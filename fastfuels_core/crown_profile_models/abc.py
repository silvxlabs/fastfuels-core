from abc import ABC, abstractmethod


class CrownProfileModel(ABC):
    """
    Abstract base class representing a tree crown profile model.
    The crown profile model is a rotational solid that can be queried at any
    height to get a radius, or crown width, at that height.

    A profile describes one tree or many trees at once. Subclasses store their
    parameters as ``[n_trees, 1]`` arrays, so every method broadcasts over
    trees: per-tree methods return a scalar for a single tree and a
    ``(n_trees,)`` array for several.
    """

    @abstractmethod
    def get_radius_at_height(self, height):
        """
        Abstract method to get the radius of the crown at a given height.

        Returns a scalar for a single tree and height, a 1D array for a single
        tree and several heights, and a ``[n_trees, n_heights]`` array for
        several trees.
        """
        raise NotImplementedError(
            "The get_radius method must be implemented in a subclass."
        )

    @abstractmethod
    def get_max_radius(self):
        """
        Abstract method to get the maximum radius of the crown.

        Returns a scalar for a single tree and a ``(n_trees,)`` array for
        several trees.
        """
        raise NotImplementedError(
            "The get_max_radius method must be implemented in a subclass."
        )

    @abstractmethod
    def get_max_radius_height(self):
        """
        Abstract method to get the height (m) at which the crown attains its
        maximum radius.

        Returns a scalar for a single tree and a ``(n_trees,)`` array for
        several trees.
        """
        raise NotImplementedError(
            "The get_max_radius_height method must be implemented in a subclass."
        )
