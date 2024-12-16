from abc import ABC, abstractmethod


class CrownProfileModel(ABC):
    """
    Abstract base class representing a tree crown profile model.
    The crown profile model is a rotational solid that can be queried at any
    height to get a radius, or crown width, at that height.

    This abstract class provides methods to get the radius of the crown at a
    given height and to get the maximum radius of the crown.
    """

    @abstractmethod
    def get_radius_at_height(self, height):
        """
        Abstract method to get the radius of the crown at a given height.
        """
        raise NotImplementedError(
            "The get_radius method must be implemented in a subclass."
        )

    @abstractmethod
    def get_max_radius(self) -> float:
        """
        Abstract method to get the maximum radius of the crown.
        """
        raise NotImplementedError(
            "The get_max_radius method must be implemented in a subclass."
        )
