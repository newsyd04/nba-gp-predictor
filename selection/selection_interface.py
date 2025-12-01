from abc import ABC, abstractmethod

class ISelection(ABC):

    @abstractmethod
    def create_parents_pool(self, pop: list) -> list:
        """Creates a pool of parents from the population using a selection method."""
        pass
