from qsprpred.models.metrics import Metric
from qsprpred.models.tasks import ModelTasks


class NaNR2Scorer(Metric):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.func(y_true, y_pred, sample_weight=None)
    
    def supportsTask(self, task: ModelTasks) -> bool:
        """Return true if the scorer supports the given task.

        Args:
            task (ModelTasks): Task of the model.

        Returns:
            bool: True if the scorer supports the given task.
        """
        if task in [ModelTasks.REGRESSION, ModelTasks.MULTITASK_REGRESSION]:
            return True
        else:
            return False

    @property
    def needsProbasToScore(self) -> bool:
        """Return True if the scorer needs probabilities to score.

        Returns:
            bool: True if the scorer needs probabilities to score.
        """
        return False

    @property
    def needsDiscreteToScore(self) -> bool:
        """Return True if the scorer needs discrete values to score.

        Returns:
            bool: True if the scorer needs discrete values to score.
        """
        return False

    @property
    def isClassificationMetric(self) -> bool:
        """Return true if the scorer supports any type of classification tasks."""
        return False
    @property
    def isRegressionMetric(self) -> bool:
        """Return true if the scorer supports any type of regression tasks."""
        return True