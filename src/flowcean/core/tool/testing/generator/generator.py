import sys
from abc import abstractmethod
from pathlib import Path

from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.polars.environments.dataframe import DataFrame, collect


class TestcaseGenerator(IncrementalEnvironment):
    """A generator that produces test cases for a model."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the generator to its initial state."""

    def save_csv(
        self,
        path: str | Path,
        *,
        test_case_count: int | None = None,
        separator: str = ",",
    ) -> None:
        """Save the generated test cases to a CSV file.

        Args:
            path: The path where the CSV file should be saved.
                If the path does not have a suffix, '.csv' will be added.
            test_case_count: The number of test cases to save. If None, all
                available test cases will be saved. If the number of test cases
                is not defined, a ValueError will be raised.
            separator: The value separator to use in the CSV file.
                Defaults to ','.
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".csv")

        # Collect test cases and save them to a CSV file
        df = self.__collect_to_df(test_case_count)

        df.data.sink_csv(
            path,
            separator=separator,
            engine="streaming",
            include_header=True,
        )

    def save_excel(
        self,
        path: str | Path,
        *,
        test_case_count: int | None = None,
        worksheet_name: str = "Test Cases",
    ) -> None:
        """Save the generated test cases to an Excel file.

        Args:
            path: The path where the Excel file should be saved.
                If the path does not have a suffix, '.xlsx' will be added.
            test_case_count: The number of test cases to save. If None, all
                available test cases will be saved. If the number of test cases
                is not defined, a ValueError will be raised.
            worksheet_name: The name of the worksheet in the Excel file.
                Defaults to 'Test Cases'.
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".xlsx")

        # Collect test cases and save them to a XLSX file
        df = self.__collect_to_df(test_case_count)

        df.data.collect(engine="streaming").write_excel(
            workbook=path,
            worksheet=worksheet_name,
            include_header=True,
        )

    def __collect_to_df(self, n: int | None) -> DataFrame:
        """Collect the first n test cases and return them as a DataFrame."""
        # Make sure the number of test cases is defined
        if n is None and self.num_steps() is None:
            msg = (
                "Cannot save test cases to file without a defined "
                "number of cases."
            )
            raise ValueError(msg)

        n = min(
            n or sys.maxsize,
            self.num_steps() or sys.maxsize,
        )

        # Collect the data from the generator
        return collect(self, n)
