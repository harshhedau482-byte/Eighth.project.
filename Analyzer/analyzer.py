"""
NumPy Analyzer - A menu-driven toolkit for NumPy array operations.

This program demonstrates NumPy functionalities integrated with
Object-Oriented Programming (OOP) principles including:
    - Class design and encapsulation
    - Constructor (__init__) for array initialization
    - Private methods for internal computations
    - @classmethod and @staticmethod for utility functions

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np


class DataAnalytics:
    """
    Encapsulates all NumPy operations for data analysis.

    Attributes:
        array (np.ndarray): The primary NumPy array being analyzed.
    """

    # Class-level attribute: tracks total instances created
    _instance_count = 0

    # ------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------
    def __init__(self, array=None):
        """
        Initialize the DataAnalytics object.

        Args:
            array (np.ndarray, optional): Initial NumPy array. Defaults to None.
        """
        self.array = np.array(array) if array is not None else None
        DataAnalytics._instance_count += 1

    # ------------------------------------------------------------
    # Class Methods and Static Methods
    # ------------------------------------------------------------
    @classmethod
    def get_instance_count(cls):
        """Class method: returns the total number of DataAnalytics instances."""
        return cls._instance_count

    @staticmethod
    def array_info(array):
        """Static method: returns basic info about a NumPy array."""
        return {
            "shape": array.shape,
            "dimensions": array.ndim,
            "size": array.size,
            "dtype": array.dtype
        }

    # ------------------------------------------------------------
    # Private Helper Methods (Encapsulation)
    # ------------------------------------------------------------
    def _ensure_array_exists(self):
        """Private: Check if an array has been created."""
        if self.array is None:
            print("Error: No array has been created yet. Please create one first.")
            return False
        return True

    def _validate_same_shape(self, other):
        """Private: Validate that two arrays have the same shape."""
        if self.array.shape != other.shape:
            print(f"Error: Array shapes do not match. "
                  f"({self.array.shape} vs {other.shape})")
            return False
        return True

    def _input_array(self, size, label="array"):
        """Private: Prompt user to enter elements and return a NumPy array."""
        while True:
            try:
                raw = input(f"Enter {size} elements for the {label} "
                            f"separated by space: ").strip()
                elements = list(map(float, raw.split()))
                if len(elements) != size:
                    print(f"Error: Expected {size} elements, got {len(elements)}.")
                    continue
                # Use int if all values are whole numbers
                if all(e.is_integer() for e in elements):
                    elements = [int(e) for e in elements]
                return np.array(elements)
            except ValueError:
                print("Error: Please enter valid numeric values.")

    def _display_array(self, arr, label="Array"):
        """Private: Display an array with a label."""
        print(f"\n{label}:")
        print(arr)

    # ------------------------------------------------------------
    # Array Creation
    # ------------------------------------------------------------
    def create_array(self):
        """Create a 1D, 2D, or 3D array based on user input."""
        print("\n--- Array Creation ---")
        print("Select the type of array to create:")
        print("1. 1D Array")
        print("2. 2D Array")
        print("3. 3D Array")
        choice = input("Enter your choice: ").strip()

        try:
            if choice == '1':
                size = int(input("Enter the number of elements: ").strip())
                self.array = self._input_array(size, "1D array")

            elif choice == '2':
                rows = int(input("Enter the number of rows: ").strip())
                cols = int(input("Enter the number of columns: ").strip())
                arr = self._input_array(rows * cols, "2D array")
                self.array = arr.reshape(rows, cols)

            elif choice == '3':
                depth = int(input("Enter the depth: ").strip())
                rows = int(input("Enter the number of rows: ").strip())
                cols = int(input("Enter the number of columns: ").strip())
                arr = self._input_array(depth * rows * cols, "3D array")
                self.array = arr.reshape(depth, rows, cols)

            else:
                print("Invalid choice.")
                return

            print("\nArray created successfully:")
            print(self.array)

        except ValueError:
            print("Error: Please enter valid integers.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def array_operations_menu(self):
        """Sub-menu for indexing and slicing."""
        if not self._ensure_array_exists():
            return

        while True:
            print("\nChoose an operation:")
            print("1. Indexing")
            print("2. Slicing")
            print("3. Go Back")
            choice = input("Enter your choice: ").strip()

            if choice == '1':
                self.index_element()
            elif choice == '2':
                self.slice_array()
            elif choice == '3':
                break
            else:
                print("Invalid option. Please try again.")

    def index_element(self):
        """Access a specific element in the array."""
        try:
            indices = input("Enter indices separated by space "
                            "(e.g., '0' for 1D, '1 2' for 2D): ").strip()
            idx = tuple(int(i) for i in indices.split())
            value = self.array[idx]
            print(f"Element at index {idx}: {value}")
        except (IndexError, ValueError):
            print("Error: Invalid index or index out of range.")

    def slice_array(self):
        """Slice the array using row/column ranges."""
        try:
            if self.array.ndim == 1:
                rng = input("Enter slice range (start:end): ").strip()
                start, end = map(int, rng.split(":"))
                sliced = self.array[start:end]
            elif self.array.ndim == 2:
                r_range = input("Enter the row range (start:end): ").strip()
                c_range = input("Enter the column range (start:end): ").strip()
                r_start, r_end = map(int, r_range.split(":"))
                c_start, c_end = map(int, c_range.split(":"))
                sliced = self.array[r_start:r_end, c_start:c_end]
            else:
                print("Slicing for 3D arrays: using first two dimensions.")
                r_range = input("Enter the row range (start:end): ").strip()
                c_range = input("Enter the column range (start:end): ").strip()
                r_start, r_end = map(int, r_range.split(":"))
                c_start, c_end = map(int, c_range.split(":"))
                sliced = self.array[:, r_start:r_end, c_start:c_end]

            print("\nSliced Array:")
            print(sliced)
        except (ValueError, IndexError):
            print("Error: Invalid slice range.")

    # ------------------------------------------------------------
    # Mathematical Operations
    # ------------------------------------------------------------
    def mathematical_operations_menu(self):
        """Sub-menu for mathematical operations."""
        if not self._ensure_array_exists():
            return

        print("\n--- Mathematical Operations ---")
        print("Choose a mathematical operation:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Dot Product (1D arrays)")
        print("6. Matrix Multiplication (2D arrays)")
        print("7. Go Back")
        choice = input("Enter your choice: ").strip()

        if choice in ('1', '2', '3', '4'):
            self._elementwise_operation(choice)
        elif choice == '5':
            self._dot_product()
        elif choice == '6':
            self._matrix_multiplication()
        elif choice == '7':
            return
        else:
            print("Invalid option.")

    def _elementwise_operation(self, choice):
        """Private: Perform element-wise operation with another array."""
        size = self.array.size
        try:
            raw = input(f"Enter the same-size array elements "
                        f"({size} elements separated by space): ").strip()
            elements = list(map(float, raw.split()))
            if len(elements) != size:
                print(f"Error: Expected {size} elements.")
                return
            if all(e.is_integer() for e in elements):
                elements = [int(e) for e in elements]
            other = np.array(elements).reshape(self.array.shape)

            if choice == '1':
                result = self.array + other
                op_name = "Addition"
            elif choice == '2':
                result = self.array - other
                op_name = "Subtraction"
            elif choice == '3':
                result = self.array * other
                op_name = "Multiplication"
            else:
                # Avoid division by zero
                if np.any(other == 0):
                    print("Error: Division by zero is not allowed.")
                    return
                result = self.array / other
                op_name = "Division"

            self._display_array(self.array, "Original Array")
            self._display_array(other, "Second Array")
            self._display_array(result, f"Result of {op_name}")

        except ValueError:
            print("Error: Please enter valid numeric values.")

    def _dot_product(self):
        """Private: Compute dot product of two 1D arrays."""
        if self.array.ndim != 1:
            print("Error: Dot product requires 1D arrays.")
            return
        try:
            raw = input(f"Enter {self.array.size} elements for the second "
                        f"array separated by space: ").strip()
            elements = list(map(float, raw.split()))
            if len(elements) != self.array.size:
                print(f"Error: Expected {self.array.size} elements.")
                return
            other = np.array(elements)
            result = np.dot(self.array, other)
            print(f"\nDot Product: {result}")
        except ValueError:
            print("Error: Please enter valid numeric values.")

    def _matrix_multiplication(self):
        """Private: Perform matrix multiplication on 2D arrays."""
        if self.array.ndim != 2:
            print("Error: Matrix multiplication requires 2D arrays.")
            return
        try:
            rows, cols = self.array.shape
            print(f"Enter elements for a {cols}x{rows} matrix "
                  f"(result will be {rows}x{rows}):")
            total = cols * rows
            raw = input(f"Enter {total} elements separated by space: ").strip()
            elements = list(map(float, raw.split()))
            if len(elements) != total:
                print(f"Error: Expected {total} elements.")
                return
            other = np.array(elements).reshape(cols, rows)
            result = np.matmul(self.array, other)
            self._display_array(self.array, "Original Matrix")
            self._display_array(other, "Second Matrix")
            self._display_array(result, "Matrix Multiplication Result")
        except ValueError:
            print("Error: Please enter valid numeric values.")

    # ------------------------------------------------------------
    # Combining and Splitting
    # ------------------------------------------------------------
    def combine_split_menu(self):
        """Sub-menu for combining and splitting arrays."""
        if not self._ensure_array_exists():
            return

        print("\n--- Combine or Split Arrays ---")
        print("Choose an option:")
        print("1. Combine Arrays")
        print("2. Split Array")
        print("3. Go Back")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            self.combine_arrays()
        elif choice == '2':
            self.split_array()
        elif choice == '3':
            return
        else:
            print("Invalid option.")

    def combine_arrays(self):
        """Combine the current array with another array."""
        try:
            size = self.array.size
            raw = input(f"Enter the elements of another array to combine "
                        f"({size} elements separated by space): ").strip()
            elements = list(map(float, raw.split()))
            if len(elements) != size:
                print(f"Error: Expected {size} elements.")
                return
            if all(e.is_integer() for e in elements):
                elements = [int(e) for e in elements]
            other = np.array(elements).reshape(self.array.shape)

            self._display_array(self.array, "Original Array")
            self._display_array(other, "Second Array")

            if self.array.ndim == 1:
                combined = np.concatenate((self.array, other))
                print("\nCombined Array (Concatenate):")
            else:
                combined = np.vstack((self.array, other))
                print("\nCombined Array (Vertical Stack):")
            print(combined)

        except ValueError:
            print("Error: Please enter valid numeric values.")

    def split_array(self):
        """Split the array into smaller arrays."""
        try:
            if self.array.ndim == 1:
                parts = int(input("Enter number of equal parts to split into: ").strip())
                if self.array.size % parts != 0:
                    print("Error: Array size is not divisible by number of parts.")
                    return
                result = np.split(self.array, parts)
                print("\nSplit Arrays:")
                for i, sub in enumerate(result, 1):
                    print(f"  Part {i}: {sub}")
            else:
                axis = int(input("Enter axis to split along (0=rows, 1=columns): ").strip())
                parts = int(input("Enter number of equal parts: ").strip())
                if self.array.shape[axis] % parts != 0:
                    print(f"Error: Axis size {self.array.shape[axis]} "
                          f"is not divisible by {parts}.")
                    return
                result = np.split(self.array, parts, axis=axis)
                print("\nSplit Arrays:")
                for i, sub in enumerate(result, 1):
                    print(f"  Part {i}:")
                    print(sub)
        except ValueError:
            print("Error: Please enter valid integers.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # ------------------------------------------------------------
    # Search, Sort, and Filter
    # ------------------------------------------------------------
    def search_sort_filter_menu(self):
        """Sub-menu for search, sort, and filter operations."""
        if not self._ensure_array_exists():
            return

        print("\n--- Search, Sort, and Filter ---")
        print("Choose an option:")
        print("1. Search a value")
        print("2. Sort the array")
        print("3. Filter values")
        print("4. Go Back")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            self.search_value()
        elif choice == '2':
            self.sort_array()
        elif choice == '3':
            self.filter_array()
        elif choice == '4':
            return
        else:
            print("Invalid option.")

    def search_value(self):
        """Search for a specific value in the array."""
        try:
            value = float(input("Enter the value to search: ").strip())
            if value.is_integer():
                value = int(value)
            matches = np.where(self.array == value)
            if len(matches[0]) == 0:
                print(f"Value {value} not found in the array.")
            else:
                print(f"Value {value} found at indices: {list(zip(*matches))}")
        except ValueError:
            print("Error: Please enter a valid numeric value.")

    def sort_array(self):
        """Sort the array in ascending or descending order."""
        print("\nSort order:")
        print("1. Ascending")
        print("2. Descending")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            sorted_arr = np.sort(self.array, axis=-1)
            label = "Sorted Array (Ascending)"
        elif choice == '2':
            sorted_arr = np.sort(self.array, axis=-1)[..., ::-1]
            label = "Sorted Array (Descending)"
        else:
            print("Invalid choice.")
            return

        self._display_array(self.array, "Original Array")
        print(f"\n{label}:")
        print(sorted_arr)
        if self.array.ndim > 1:
            print("(Sorting applied row-wise.)")

    def filter_array(self):
        """Filter array elements based on a condition."""
        print("\nFilter conditions:")
        print("1. Greater than (>)")
        print("2. Less than (<)")
        print("3. Equal to (==)")
        print("4. Not equal to (!=)")
        print("5. Greater than or equal to (>=)")
        print("6. Less than or equal to (<=)")
        choice = input("Enter your choice: ").strip()

        try:
            threshold = float(input("Enter the threshold value: ").strip())
            if threshold.is_integer():
                threshold = int(threshold)

            ops = {
                '1': self.array > threshold,
                '2': self.array < threshold,
                '3': self.array == threshold,
                '4': self.array != threshold,
                '5': self.array >= threshold,
                '6': self.array <= threshold,
            }

            if choice not in ops:
                print("Invalid choice.")
                return

            mask = ops[choice]
            filtered = self.array[mask]
            self._display_array(self.array, "Original Array")
            print(f"\nFiltered Values (matching condition):")
            print(filtered if filtered.size > 0 else "No elements matched the condition.")

        except ValueError:
            print("Error: Please enter a valid numeric value.")

    # ------------------------------------------------------------
    # Aggregates and Statistics
    # ------------------------------------------------------------
    def aggregates_menu(self):
        """Sub-menu for aggregate and statistical operations."""
        if not self._ensure_array_exists():
            return

        while True:
            print("\n--- Aggregates and Statistics ---")
            print("Choose an aggregate/statistical operation:")
            print("1. Sum")
            print("2. Mean")
            print("3. Median")
            print("4. Standard Deviation")
            print("5. Variance")
            print("6. Minimum")
            print("7. Maximum")
            print("8. Percentile")
            print("9. Correlation Coefficient")
            print("10. Go Back")
            choice = input("Enter your choice: ").strip()

            self._display_array(self.array, "Original Array")

            if choice == '1':
                print(f"\nSum of Array: {np.sum(self.array)}")
            elif choice == '2':
                print(f"\nMean of Array: {np.mean(self.array)}")
            elif choice == '3':
                print(f"\nMedian of Array: {np.median(self.array)}")
            elif choice == '4':
                print(f"\nStandard Deviation: {np.std(self.array):.4f}")
            elif choice == '5':
                print(f"\nVariance: {np.var(self.array):.4f}")
            elif choice == '6':
                print(f"\nMinimum Value: {np.min(self.array)}")
            elif choice == '7':
                print(f"\nMaximum Value: {np.max(self.array)}")
            elif choice == '8':
                self._compute_percentile()
            elif choice == '9':
                self._compute_correlation()
            elif choice == '10':
                break
            else:
                print("Invalid option. Please try again.")

    def _compute_percentile(self):
        """Private: Compute percentile of the array."""
        try:
            p = float(input("Enter the percentile (0-100): ").strip())
            if not 0 <= p <= 100:
                print("Error: Percentile must be between 0 and 100.")
                return
            value = np.percentile(self.array, p)
            print(f"\n{p}th Percentile: {value:.4f}")
        except ValueError:
            print("Error: Please enter a valid number.")

    def _compute_correlation(self):
        """Private: Compute correlation between two 1D arrays."""
        print("Note: Correlation requires two 1D arrays of equal length.")
        try:
            size = int(input("Enter the size of the second array: ").strip())
            raw = input(f"Enter {size} elements separated by space: ").strip()
            elements = list(map(float, raw.split()))
            if len(elements) != size:
                print(f"Error: Expected {size} elements.")
                return
            other = np.array(elements)

            flat_self = self.array.flatten()
            if flat_self.size != other.size:
                print("Error: Array sizes do not match.")
                return

            corr = np.corrcoef(flat_self, other)[0, 1]
            print(f"\nCorrelation Coefficient: {corr:.4f}")
        except ValueError:
            print("Error: Please enter valid numeric values.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


# ============================================================
# Main Program
# ============================================================
def display_main_menu():
    """Display the main menu."""
    print("\n" + "=" * 55)
    print("       Welcome to the NumPy Analyzer!")
    print("=" * 55)
    print("Choose an option:")
    print("1. Create a NumPy Array")
    print("2. Perform Mathematical Operations")
    print("3. Combine or Split Arrays")
    print("4. Search, Sort, or Filter Arrays")
    print("5. Compute Aggregates and Statistics")
    print("6. Exit")
    print("=" * 55)


def main():
    """Main program loop."""
    print("=" * 60)
    print("Welcome to the NumPy Analyzer!")
    print("=" * 60)
    print("This toolkit demonstrates NumPy functionalities integrated")
    print("with Object-Oriented Programming principles.\n")

    analyzer = DataAnalytics()

    while True:
        display_main_menu()
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            analyzer.create_array()
            if analyzer.array is not None:
                analyzer.array_operations_menu()
        elif choice == '2':
            analyzer.mathematical_operations_menu()
        elif choice == '3':
            analyzer.combine_split_menu()
        elif choice == '4':
            analyzer.search_sort_filter_menu()
        elif choice == '5':
            analyzer.aggregates_menu()
        elif choice == '6':
            print("\nThank you for using the NumPy Analyzer! Goodbye!")
            break
        else:
            print("Invalid option. Please select a valid option from the menu.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()