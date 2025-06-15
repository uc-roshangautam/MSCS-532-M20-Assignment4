# heapsort_implementation.py
# Assignment 4: Heap Data Structures - Heapsort Implementation
# Complete implementation with performance analysis and comparison

import time
import random
import matplotlib.pyplot as plt

class HeapSort:
    """
    Implementation of Heapsort algorithm using max-heap.
    
    Time Complexity: O(n log n) for all cases (worst, average, best)
    Space Complexity: O(1) - in-place sorting
    """
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
    
    def heapify(self, arr, n, i):
        """
        Maintain heap property for subtree rooted at index i.
        
        Args:
            arr: Array to heapify
            n: Size of heap
            i: Root index of subtree
        """
        largest = i  # Initialize largest as root
        left = 2 * i + 1  # Left child
        right = 2 * i + 2  # Right child
        
        # Check if left child exists and is greater than root
        if left < n:
            self.comparisons += 1
            if arr[left] > arr[largest]:
                largest = left
        
        # Check if right child exists and is greater than largest so far
        if right < n:
            self.comparisons += 1
            if arr[right] > arr[largest]:
                largest = right
        
        # If largest is not root, swap and recursively heapify
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.swaps += 1
            self.heapify(arr, n, largest)
    
    def build_max_heap(self, arr):
        """
        Build max heap from unsorted array.
        
        Time Complexity: O(n)
        """
        n = len(arr)
        # Start from last non-leaf node and heapify each node
        for i in range(n // 2 - 1, -1, -1):
            self.heapify(arr, n, i)
    
    def heap_sort(self, arr):
        """
        Sort array using heapsort algorithm.
        
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        self.comparisons = 0
        self.swaps = 0
        
        n = len(arr)
        
        # Build max heap
        self.build_max_heap(arr)
        
        # Extract elements from heap one by one
        for i in range(n - 1, 0, -1):
            # Move current root to end
            arr[0], arr[i] = arr[i], arr[0]
            self.swaps += 1
            
            # Call heapify on reduced heap
            self.heapify(arr, i, 0)
        
        return arr

# Comparison sorting algorithms for benchmarking
class QuickSort:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
    
    def partition(self, arr, low, high):
        # Use random pivot to avoid worst-case O(n²) behavior
        pivot_idx = random.randint(low, high)
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            self.comparisons += 1
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                if i != j:
                    self.swaps += 1
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.swaps += 1
        return i + 1
    
    def quick_sort_helper(self, arr, low, high):
        # Use iterative approach to avoid stack overflow
        stack = [(low, high)]
        
        while stack:
            low, high = stack.pop()
            if low < high:
                pi = self.partition(arr, low, high)
                stack.append((low, pi - 1))
                stack.append((pi + 1, high))
    
    def quick_sort(self, arr):
        self.comparisons = 0
        self.swaps = 0
        if len(arr) > 1:
            self.quick_sort_helper(arr, 0, len(arr) - 1)
        return arr

class MergeSort:
    def __init__(self):
        self.comparisons = 0
        self.merges = 0
    
    def merge(self, arr, left, mid, right):
        # Create temp arrays for left and right subarrays
        left_arr = arr[left:mid + 1]
        right_arr = arr[mid + 1:right + 1]
        
        i = j = 0
        k = left
        
        # Merge the temp arrays back into arr[left:right+1]
        while i < len(left_arr) and j < len(right_arr):
            self.comparisons += 1
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i += 1
            else:
                arr[k] = right_arr[j]
                j += 1
            k += 1
            self.merges += 1
        
        # Copy remaining elements
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i += 1
            k += 1
            self.merges += 1
        
        while j < len(right_arr):
            arr[k] = right_arr[j]
            j += 1
            k += 1
            self.merges += 1
    
    def merge_sort_helper(self, arr, left, right):
        if left < right:
            mid = (left + right) // 2
            self.merge_sort_helper(arr, left, mid)
            self.merge_sort_helper(arr, mid + 1, right)
            self.merge(arr, left, mid, right)
    
    def merge_sort(self, arr):
        self.comparisons = 0
        self.merges = 0
        self.merge_sort_helper(arr, 0, len(arr) - 1)
        return arr

# Performance testing and comparison
class SortingBenchmark:
    def __init__(self):
        self.heap_sort = HeapSort()
        self.quick_sort = QuickSort()
        self.merge_sort = MergeSort()
    
    def generate_test_data(self, size, data_type="random"):
        """Generate test data of different types."""
        if data_type == "random":
            return [random.randint(1, 1000) for _ in range(size)]
        elif data_type == "sorted":
            return list(range(1, size + 1))
        elif data_type == "reverse":
            return list(range(size, 0, -1))
        elif data_type == "duplicates":
            return [random.randint(1, size // 10) for _ in range(size)]
    
    def benchmark_algorithm(self, sort_func, data):
        """Benchmark a sorting algorithm."""
        test_data = data.copy()
        start_time = time.time()
        sort_func(test_data)
        end_time = time.time()
        return end_time - start_time
    
    def run_comparison(self, sizes=[100, 500, 1000, 2000, 5000]):
        """Run comprehensive comparison of sorting algorithms."""
        data_types = ["random", "sorted", "reverse", "duplicates"]
        results = {
            "HeapSort": {dt: [] for dt in data_types},
            "QuickSort": {dt: [] for dt in data_types},
            "MergeSort": {dt: [] for dt in data_types}
        }
        
        for size in sizes:
            print(f"\nTesting with array size: {size}")
            
            for data_type in data_types:
                test_data = self.generate_test_data(size, data_type)
                
                # Test HeapSort
                heap_time = self.benchmark_algorithm(self.heap_sort.heap_sort, test_data)
                results["HeapSort"][data_type].append(heap_time)
                
                # Test QuickSort
                quick_time = self.benchmark_algorithm(self.quick_sort.quick_sort, test_data)
                results["QuickSort"][data_type].append(quick_time)
                
                # Test MergeSort
                merge_time = self.benchmark_algorithm(self.merge_sort.merge_sort, test_data)
                results["MergeSort"][data_type].append(merge_time)
                
                print(f"  {data_type:10} - Heap: {heap_time:.6f}s, Quick: {quick_time:.6f}s, Merge: {merge_time:.6f}s")
        
        return results, sizes

# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    heap_sorter = HeapSort()
    
    # Test with different array types
    test_arrays = {
        "Random": [64, 34, 25, 12, 22, 11, 90],
        "Sorted": [1, 2, 3, 4, 5, 6, 7],
        "Reverse": [7, 6, 5, 4, 3, 2, 1],
        "Duplicates": [5, 2, 8, 2, 9, 1, 5]
    }
    
    print("=== HEAPSORT TESTING ===")
    for name, arr in test_arrays.items():
        original = arr.copy()
        sorted_arr = heap_sorter.heap_sort(arr.copy())
        print(f"{name:10}: {original} -> {sorted_arr}")
        print(f"           Comparisons: {heap_sorter.comparisons}, Swaps: {heap_sorter.swaps}")
    
    # Run performance comparison
    print("\n=== PERFORMANCE COMPARISON ===")
    benchmark = SortingBenchmark()
    results, sizes = benchmark.run_comparison()
