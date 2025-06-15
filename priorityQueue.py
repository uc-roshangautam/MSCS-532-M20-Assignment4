
import heapq
from dataclasses import dataclass
from typing import Optional, List
import time
import random

@dataclass
class Task:
    """
    Represents a task with various attributes for priority scheduling.
    """
    task_id: int
    priority: int
    arrival_time: float
    deadline: float
    name: str = ""
    estimated_duration: float = 0.0
    
    def __lt__(self, other):
        """For max-heap, we want higher priority values to come first."""
        return self.priority > other.priority
    
    def __eq__(self, other):
        return self.task_id == other.task_id
    
    def __repr__(self):
        return f"Task(id={self.task_id}, priority={self.priority}, name='{self.name}')"

class PriorityQueue:
    """
    Binary heap-based priority queue implementation using array representation.
    
    Uses max-heap where higher priority values indicate higher priority tasks.
    """
    
    def __init__(self):
        self.heap = []  # Array-based binary heap
        self.task_positions = {}  # Maps task_id to index for O(1) lookup
        self.size = 0
    
    def _parent(self, i):
        """Get parent index."""
        return (i - 1) // 2
    
    def _left_child(self, i):
        """Get left child index."""
        return 2 * i + 1
    
    def _right_child(self, i):
        """Get right child index."""
        return 2 * i + 2
    
    def _swap(self, i, j):
        """Swap elements at indices i and j and update position mapping."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        # Update position mapping
        self.task_positions[self.heap[i].task_id] = i
        self.task_positions[self.heap[j].task_id] = j
    
    def _heapify_up(self, i):
        """
        Restore heap property by moving element up.
        Time Complexity: O(log n)
        """
        parent = self._parent(i)
        if i > 0 and self.heap[i].priority > self.heap[parent].priority:
            self._swap(i, parent)
            self._heapify_up(parent)
    
    def _heapify_down(self, i):
        """
        Restore heap property by moving element down.
        Time Complexity: O(log n)
        """
        largest = i
        left = self._left_child(i)
        right = self._right_child(i)
        
        if left < self.size and self.heap[left].priority > self.heap[largest].priority:
            largest = left
        
        if right < self.size and self.heap[right].priority > self.heap[largest].priority:
            largest = right
        
        if largest != i:
            self._swap(i, largest)
            self._heapify_down(largest)
    
    def insert(self, task: Task):
        """
        Insert a new task into the priority queue.
        Time Complexity: O(log n)
        """
        if task.task_id in self.task_positions:
            raise ValueError(f"Task with ID {task.task_id} already exists")
        
        # Add task to end of heap
        self.heap.append(task)
        self.task_positions[task.task_id] = self.size
        self.size += 1
        
        # Restore heap property
        self._heapify_up(self.size - 1)
    
    def extract_max(self) -> Optional[Task]:
        """
        Remove and return the task with highest priority.
        Time Complexity: O(log n)
        """
        if self.is_empty():
            return None
        
        # Store the maximum task
        max_task = self.heap[0]
        
        # Move last element to root
        last_task = self.heap[self.size - 1]
        self.heap[0] = last_task
        self.task_positions[last_task.task_id] = 0
        
        # Remove the last element
        self.heap.pop()
        del self.task_positions[max_task.task_id]
        self.size -= 1
        
        # Restore heap property if heap is not empty
        if self.size > 0:
            self._heapify_down(0)
        
        return max_task
    
    def increase_key(self, task_id: int, new_priority: int):
        """
        Increase the priority of an existing task.
        Time Complexity: O(log n)
        """
        if task_id not in self.task_positions:
            raise ValueError(f"Task with ID {task_id} not found")
        
        index = self.task_positions[task_id]
        old_priority = self.heap[index].priority
        
        if new_priority <= old_priority:
            raise ValueError("New priority must be greater than current priority")
        
        # Update priority
        self.heap[index].priority = new_priority
        
        # Restore heap property
        self._heapify_up(index)
    
    def decrease_key(self, task_id: int, new_priority: int):
        """
        Decrease the priority of an existing task.
        Time Complexity: O(log n)
        """
        if task_id not in self.task_positions:
            raise ValueError(f"Task with ID {task_id} not found")
        
        index = self.task_positions[task_id]
        old_priority = self.heap[index].priority
        
        if new_priority >= old_priority:
            raise ValueError("New priority must be less than current priority")
        
        # Update priority
        self.heap[index].priority = new_priority
        
        # Restore heap property
        self._heapify_down(index)
    
    def peek(self) -> Optional[Task]:
        """
        Return the highest priority task without removing it.
        Time Complexity: O(1)
        """
        return self.heap[0] if not self.is_empty() else None
    
    def is_empty(self) -> bool:
        """
        Check if the priority queue is empty.
        Time Complexity: O(1)
        """
        return self.size == 0
    
    def get_size(self) -> int:
        """Get the number of tasks in the queue."""
        return self.size
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """Get a task by its ID."""
        if task_id in self.task_positions:
            index = self.task_positions[task_id]
            return self.heap[index]
        return None
    
    def display(self):
        """Display the current state of the priority queue."""
        if self.is_empty():
            print("Priority Queue is empty")
            return
        
        print("Priority Queue contents (by priority):")
        for i, task in enumerate(self.heap):
            print(f"  {i}: {task}")

class TaskScheduler:
    """
    Task scheduler using priority queue for efficient task management.
    """
    
    def __init__(self):
        self.pq = PriorityQueue()
        self.completed_tasks = []
        self.current_time = 0.0
    
    def add_task(self, task: Task):
        """Add a new task to the scheduler."""
        self.pq.insert(task)
        print(f"Added task: {task}")
    
    def execute_next_task(self):
        """Execute the highest priority task."""
        task = self.pq.extract_max()
        if task:
            print(f"Executing task: {task}")
            self.current_time += task.estimated_duration
            self.completed_tasks.append(task)
            return task
        else:
            print("No tasks to execute")
            return None
    
    def update_priority(self, task_id: int, new_priority: int):
        """Update the priority of an existing task."""
        task = self.pq.get_task(task_id)
        if task:
            old_priority = task.priority
            if new_priority > old_priority:
                self.pq.increase_key(task_id, new_priority)
            elif new_priority < old_priority:
                self.pq.decrease_key(task_id, new_priority)
            print(f"Updated task {task_id} priority from {old_priority} to {new_priority}")
        else:
            print(f"Task {task_id} not found")
    
    def get_status(self):
        """Get current scheduler status."""
        print(f"\n=== Scheduler Status ===")
        print(f"Current time: {self.current_time}")
        print(f"Pending tasks: {self.pq.get_size()}")
        print(f"Completed tasks: {len(self.completed_tasks)}")
        self.pq.display()

# Performance analysis and testing
class PriorityQueueBenchmark:
    """Benchmark priority queue operations."""
    
    def __init__(self):
        self.pq = PriorityQueue()
    
    def benchmark_insertions(self, n_tasks: int):
        """Benchmark insertion operations."""
        start_time = time.time()
        
        for i in range(n_tasks):
            task = Task(
                task_id=i,
                priority=random.randint(1, 100),
                arrival_time=time.time(),
                deadline=time.time() + random.randint(1, 10),
                name=f"Task_{i}"
            )
            self.pq.insert(task)
        
        end_time = time.time()
        return end_time - start_time
    
    def benchmark_extractions(self, n_extractions: int):
        """Benchmark extraction operations."""
        start_time = time.time()
        
        for _ in range(min(n_extractions, self.pq.get_size())):
            self.pq.extract_max()
        
        end_time = time.time()
        return end_time - start_time
    
    def run_comprehensive_benchmark(self, sizes=[100, 500, 1000, 5000]):
        """Run comprehensive benchmark."""
        results = {}
        
        for size in sizes:
            print(f"\nBenchmarking with {size} tasks:")
            
            # Reset priority queue
            self.pq = PriorityQueue()
            
            # Benchmark insertions
            insert_time = self.benchmark_insertions(size)
            print(f"  Insertion time: {insert_time:.6f}s ({insert_time/size*1000:.3f}ms per task)")
            
            # Benchmark extractions
            extract_time = self.benchmark_extractions(size)
            print(f"  Extraction time: {extract_time:.6f}s ({extract_time/size*1000:.3f}ms per task)")
            
            results[size] = {
                'insert_time': insert_time,
                'extract_time': extract_time,
                'insert_per_op': insert_time / size,
                'extract_per_op': extract_time / size
            }
        
        return results

# Example usage and demonstration
if __name__ == "__main__":
    print("=== PRIORITY QUEUE DEMONSTRATION ===")
    
    # Create scheduler
    scheduler = TaskScheduler()
    
    # Create sample tasks
    tasks = [
        Task(1, 10, time.time(), time.time() + 5, "Critical System Update", 2.0),
        Task(2, 3, time.time(), time.time() + 10, "Regular Backup", 1.5),
        Task(3, 8, time.time(), time.time() + 3, "Security Scan", 3.0),
        Task(4, 1, time.time(), time.time() + 15, "Log Cleanup", 0.5),
        Task(5, 7, time.time(), time.time() + 8, "Database Optimization", 4.0),
    ]
    
    # Add tasks to scheduler
    for task in tasks:
        scheduler.add_task(task)
    
    print("\nInitial state:")
    scheduler.get_status()
    
    # Execute some tasks
    print("\n=== Executing Tasks ===")
    for i in range(3):
        scheduler.execute_next_task()
    
    print("\nAfter executing 3 tasks:")
    scheduler.get_status()
    
    # Update priority of remaining task
    print("\n=== Priority Update ===")
    scheduler.update_priority(2, 9)  # Increase backup priority
    
    scheduler.get_status()
    
    # Execute remaining tasks
    print("\n=== Executing Remaining Tasks ===")
    while not scheduler.pq.is_empty():
        scheduler.execute_next_task()
    
    scheduler.get_status()
    
    # Performance benchmarking
    print("\n=== PERFORMANCE BENCHMARKING ===")
    benchmark = PriorityQueueBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n=== TIME COMPLEXITY ANALYSIS ===")
    print("Priority Queue Operations:")
    print("- Insert: O(log n) - element bubbles up to correct position")
    print("- Extract Max: O(log n) - root removed, last element moved to root, heapify down")
    print("- Increase Key: O(log n) - element bubbles up to correct position")
    print("- Decrease Key: O(log n) - element bubbles down to correct position")
    print("- Peek: O(1) - just return root element")
    print("- Is Empty: O(1) - check size variable")
    
    print("\nSpace Complexity: O(n) - array storage for n elements")
    print("Additional space for task position mapping: O(n)")
